// gate_retriever.sh --disable-rerank with systemctl, curl and sleep mocked: it removes only the
// activation link, keeps the weights, and puts the link back if the retriever doesn't come up.
// (Ported from tests/test_retriever_gate.py's ShellRollbackTests, 2026-10-27.)
use std::{fs, path::PathBuf, process::Command};

struct Run {
    code: i32,
    linked: bool,
    weights: bool,
    calls: String,
}

fn rollback(name: &str, healthy: bool, extra: &[&str]) -> Run {
    let root = std::env::temp_dir().join(format!("gate-rollback-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("models/reranker-v1")).unwrap();
    fs::create_dir_all(root.join("bin")).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../gate_retriever.sh");
    fs::copy(script, root.join("gate_retriever.sh")).unwrap();
    fs::write(root.join("models/reranker-v1/config.json"), "{}").unwrap();
    std::os::unix::fs::symlink("reranker-v1", root.join("models/production-rerank")).unwrap();
    let calls = root.join("calls");
    for (name, body) in [("systemctl", "printf '%s\\n' \"$*\" >> \"$MOCK_CALLS\"".to_string()),
                         ("curl", format!("exit {}", if healthy { 0 } else { 1 })), ("sleep", "exit 0".to_string())] {
        let p = root.join("bin").join(name);
        fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = format!("{}:{}", root.join("bin").display(), std::env::var("PATH").unwrap_or_default());
    let out = Command::new("bash").arg(root.join("gate_retriever.sh")).arg("--disable-rerank").args(extra)
        .env("PATH", path).env("MOCK_CALLS", &calls).output().unwrap();
    let run = Run {
        code: out.status.code().unwrap_or(-1),
        linked: fs::symlink_metadata(root.join("models/production-rerank")).is_ok_and(|m| m.file_type().is_symlink()),
        weights: root.join("models/reranker-v1/config.json").is_file(),
        calls: fs::read_to_string(&calls).unwrap_or_default(),
    };
    let _ = fs::remove_dir_all(&root);
    run
}

#[test]
fn healthy_rollback_removes_only_activation() {
    let r = rollback("healthy", true, &[]);
    assert_eq!(r.code, 0);
    assert!(!r.linked);
    assert!(r.weights);
    assert_eq!(r.calls.matches("restart riverguide-retriever").count(), 1);
}

#[test]
fn failed_health_restores_previous_link() {
    let r = rollback("unhealthy", false, &[]);
    assert_eq!(r.code, 1);
    assert!(r.linked);
    assert!(r.weights);
    assert_eq!(r.calls.matches("restart riverguide-retriever").count(), 2);
}

#[test]
fn mixed_rollback_and_promotion_flags_do_nothing() {
    let r = rollback("mixed", true, &["--switch"]);
    assert_eq!(r.code, 2);
    assert!(r.linked);
    assert!(r.weights);
    assert_eq!(r.calls, "");
}
