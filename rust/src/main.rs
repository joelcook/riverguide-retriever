// riverguide-lab: the lab's tools in Rust, replacing the repo's Python one piece at a time.
// Run from the repo root (runs/, logs/, lab/), as bakeoff.sh and promote_engine.sh do.
//
//   riverguide-lab verdict ENGINE...             the bake-off's survivor, picked only among fair runs,
//                                                and what promote_engine.sh may remove (verdict.rs)
//   riverguide-lab preflight ENGINE COMMAND      exit 1 if an engine's best settings serve fewer
//                                                students at once than production does (fair.rs)
//   riverguide-lab accept ENGINE LOG             its drafts' mean acceptance in LOG, or "none"
//   riverguide-lab record attempt ENGINE SETTINGS COMMAND OUTCOME
//   riverguide-lab record started ENGINE SETTINGS COMMAND MEMORY_MIB FUNDRAISING_OK LOG_BYTES
//   riverguide-lab record failed ENGINE [DIAGNOSIS]
//   riverguide-lab record crashed ENGINE STAGE LOG
//   riverguide-lab record refused ENGINE WHY
//                                                bakeoff.sh's results, runs/bake-ENGINE.json
//   riverguide-lab json FILE [KEY.KEY.0...]      one value: a string bare, a list of strings one per
//                                                line, anything else as JSON
//   riverguide-lab ollama-ps [--partial]         model names in Ollama's /api/ps reply (stdin); with
//                                                --partial, only those split onto the CPU
//   riverguide-lab patch-sglang SITE_PACKAGES    let SGLang's MTP draft share the target's embedding
//                                                and output head (patch_sglang below)
//
// Rust since 2026-10-25: bakeoff_verdict.py ported and checked against its verdict on that day's
// bake-off, plus the fairness gate that bake-off lacked. SGLang was asked for one conversation at a
// time while Ollama served four, its DFlash2 drafter accepted no drafts, and both were deleted as
// losers. Now a run like that is reported as unfair and its engine is kept for a re-run.
//   cargo install --locked --path rust --root ~/.local
mod fair;
mod verdict;

use serde_json::{json, Value};
use std::{fs, io::Read, io::Write, process::ExitCode};

type Res = Result<u8, String>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match a.as_slice() {
        ["verdict", engines @ ..] if !engines.is_empty() => verdict::run(engines),
        ["preflight", engine, command] => fair::preflight(engine, command),
        ["accept", engine, log] => {
            match fair::acceptance(fair::family(engine), &read_lossy(log)) {
                Some(a) => println!("{} {:.2}", if fair::accepted(&a) { "ok" } else { "low" }, match a { fair::Accept::Len(x) | fair::Accept::Rate(x) => x }),
                None => println!("none"),
            }
            Ok(0)
        }
        ["record", rest @ ..] => record(rest),
        ["json", file] => json_get(file, ""),
        ["json", file, key] => json_get(file, key),
        ["ollama-ps", rest @ ..] => ollama_ps(rest.contains(&"--partial")),
        ["patch-sglang", site] => patch_sglang(site),
        _ => {
            eprintln!("usage: riverguide-lab verdict|preflight|accept|record|json|ollama-ps|patch-sglang (see rust/src/main.rs)");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("riverguide-lab: {e}");
            ExitCode::from(1)
        }
    }
}

pub fn read_json(path: &str) -> Result<Value, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("{path}: {e}"))
}

pub fn write_json(path: &str, v: &Value) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    fs::write(path, s).map_err(|e| format!("{path}: {e}"))
}

pub fn read_lossy(path: &str) -> String {
    fs::read(path).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

/// The last `n` characters, as Python's s[-n:] gives them.
pub fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    s.chars().skip(count.saturating_sub(n)).collect()
}

fn attempts_path(engine: &str) -> String {
    format!("runs/.bake-{engine}.attempts.jsonl")
}

/// The attempts bakeoff.sh noted for this engine, best settings first; the note file goes.
fn take_attempts(engine: &str) -> Vec<Value> {
    let path = attempts_path(engine);
    let out = read_lossy(&path).lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    let _ = fs::remove_file(&path);
    out
}

/// Each settings' log tail, as bakeoff.sh recorded an engine that didn't start.
fn attempt_logs(engine: &str) -> Vec<String> {
    let prefix = format!("bake-{engine}-s");
    let mut names: Vec<String> = fs::read_dir("logs")
        .map(|d| d.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect())
        .unwrap_or_default();
    names.retain(|n| n.starts_with(&prefix) && n.ends_with(".log"));
    names.sort();
    names.into_iter().map(|n| format!("logs/{n}")).collect()
}

fn parse<T: std::str::FromStr>(what: &str, s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("{what} must be a number, not {s:?}"))
}

fn record(a: &[&str]) -> Res {
    match a {
        ["attempt", engine, settings, command, outcome] => {
            let line = json!({"settings": parse::<u64>("SETTINGS", settings)?, "command": command, "outcome": outcome});
            let mut f = fs::OpenOptions::new().create(true).append(true).open(attempts_path(engine)).map_err(|e| e.to_string())?;
            writeln!(f, "{line}").map_err(|e| e.to_string())?;
        }
        ["started", engine, settings, command, memory, fund, log_bytes] => {
            let mut v = json!({"engine": engine, "started": true, "settings": parse::<u64>("SETTINGS", settings)?,
                "command": command, "memory_mib": parse::<u64>("MEMORY_MIB", memory)?, "fundraising_ok": *fund == "true"});
            // How much of the server's log covers the golden set and the load test: the fairness
            // gate reads only that part (fundraising's call comes after, and is only reported).
            if let Ok(n) = log_bytes.parse::<u64>() {
                v["log_bytes"] = json!(n);
            }
            v["attempts"] = json!(take_attempts(engine));
            write_json(&format!("runs/bake-{engine}.json"), &v)?;
        }
        ["failed", engine, note @ ..] if note.len() <= 1 => {
            let why: serde_json::Map<String, Value> =
                attempt_logs(engine).into_iter().map(|f| { let t = tail_chars(&read_lossy(&f), 400); (f, json!(t)) }).collect();
            let mut v = json!({"engine": engine, "started": false, "why": why, "attempts": take_attempts(engine)});
            if let Some(n) = note.first().filter(|n| !n.is_empty()) {
                v["diagnosis"] = json!(n); // what bakeoff.sh learned about why (diagnose in bakeoff.sh)
            }
            write_json(&format!("runs/bake-{engine}.json"), &v)?;
        }
        ["crashed", engine, stage, log] => {
            let v = json!({"engine": engine, "started": false,
                "why": {format!("crashed during {stage}"): tail_chars(&read_lossy(log), 400)}, "attempts": take_attempts(engine)});
            write_json(&format!("runs/bake-{engine}.json"), &v)?;
        }
        ["refused", engine, why] => {
            let v = json!({"engine": engine, "started": false, "refused": why, "why": {"preflight": why}, "attempts": take_attempts(engine)});
            write_json(&format!("runs/bake-{engine}.json"), &v)?;
        }
        _ => return Err("record attempt|started|failed|crashed|refused: see rust/src/main.rs".into()),
    }
    Ok(0)
}

fn json_get(file: &str, key: &str) -> Res {
    let root = read_json(file)?;
    let mut v = &root;
    for k in key.split('.').filter(|k| !k.is_empty()) {
        v = match v {
            Value::Array(a) => k.parse::<usize>().ok().and_then(|i| a.get(i)),
            Value::Object(o) => o.get(k),
            _ => None,
        }
        .ok_or_else(|| format!("{file} has no {key}"))?;
    }
    match v {
        Value::String(s) => println!("{s}"),
        Value::Array(a) if a.iter().all(Value::is_string) => a.iter().filter_map(Value::as_str).for_each(|s| println!("{s}")),
        other => println!("{other}"),
    }
    Ok(0)
}

/// SGLang builds its Qwen3.5-family MTP draft with a full 16-bit embedding and output head (248K x
/// 5120 each, 2.37 GiB apiece), loads the target's embedding into it, then throws both away for the
/// target's own tensors. On a 24 GB card that brief 4.7 GiB peak is what stops it starting. With
/// RIVERGUIDE_MTP_SHARED_EMBED=1 in the environment, the patched SGLang builds both on the meta
/// device, so nothing is allocated before the share; unset, it behaves exactly as shipped.
/// Idempotent: bakeoff.sh runs it after every install. (Was engine_patches/sglang_mtp_shared_embed.py.)
fn patch_sglang(site: &str) -> Res {
    const GATE: &str = r#"(torch.device("meta") if is_nextn and os.environ.get("RIVERGUIDE_MTP_SHARED_EMBED") else __import__("contextlib").nullcontext())"#;
    let models = std::path::Path::new(site).join("sglang/srt/models");
    let edits = [
        ("qwen3_5.py", "        self.embed_tokens = self._build_embed_tokens(config)\n".to_string(),
         format!("        with {GATE}:  # riverguide: MTP draft shares the target's\n            self.embed_tokens = self._build_embed_tokens(config)\n")),
        ("qwen3_5_mtp.py", "                self.lm_head = ParallelLMHead(\n".to_string(),
         "                with (torch.device(\"meta\") if __import__(\"os\").environ.get(\"RIVERGUIDE_MTP_SHARED_EMBED\") else __import__(\"contextlib\").nullcontext()):  # riverguide: shares the target's\n                  self.lm_head = ParallelLMHead(\n".to_string()),
    ];
    for (name, old, new) in edits {
        let path = models.join(name);
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if text.contains("riverguide:") {
            println!("{name}: already patched");
            continue;
        }
        let n = text.matches(old.as_str()).count();
        if n != 1 {
            return Err(format!("{name}: expected one match for {old:?}, found {n}"));
        }
        fs::write(&path, text.replacen(old.as_str(), &new, 1)).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("{name}: patched");
    }
    Ok(0)
}

fn ollama_ps(partial: bool) -> Res {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::Null); // Ollama down: nothing loaded
    for m in v["models"].as_array().into_iter().flatten() {
        let (size, vram) = (m["size"].as_u64().unwrap_or(0), m["size_vram"].as_u64().unwrap_or(0));
        if let Some(name) = m["name"].as_str().filter(|_| !partial || vram < size) {
            println!("{name}");
        }
    }
    Ok(0)
}
