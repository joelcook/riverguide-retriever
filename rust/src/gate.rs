// The retriever gate's promotion decisions (ported from retriever_gate.py, 2026-10-27).
//
//   riverguide-lab gate prepare PLAN --candidate NAME     freeze a committed plan before measuring;
//                                                         prints the experiment's manifest.json
//   riverguide-lab gate add MANIFEST --arm production|candidate --round N --artifact FILE
//   riverguide-lab gate evaluate MANIFEST                 the verdict, as JSON
//   riverguide-lab gate check-switch RESULT               exit 0 only if RESULT may switch production now
//   (prepare and check-switch also take --riverguide DIR, default /home/irsc/Projects/riverguide)
//
// gate_retriever.sh owns the services; this owns the data and the decision, with no model calls.
// A committed plan names one objective (quality, latency, cost or reliability) and its minimum gain
// before anything is measured. The candidate wins only with that gain overall and in every paired
// round, every guardrail held (docs/retriever-promotions.md). A switch replays the raw results and
// checks that the setup still matches what was measured.
//
// The port wrote the same manifests and verdicts as the Python, byte for byte (Python's float repr
// and \u escapes, statistics.mean's exactly rounded average, its error messages), on 813 generated
// and real experiments before retriever_gate.py was deleted. Since then a run also names its judge,
// and every run of an experiment must have the same one.
use crate::verdict::py_repr;
use crate::Res;
use chrono::{DateTime, FixedOffset, NaiveDateTime, Utc};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{fs, io};

const VERSION: i64 = 2;
const PLAN_KEYS: [&str; 6] = ["candidate", "minimum_gain", "objective", "production", "production_reranker", "rounds"];
const OBJECTIVES: [&str; 4] = ["cost", "latency", "quality", "reliability"];
const GRADE_REASONS: [&str; 7] = ["missing ", "contains ", "no canonical link", "expected no tool calls",
    "answered without searching", "called ", "judge: "];
// What a switch checks is still the code that measured: a changed file means measure again.
const SOURCES: [&str; 4] = ["retriever_service.py", "rerank.py", "rust/src/gate.rs", "gate_retriever.sh"];
const RIVERGUIDE: &str = "/home/irsc/Projects/riverguide";
const USAGE: &str = "usage: riverguide-lab gate prepare|add|evaluate|check-switch PATH [--riverguide DIR] [--candidate NAME] \
    [--arm production|candidate] [--round N] [--artifact FILE] (see rust/src/gate.rs)";

type R<T> = Result<T, String>;

pub fn run(a: &[&str]) -> Res {
    let usage = |why: &str| {
        eprintln!("{USAGE}\nriverguide-lab gate: {why}");
        Ok(2)
    };
    let Some((&action, rest)) = a.split_first() else { return usage("an action is required") };
    let (mut path, mut opts) = (None, HashMap::new());
    let mut i = 0;
    while i < rest.len() {
        match rest[i] {
            k @ ("--riverguide" | "--candidate" | "--arm" | "--round" | "--artifact") => {
                let Some(v) = rest.get(i + 1) else { return usage(&format!("{k} takes a value")) };
                opts.insert(k, *v);
                i += 2;
            }
            p if path.is_none() && !p.starts_with("--") => {
                path = Some(PathBuf::from(p));
                i += 1;
            }
            other => return usage(&format!("unexpected {other:?}")),
        }
    }
    let Some(path) = path else { return usage("a path is required") };
    let riverguide = PathBuf::from(opts.get("--riverguide").copied().unwrap_or(RIVERGUIDE));
    let root = fs::canonicalize(".").map_err(|e| e.to_string())?;
    let done = |r: R<()>| r.map(|_| 0).map_err(|e| format!("retriever gate: {e}"));
    match action {
        "prepare" => {
            let Some(candidate) = opts.get("--candidate") else { return usage("prepare needs --candidate") };
            done(prepare(&root, &riverguide, &path, candidate).map(|m| println!("{}", m.display())))
        }
        "add" => {
            let (Some(arm), Some(round), Some(artifact)) = (opts.get("--arm"), opts.get("--round"), opts.get("--artifact")) else {
                return usage("add needs --arm, --round, --artifact");
            };
            if !["production", "candidate"].contains(arm) {
                return usage("--arm must be production or candidate");
            }
            let Ok(round) = round.trim().parse::<i64>() else { return usage("--round must be a whole number") };
            if round == 0 {
                return usage("add needs --arm, --round, --artifact");
            }
            done(add(&path, arm, round, Path::new(artifact)))
        }
        "evaluate" => done(evaluate_file(&path).map(|v| println!("{}", dumps(&v)))),
        "check-switch" => done(check_switch(&root, &riverguide, &path).map(|s| println!("{s}"))),
        _ => usage("the action must be prepare, add, evaluate or check-switch"),
    }
}

// ---------- files, git, and Python's view of them ----------

/// An OSError as Python prints one: "[Errno 2] No such file or directory: '/path'".
fn os(path: &Path) -> impl Fn(io::Error) -> String + '_ {
    move |e| match e.raw_os_error() {
        Some(n) => {
            let text = e.to_string();
            let text = text.split(" (os error").next().unwrap_or(&text).to_string();
            format!("[Errno {n}] {text}: {}", py_repr(&json!(path.to_string_lossy())))
        }
        None => e.to_string(),
    }
}

fn read(path: &Path) -> R<String> {
    fs::read_to_string(path).map_err(os(path))
}

fn load(path: &Path) -> R<Value> {
    serde_json::from_str(&read(path)?).map_err(|e| e.to_string())
}

fn save(path: &Path, value: &Value) -> R<()> {
    fs::write(path, dumps(value) + "\n").map_err(os(path))
}

fn digest(path: &Path) -> R<String> {
    let bytes = fs::read(path).map_err(os(path))?;
    Ok(Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect())
}

/// Python's str.strip(): its whitespace is Unicode's plus the four ASCII separators \x1c-\x1f.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

fn git_command(root: &Path, args: &[&str]) -> (Command, Value) {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(args).stderr(Stdio::inherit());
    let mut shown = vec![json!("git"), json!("-C"), json!(root.to_string_lossy())];
    shown.extend(args.iter().map(|a| json!(a)));
    (command, Value::Array(shown))
}

/// subprocess.check_output's CalledProcessError, as Python prints it.
fn failed(shown: &Value, status: std::process::ExitStatus) -> String {
    format!("Command '{}' returned non-zero exit status {}.", py_repr(shown), status.code().unwrap_or(-1))
}

fn git(root: &Path, args: &[&str]) -> R<String> {
    let (mut command, shown) = git_command(root, args);
    let out = command.stdout(Stdio::piped()).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(failed(&shown, out.status));
    }
    Ok(py_strip(&String::from_utf8_lossy(&out.stdout)).to_string())
}

/// Python's ==, for values that came from JSON: 1 == 1.0 == true, dicts in any order.
fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w))),
        _ => match (py_number(a), py_number(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

fn py_number(v: &Value) -> Option<f64> {
    match v {
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Python's truth: empty, zero, false and None are false.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// d[k]: a missing key fails as Python's KeyError prints, 'k'.
fn key<'a>(v: &'a Value, k: &str) -> R<&'a Value> {
    v.as_object().and_then(|o| o.get(k)).ok_or_else(|| format!("'{k}'"))
}

/// d.get(k): None when missing.
fn opt<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or(&Value::Null)
}

fn is_int(v: &Value) -> bool {
    matches!(v, Value::Number(n) if n.is_i64() || n.is_u64())
}

/// A question ID as a Python set holds it: 1, 1.0 and true are the same; lists can't be one.
fn id(v: &Value) -> R<String> {
    match v {
        Value::String(s) => Ok(format!("s{s}")),
        Value::Null => Ok("null".into()),
        Value::Array(_) => Err("unhashable type: 'list'".into()),
        Value::Object(_) => Err("unhashable type: 'dict'".into()),
        _ => {
            let x = py_number(v).unwrap_or(f64::NAN);
            Ok(if x.fract() == 0.0 { format!("n{x:.0}") } else { format!("f{x:e}") })
        }
    }
}

fn ids(v: &Value) -> R<Vec<String>> {
    v.as_array().ok_or("question IDs must be a list")?.iter().map(id).collect()
}

/// datetime.fromisoformat for what the harness and prepare write; naive and aware don't compare.
enum When {
    Aware(DateTime<FixedOffset>),
    Naive(NaiveDateTime),
}

fn when(v: &Value) -> R<When> {
    let s = v.as_str().ok_or("fromisoformat: argument must be str")?;
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(When::Aware(t));
    }
    for f in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, f) {
            return Ok(When::Naive(t));
        }
    }
    Err(format!("Invalid isoformat string: {}", py_repr(&json!(s))))
}

fn earlier(a: &When, b: &When) -> R<bool> {
    match (a, b) {
        (When::Aware(x), When::Aware(y)) => Ok(x < y),
        (When::Naive(x), When::Naive(y)) => Ok(x < y),
        _ => Err("can't compare offset-naive and offset-aware datetimes".into()),
    }
}

/// datetime.now(timezone.utc).isoformat(): microseconds unless there are none.
fn iso_now(t: DateTime<Utc>) -> String {
    let micros = t.timestamp_subsec_micros();
    let base = t.format("%Y-%m-%dT%H:%M:%S");
    if micros == 0 { format!("{base}+00:00") } else { format!("{base}.{micros:06}+00:00") }
}

fn random_hex8() -> R<String> {
    let mut b = [0u8; 4];
    fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).map_err(|e| e.to_string())?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

// ---------- Python's json.dumps(indent=2), repr(float) and format(x, ".4g") ----------

pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, v, 0);
    out
}

fn write_json(out: &mut String, v: &Value, level: usize) {
    let pad = |out: &mut String, n: usize| out.push_str(&"  ".repeat(n));
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_f64() => out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN))),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_str(out, s),
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Object(o) if o.is_empty() => out.push_str("{}"),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                out.push_str(if i == 0 { "\n" } else { ",\n" });
                pad(out, level + 1);
                write_json(out, item, level + 1);
            }
            out.push('\n');
            pad(out, level);
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, item)) in o.iter().enumerate() {
                out.push_str(if i == 0 { "\n" } else { ",\n" });
                pad(out, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write_json(out, item, level + 1);
            }
            out.push('\n');
            pad(out, level);
            out.push('}');
        }
    }
}

/// ensure_ascii: everything outside ' '..'~' as \uXXXX (UTF-16), the five short escapes kept.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Python's repr(float): the shortest digits that round-trip (Rust finds the same ones), fixed
/// from 1e-4 up to 1e16, else 1e-05 / 1e+16.
fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let s = format!("{x:e}");
    let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let body = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let whole = exp as usize + 1;
            if digits.len() > whole {
                format!("{}.{}", &digits[..whole], &digits[whole..])
            } else {
                format!("{digits}{}.0", "0".repeat(whole - digits.len()))
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    format!("{sign}{body}")
}

/// format(x, ".4g"): four significant digits, trailing zeros dropped, 1e-05 below 1e-4.
fn general4(x: f64) -> String {
    const P: i32 = 4;
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let trim = |s: String| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    let s = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if x == 0.0 || (-4..P).contains(&exp) {
        trim(format!("{:.*}", (P - 1 - if x == 0.0 { 0 } else { exp }) as usize, x))
    } else {
        format!("{}e{}{:02}", trim(mantissa.to_string()), if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// str() of a JSON number as Python prints it in an f-string.
fn py_str(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_f64() => float_repr(n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Null => "None".into(),
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// statistics.mean: the exact sum of the floats over n, rounded once (Python adds them as
/// fractions). The sum is a big integer in units of the smallest float's last bit.
fn exact_mean(xs: &[f64]) -> f64 {
    let parts: Vec<(u64, i32)> = xs
        .iter()
        .map(|x| {
            let bits = x.abs().to_bits();
            let (e, frac) = ((bits >> 52) as i32, bits & ((1 << 52) - 1));
            if e == 0 { (frac, -1074) } else { (frac | 1 << 52, e - 1075) }
        })
        .collect();
    let low = parts.iter().map(|p| p.1).min().unwrap_or(0);
    let mut sum: Vec<u32> = Vec::new();
    for &(m, e) in &parts {
        let shift = (e - low) as usize;
        let (mut i, mut v, mut carry) = (shift / 32, u128::from(m) << (shift % 32), 0u128);
        while v != 0 || carry != 0 {
            if i >= sum.len() {
                sum.resize(i + 1, 0);
            }
            let s = u128::from(sum[i]) + (v & 0xffff_ffff) + carry;
            sum[i] = s as u32;
            carry = s >> 32;
            v >>= 32;
            i += 1;
        }
    }
    let n = xs.len() as u128;
    let bits = |b: &[u32]| b.iter().rposition(|&w| w != 0).map_or(0, |i| i * 32 + 32 - b[i].leading_zeros() as usize);
    let len = bits(&sum);
    if len == 0 {
        return 0.0;
    }
    // Keep enough bits that the quotient has 56: 53 kept, one to round on, and room to spare.
    let target = 56 + (128 - n.leading_zeros() as usize);
    let (top, mut exp, mut sticky) = if len <= target {
        let mut v = 0u128;
        for (i, w) in sum.iter().enumerate() {
            v |= u128::from(*w) << (32 * i);
        }
        (v << (target - len), low - (target - len) as i32, false)
    } else {
        let drop = len - target;
        let bit = |i: usize| sum.get(i / 32).is_some_and(|w| w >> (i % 32) & 1 == 1);
        let v = (0..target).fold(0u128, |v, j| v | (u128::from(bit(drop + j)) << j));
        (v, low + drop as i32, (0..drop).any(bit))
    };
    let (q, r) = (top / n, top % n);
    sticky |= r != 0;
    let drop = (128 - q.leading_zeros() as i32) - 53;
    let (mut kept, rest, half) = (q >> drop, q & ((1u128 << drop) - 1), 1u128 << (drop - 1));
    if rest > half || (rest == half && (sticky || kept & 1 == 1)) {
        kept += 1;
    }
    exp += drop;
    if kept == 1 << 53 {
        kept >>= 1;
        exp += 1;
    }
    let mut x = kept as f64;
    while exp != 0 {
        let step = exp.clamp(-1000, 1000);
        x *= f64::from_bits(((step + 1023) as u64) << 52);
        exp -= step;
    }
    x
}

// ---------- the plan and the setup it was measured on ----------

fn number(v: &Value, name: &str, minimum: f64, shown: &str) -> R<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|x| x.is_finite() && *x >= minimum),
        _ => None,
    }
    .ok_or_else(|| format!("{name} must be a finite number >= {shown}"))
}

fn validate_plan(plan: &Value) -> R<()> {
    let keys: BTreeSet<&str> = plan.as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default();
    if keys != PLAN_KEYS.into_iter().collect() {
        return Err(format!("plan must contain exactly {}", py_repr(&json!(PLAN_KEYS))));
    }
    for k in ["candidate", "production", "production_reranker"] {
        let ok = plan[k].as_str().is_some_and(|name| {
            !name.is_empty() && name != "." && name != ".." && name.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        });
        if !ok {
            return Err(format!("invalid {k} name"));
        }
    }
    let objective = plan["objective"].as_str().filter(|o| OBJECTIVES.contains(o));
    let Some(objective) = objective else {
        return Err(format!("objective must be one of {}", py_repr(&json!(OBJECTIVES))));
    };
    let rounds = &plan["rounds"];
    if !is_int(rounds) || rounds.as_i64().is_none_or(|r| r < 4 || r % 2 != 0) {
        return Err("rounds must be even and at least 4, to balance arm order".into());
    }
    let gain = number(&plan["minimum_gain"], "minimum_gain", 0.0, "0")?;
    if objective == "latency" || objective == "cost" {
        if !(0.05..1.0).contains(&gain) {
            return Err("latency/cost minimum_gain must be a fraction >= 0.05 and < 1".into());
        }
    } else if gain < 1.0 || gain.fract() != 0.0 {
        return Err("quality/reliability minimum_gain must be a whole question per round, at least 1".into());
    }
    Ok(())
}

/// A model's identity without reading gigabytes: its config's bytes, and each file's name, size and
/// modification time. None when the model isn't there.
fn model_identity(path: &Path) -> R<Value> {
    if fs::metadata(path).is_err() {
        return Ok(Value::Null);
    }
    let path = fs::canonicalize(path).map_err(os(path))?;
    let config = path.join("config.json");
    if !fs::metadata(&config).is_ok_and(|m| m.is_file()) {
        return Err(format!("no model config in {}", path.display()));
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(&path).map_err(os(&path))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let files: Vec<Value> = entries
        .iter()
        .filter_map(|p| {
            let m = fs::metadata(p).ok().filter(fs::Metadata::is_file)?;
            Some(json!([p.file_name()?.to_string_lossy(), m.len(), m.mtime() * 1_000_000_000 + m.mtime_nsec()]))
        })
        .collect();
    Ok(json!({"path": path.display().to_string(), "config_sha256": digest(&config)?, "files": files}))
}

fn state(root: &Path, riverguide: &Path, candidate: &str) -> R<Value> {
    if git(riverguide, &["rev-parse", "HEAD"])? != git(riverguide, &["rev-parse", "main"])? {
        return Err("RiverGuide's working checkout is not main".into());
    }
    let (mut command, shown) = git_command(riverguide, &["diff", "--exit-code", "HEAD", "--", "evals", "web", "prompt.md"]);
    let status = command.stdout(Stdio::null()).status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(failed(&shown, status));
    }
    let model = candidate.strip_prefix("rerank-").unwrap_or(candidate);
    let mut v = Map::new();
    v.insert("production".into(), model_identity(&root.join("models/production"))?);
    v.insert("production_reranker".into(), model_identity(&root.join("models/production-rerank"))?);
    v.insert("candidate".into(), model_identity(&root.join("models").join(model))?);
    v.insert("corpus_commit".into(), key(&load(&root.join("index/meta.json"))?, "corpus_commit")?.clone());
    v.insert("riverguide_main".into(), json!(git(riverguide, &["rev-parse", "main"])?));
    v.insert("engine_sha256".into(), json!(digest(&root.join("lab/production.env"))?));
    let mut sources = Map::new();
    for p in SOURCES {
        sources.insert(p.into(), json!(digest(&root.join(p))?));
    }
    v.insert("source_sha256".into(), Value::Object(sources));
    Ok(Value::Object(v))
}

fn name_of(identity: &Value) -> String {
    identity["path"].as_str().and_then(|p| Path::new(p).file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn prepare(root: &Path, riverguide: &Path, plan_path: &Path, candidate: &str) -> R<PathBuf> {
    let root = fs::canonicalize(root).map_err(os(root))?;
    let riverguide = fs::canonicalize(riverguide).map_err(os(riverguide))?;
    let plan_path = fs::canonicalize(plan_path).map_err(os(plan_path))?;
    let relative = plan_path
        .strip_prefix(&root)
        .map_err(|_| format!("'{}' is not in the subpath of '{}'", plan_path.display(), root.display()))?
        .to_string_lossy()
        .into_owned();
    git(&root, &["ls-files", "--error-unmatch", &relative])?;
    let committed = git(&root, &["show", &format!("HEAD:{relative}")])?;
    if committed != py_strip(&read(&plan_path)?) {
        return Err("commit the exact plan before measuring; staged or untracked plans do not qualify".into());
    }
    let plan = load(&plan_path)?;
    validate_plan(&plan)?;
    let current = state(&root, &riverguide, candidate)?;
    if plan["candidate"] != candidate || current["candidate"].is_null() {
        return Err("plan candidate does not match the model being measured".into());
    }
    if current["production"].is_null() || name_of(&current["production"]) != plan["production"] {
        return Err("plan production does not match the active retriever".into());
    }
    let rr = &current["production_reranker"];
    if (if truthy(rr) { name_of(rr) } else { "none".into() }) != plan["production_reranker"] {
        return Err("plan production_reranker does not match production".into());
    }
    if candidate.starts_with("rerank-") && truthy(rr) {
        return Err("this harness compares an added reranker to retriever-only; an existing reranker needs a separate fair comparison".into());
    }
    let golden_path = riverguide.join("evals/golden.yaml");
    let golden: Value = serde_yaml::from_str(&read(&golden_path)?).map_err(|e| e.to_string())?;
    let rows = golden.as_array().ok_or("the golden set must be a list")?;
    let judged = rows.iter().filter(|r| truthy(opt(r, "expected_summary"))).map(|r| key(r, "id").cloned()).collect::<R<Vec<_>>>()?;
    if judged.is_empty() {
        return Err("no semantic checks declared in the golden set".into());
    }
    let folder = root
        .join("runs/retriever-gates")
        .join(format!("{}-{candidate}-{}", Utc::now().format("%Y%m%dT%H%M%SZ"), random_hex8()?));
    fs::create_dir_all(folder.parent().unwrap_or(&root)).map_err(os(&folder))?;
    fs::create_dir(&folder).map_err(os(&folder))?;
    let manifest = folder.join("manifest.json");
    let mut m = Map::new();
    m.insert("version".into(), json!(VERSION));
    m.insert("prepared_at".into(), json!(iso_now(Utc::now())));
    m.insert("plan".into(), plan);
    m.insert("plan_path".into(), json!(relative));
    m.insert("plan_sha256".into(), json!(digest(&plan_path)?));
    m.insert("plan_commit".into(), json!(git(&root, &["rev-parse", "HEAD"])?));
    m.insert("root".into(), json!(root.display().to_string()));
    m.insert("state".into(), current);
    m.insert("expected_ids".into(), Value::Array(rows.iter().map(|r| key(r, "id").cloned()).collect::<R<_>>()?));
    m.insert("semantic_ids".into(), Value::Array(judged));
    m.insert("runs".into(), json!([]));
    save(&manifest, &Value::Object(m))?;
    Ok(manifest)
}

fn add(manifest_path: &Path, arm: &str, round: i64, artifact: &Path) -> R<()> {
    let mut manifest = load(manifest_path)?;
    let resolved = fs::canonicalize(artifact).map_err(os(artifact))?;
    let entry = json!({"arm": arm, "round": round, "artifact": resolved.display().to_string(), "sha256": digest(artifact)?});
    manifest.get_mut("runs").and_then(Value::as_array_mut).ok_or("'runs'")?.push(entry);
    save(manifest_path, &manifest)
}

// ---------- the measurements and the decision ----------

/// An answer that finished: text, no error, and only grading reasons. Any other reason (a fallback,
/// a dropped stream) is an operational failure, never a success.
fn completed(row: &Value) -> R<bool> {
    let response = opt(row, "response").as_str().map(py_strip).unwrap_or("");
    if response.is_empty() || truthy(opt(row, "error")) {
        return Ok(false);
    }
    let reasons = key(row, "reasons")?.as_array().ok_or("'reasons' must be a list")?;
    Ok(reasons.iter().all(|r| {
        r.as_str().is_some_and(|r| GRADE_REASONS.iter().any(|g| r.starts_with(g)) || r.contains(" not called (called:"))
    }))
}

/// A sum as Python's sum() keeps it: whole while every term is.
#[derive(Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn of(v: &Value) -> Num {
        match v.as_i64().filter(|_| is_int(v)) {
            Some(i) => Num::Int(i),
            None => Num::Float(v.as_f64().unwrap_or(f64::NAN)),
        }
    }
    fn add(self, other: Num) -> Num {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a + b),
            _ => Num::Float(self.f() + other.f()),
        }
    }
    fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(x) => x,
        }
    }
    fn json(self) -> Value {
        match self {
            Num::Int(i) => json!(i),
            Num::Float(x) => json!(x),
        }
    }
}

struct Metrics {
    total: usize,
    passed: i64,
    answer_passed: i64,
    semantic_passed: i64,
    completion_failures: i64,
    median: f64,
    p95: f64,
    mean: f64,
    tool_calls: usize,
    cost: Option<Num>,
    categories: Vec<(String, i64, i64)>,
}

impl Metrics {
    fn of(rows: &[&Value], semantic: &HashSet<String>) -> R<Metrics> {
        let mut times = rows.iter().map(|r| number(key(r, "seconds")?, "seconds", 0.000001, "1e-06")).collect::<R<Vec<f64>>>()?;
        times.sort_by(f64::total_cmp);
        // The harness's usage and cost_usd include the semantic judge's, so they can't show serving
        // cost: only separately metered fields count, positive for every answer.
        let measured = rows.iter().all(|r| {
            let usage = opt(r, "serving_usage").as_object().is_some_and(|u| u.values().any(|v| matches!(v, Value::Number(n) if n.as_f64().is_some_and(|x| x > 0.0))));
            let cost = matches!(opt(r, "serving_cost_usd"), Value::Number(n) if n.as_f64().is_some_and(|x| x.is_finite() && x > 0.0));
            usage && cost
        });
        let count = |f: &dyn Fn(&Value) -> R<bool>| rows.iter().try_fold(0i64, |n, r| Ok::<i64, String>(n + i64::from(f(r)?)));
        let names = rows.iter().map(|r| key(r, "category")?.as_str().map(str::to_string).ok_or_else(|| "categories must be strings".to_string())).collect::<R<BTreeSet<_>>>()?;
        let mut categories = Vec::new();
        for cat in names {
            let mine = |field: &str| rows.iter().filter(|r| r["category"] == cat.as_str()).map(|r| i64::from(r[field] == true)).sum::<i64>();
            categories.push((cat.clone(), mine("passed"), mine("answer_passed")));
        }
        let n = times.len();
        Ok(Metrics {
            total: rows.len(),
            passed: count(&|r| Ok(r["passed"] == true))?,
            answer_passed: count(&|r| Ok(r["answer_passed"] == true))?,
            semantic_passed: count(&|r| {
                Ok(semantic.contains(&id(key(r, "id")?)?) && completed(r)? && opt(r, "judge").is_object() && r["judge"]["pass"] == true)
            })?,
            completion_failures: count(&|r| Ok(!completed(r)?))?,
            median: if n % 2 == 1 { times[n / 2] } else { (times[n / 2 - 1] + times[n / 2]) / 2.0 },
            p95: times[(0.95 * n as f64).ceil() as usize - 1],
            mean: exact_mean(&times),
            tool_calls: rows.iter().map(|r| r["tool_calls"].as_array().map_or(0, Vec::len)).sum(),
            cost: measured.then(|| rows.iter().fold(Num::Int(0), |s, r| s.add(Num::of(&r["serving_cost_usd"])))),
            categories,
        })
    }

    fn json(&self) -> Map<String, Value> {
        let mut cats = Map::new();
        for (cat, passed, answer_passed) in &self.categories {
            cats.insert(cat.clone(), json!({"passed": passed, "answer_passed": answer_passed}));
        }
        let v = json!({"total": self.total, "passed": self.passed, "answer_passed": self.answer_passed,
            "semantic_passed": self.semantic_passed, "completion_failures": self.completion_failures,
            "median_seconds": self.median, "p95_seconds": self.p95, "mean_seconds": self.mean,
            "tool_calls": self.tool_calls, "cost_usd": self.cost.map_or(Value::Null, Num::json), "categories": cats});
        v.as_object().cloned().unwrap_or_default()
    }

    fn field(&self, name: &str) -> i64 {
        match name {
            "passed" => self.passed,
            "answer_passed" => self.answer_passed,
            _ => self.semantic_passed,
        }
    }
}

fn evaluate(manifest: &Value, artifacts: &[(Value, Value)]) -> R<Value> {
    if !py_eq(opt(manifest, "version"), &json!(VERSION)) || !truthy(opt(manifest, "prepared_at")) || !truthy(opt(manifest, "plan_commit")) {
        return Err("no predeclared version-2 experiment; legacy tolerance passes cannot authorize promotion".into());
    }
    let plan = key(manifest, "plan")?;
    validate_plan(plan)?;
    let (expected_list, semantic_list) = (ids(key(manifest, "expected_ids")?)?, ids(key(manifest, "semantic_ids")?)?);
    let expected: HashSet<String> = expected_list.iter().cloned().collect();
    let semantic: HashSet<String> = semantic_list.into_iter().collect();
    if expected.is_empty() || expected.len() != expected_list.len() || semantic.is_empty() || !semantic.is_subset(&expected) {
        return Err("invalid expected question IDs or semantic checks".into());
    }
    let rounds = plan["rounds"].as_i64().unwrap_or(0);
    let prepared = when(key(manifest, "prepared_at")?)?;
    let mut seen = HashSet::new();
    let mut signature: Option<(Value, Value, Value)> = None;
    let mut categories: Option<HashMap<String, Value>> = None;
    let mut per_round: BTreeMap<(String, i64), Metrics> = BTreeMap::new();
    let (mut production_rows, mut candidate_rows): (Vec<&Value>, Vec<&Value>) = (Vec::new(), Vec::new());
    for (spec, artifact) in artifacts {
        let (arm, rd) = (key(spec, "arm")?, key(spec, "round")?);
        let arm = arm.as_str().filter(|a| *a == "production" || *a == "candidate");
        let rd = rd.as_i64().filter(|_| is_int(rd));
        let (Some(arm), Some(rd)) = (arm, rd) else { return Err("duplicate or invalid arm/round".into()) };
        if !(1..=rounds).contains(&rd) || !seen.insert((arm.to_string(), rd)) {
            return Err("duplicate or invalid arm/round".into());
        }
        let (meta, results) = (key(artifact, "meta")?, key(artifact, "results")?);
        if opt(meta, "partial") != false || !py_eq(opt(meta, "tier"), &json!("full")) || opt(meta, "judge") != true {
            return Err("every run must be full, non-partial, and semantically judged".into());
        }
        if earlier(&when(key(meta, "started")?)?, &prepared)? {
            return Err("measurement predates the frozen plan".into());
        }
        let sig = (opt(meta, "model").clone(), opt(meta, "prompt_sha256").clone(), opt(meta, "judge_model").clone());
        let named = [&sig.0, &sig.1, &sig.2].iter().all(|v| v.as_str().is_some_and(|s| !s.is_empty()));
        if !named || signature.as_ref().is_some_and(|s| *s != sig) {
            return Err("model, prompt or judge differs between runs".into());
        }
        signature = Some(sig);
        let rows: Vec<&Value> = results.as_array().ok_or("'results' must be a list")?.iter().collect();
        let row_ids = rows.iter().map(|r| id(key(r, "id")?)).collect::<R<Vec<_>>>()?;
        if row_ids.len() != expected.len() || row_ids.iter().cloned().collect::<HashSet<_>>() != expected {
            return Err("missing, duplicate, or unexpected questions".into());
        }
        let cats = rows.iter().zip(&row_ids).map(|(r, i)| Ok((i.clone(), key(r, "category")?.clone()))).collect::<R<HashMap<_, _>>>()?;
        if let Some(before) = &categories {
            if before.len() != cats.len() || !cats.iter().all(|(k, v)| before.get(k).is_some_and(|w| py_eq(v, w))) {
                return Err("question categories differ between arms".into());
            }
        }
        categories = Some(cats);
        for (r, rid) in rows.iter().zip(&row_ids) {
            if !opt(r, "passed").is_boolean() || !opt(r, "answer_passed").is_boolean() {
                return Err("missing boolean grades".into());
            }
            let reasons_ok = opt(r, "reasons").as_array().is_some_and(|a| a.iter().all(Value::is_string));
            if !reasons_ok || !opt(r, "tool_calls").is_array() {
                return Err("missing reasons or tool trace".into());
            }
            if semantic.contains(rid) && completed(r)? && (!opt(r, "judge").is_object() || !opt(&r["judge"], "pass").is_boolean()) {
                return Err("missing semantic judgment".into());
            }
        }
        let m = Metrics::of(&rows, &semantic)?;
        let summary = key(artifact, "summary")?;
        if !py_eq(opt(summary, "passed"), &json!(m.passed)) || !py_eq(opt(summary, "total"), &json!(expected.len())) {
            return Err("summary does not match raw results".into());
        }
        per_round.insert((arm.to_string(), rd), m);
        if arm == "production" { production_rows.extend(rows) } else { candidate_rows.extend(rows) }
    }
    if seen.len() as i64 != 2 * rounds {
        return Err("incomplete experiment".into());
    }
    let (base, cand) = (Metrics::of(&production_rows, &semantic)?, Metrics::of(&candidate_rows, &semantic)?);
    let mut blockers = Vec::new();
    for k in ["passed", "answer_passed", "semantic_passed"] {
        if cand.field(k) < base.field(k) {
            blockers.push(format!("{k} regressed: {} vs {}", cand.field(k), base.field(k)));
        }
    }
    for ((cat, passed, answer_passed), (_, cp, ca)) in base.categories.iter().zip(&cand.categories) {
        if cp < passed {
            blockers.push(format!("{cat} passed regressed"));
        }
        if ca < answer_passed {
            blockers.push(format!("{cat} answer_passed regressed"));
        }
    }
    if cand.completion_failures > 0 {
        blockers.push(format!("candidate has {} incomplete/error answers", cand.completion_failures));
    }
    if cand.median > base.median * 1.05 {
        blockers.push("median_seconds regressed by more than 5%".into());
    }
    if cand.p95 > base.p95 * 1.05 {
        blockers.push("p95_seconds regressed by more than 5%".into());
    }
    if let (Some(b), Some(c)) = (base.cost, cand.cost) {
        if c.f() > b.f() * 1.05 {
            blockers.push("measured cost regressed by more than 5%".into());
        }
    }
    if base.cost.is_none() != cand.cost.is_none() {
        blockers.push("serving cost telemetry is missing from one arm".into());
    }
    let objective = plan["objective"].as_str().unwrap_or_default();
    let minimum = &plan["minimum_gain"];
    let gain = |b: &Metrics, c: &Metrics| -> R<Num> {
        match objective {
            "quality" => Ok(Num::Int(c.semantic_passed - b.semantic_passed)),
            "reliability" => Ok(Num::Int(b.completion_failures - c.completion_failures)),
            _ => {
                let (bv, cv) = if objective == "latency" { (Some(b.median), Some(c.median)) } else { (b.cost.map(Num::f), c.cost.map(Num::f)) };
                match (bv, cv) {
                    (Some(bv), Some(cv)) if bv > 0.0 => Ok(Num::Float(1.0 - cv / bv)),
                    _ => Err("cost objective requires positive, metered costs and token usage for every answer; zero placeholders do not qualify".into()),
                }
            }
        }
    };
    let mut observed = gain(&base, &cand)?;
    if let Num::Int(g) = observed {
        observed = Num::Float(g as f64 / rounds as f64);
    }
    let minimum_f = minimum.as_f64().unwrap_or(f64::NAN);
    if observed.f() + 1e-12 < minimum_f {
        blockers.push(format!("predeclared {objective} benefit not met: {} vs required {}", general4(observed.f()), py_str(minimum)));
    }
    let mut gains = Vec::new();
    for rd in 1..=rounds {
        gains.push(gain(&per_round[&("production".to_string(), rd)], &per_round[&("candidate".to_string(), rd)])?);
    }
    if gains.iter().any(|g| g.f() <= 0.0) {
        blockers.push("target benefit was not positive in every paired round".into());
    }
    let passed = blockers.is_empty();
    let summary = if passed {
        format!("predeclared {objective} benefit met ({} >= {}) with guardrails held", general4(observed.f()), py_str(minimum))
    } else {
        blockers.join("; ")
    };
    let runs: Vec<Value> = per_round
        .iter()
        .map(|((arm, rd), m)| {
            let mut v = Map::new();
            v.insert("arm".into(), json!(arm));
            v.insert("round".into(), json!(rd));
            v.extend(m.json());
            Value::Object(v)
        })
        .collect();
    Ok(json!({
        "version": VERSION, "candidate": plan["candidate"], "production": plan["production"],
        "production_reranker": plan["production_reranker"], "passed": passed,
        "verdict": if passed { "demonstrated end-to-end benefit" } else { "no demonstrated end-to-end benefit; retain incumbent" },
        "summary": summary, "objective": objective, "minimum_gain": minimum, "observed_gain": observed.json(),
        "round_gains": gains.iter().map(|g| g.json()).collect::<Vec<_>>(), "blockers": blockers,
        "metrics": {"production": base.json(), "candidate": cand.json()}, "runs": runs,
    }))
}

fn evaluate_file(manifest_path: &Path) -> R<Value> {
    let manifest = load(manifest_path)?;
    let root = PathBuf::from(py_str(key(&manifest, "root")?));
    let at = format!("{}:{}", py_str(key(&manifest, "plan_commit")?), py_str(key(&manifest, "plan_path")?));
    let committed: Value = serde_json::from_str(&git(&root, &["show", &at])?).map_err(|e| e.to_string())?;
    if !py_eq(&committed, key(&manifest, "plan")?) {
        return Err("frozen plan does not match its committed declaration".into());
    }
    let mut artifacts = Vec::new();
    for spec in key(&manifest, "runs")?.as_array().ok_or("'runs' must be a list")? {
        let path = PathBuf::from(py_str(key(spec, "artifact")?));
        if !py_eq(&json!(digest(&path)?), key(spec, "sha256")?) {
            return Err(format!("measurement changed: {}", path.display()));
        }
        artifacts.push((spec.clone(), load(&path)?));
    }
    let mut result = evaluate(&manifest, &artifacts)?;
    let resolved = fs::canonicalize(manifest_path).map_err(os(manifest_path))?;
    result["manifest"] = json!(resolved.display().to_string());
    Ok(result)
}

fn check_switch(root: &Path, riverguide: &Path, result_path: &Path) -> R<String> {
    let stored = load(result_path)?;
    if !py_eq(opt(&stored, "version"), &json!(VERSION)) || !truthy(opt(&stored, "manifest")) {
        return Err("legacy tolerance-only result cannot authorize promotion; run a predeclared experiment".into());
    }
    if opt(&stored, "passed") != true {
        return Err("candidate did not demonstrate the required benefit".into());
    }
    let manifest_path = PathBuf::from(py_str(&stored["manifest"]));
    let manifest = load(&manifest_path)?;
    let plan_path = root.join(py_str(key(&manifest, "plan_path")?));
    if !py_eq(&json!(digest(&plan_path)?), key(&manifest, "plan_sha256")?) {
        return Err("plan changed since measurement".into());
    }
    if !py_eq(&load(&plan_path)?, key(&manifest, "plan")?) {
        return Err("frozen plan differs from its declaration".into());
    }
    let candidate = py_str(key(key(&manifest, "plan")?, "candidate")?);
    if !py_eq(&state(root, riverguide, &candidate)?, key(&manifest, "state")?) {
        return Err("production, candidate, corpus, engine, or source changed; measure again".into());
    }
    let result = evaluate_file(&manifest_path)?;
    if result["passed"] != true {
        return Err(py_str(&result["summary"]));
    }
    if !py_eq(key(&stored, "candidate")?, &result["candidate"]) || !py_eq(key(&stored, "production")?, &result["production"]) {
        return Err("stored result identity does not match the experiment".into());
    }
    Ok(py_str(&result["summary"]))
}

#[cfg(test)]
mod tests {
    // Ported from tests/test_retriever_gate.py. Where Python patched git and state, these build a
    // real repo and a real model directory.
    use super::*;

    fn experiment(objective: &str, minimum_gain: Value) -> (Value, Vec<(Value, Value)>) {
        let ids: Vec<String> = (0..10).map(|i| format!("q{i}")).collect();
        let manifest = json!({
            "version": 2, "prepared_at": "2026-10-27T10:00:00+00:00", "plan_commit": "committed",
            "plan": {"candidate": "rerank-new", "production": "v2-qwen", "production_reranker": "none",
                     "objective": objective, "minimum_gain": minimum_gain, "rounds": 4},
            "expected_ids": ids, "semantic_ids": ids,
        });
        let mut artifacts = Vec::new();
        for rd in 1..=4 {
            for arm in ["production", "candidate"] {
                let rows: Vec<Value> = ids.iter().map(|id| json!({"id": id, "category": "hard_fact", "passed": true, "answer_passed": true,
                    "response": "A finished answer", "reasons": [], "judge": {"pass": true},
                    "seconds": if arm == "production" { 10 } else { 8 }, "tool_calls": [],
                    "usage": {"input_tokens": 100}, "cost_usd": 0.01,
                    "serving_usage": {"input_tokens": 100}, "serving_cost_usd": 0.01})).collect();
                let artifact = json!({"meta": {"partial": false, "tier": "full", "judge": true,
                    "started": "2026-10-27T11:00:00+00:00", "model": "same-model", "prompt_sha256": "same-prompt",
                    "judge_model": "ollama:qwen3.8:27b"},
                    "results": rows, "summary": {"passed": 10, "total": 10}});
                artifacts.push((json!({"arm": arm, "round": rd}), artifact));
            }
        }
        (manifest, artifacts)
    }

    fn alter(artifacts: &mut [(Value, Value)], arm: &str, update: impl Fn(&mut Vec<Value>)) {
        for (spec, artifact) in artifacts.iter_mut() {
            if spec["arm"] == arm {
                update(artifact["results"].as_array_mut().unwrap());
                let passed = artifact["results"].as_array().unwrap().iter().filter(|r| r["passed"] == true).count();
                artifact["summary"]["passed"] = json!(passed);
            }
        }
    }

    fn fail(row: &mut Value, reason: &str, judge: Value) {
        row["passed"] = json!(false);
        row["answer_passed"] = json!(false);
        row["reasons"] = json!([reason]);
        if !judge.is_null() {
            row["judge"] = judge;
        }
    }

    #[test]
    fn repeatable_latency_gain_with_quality_held_wins() {
        let (m, a) = experiment("latency", json!(0.15));
        let result = evaluate(&m, &a).unwrap();
        assert_eq!(result["passed"], true);
        assert!((result["observed_gain"].as_f64().unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn tie_and_smaller_gain_do_not_justify_extra_stage() {
        for seconds in [json!(10), json!(9.9), json!(9)] {
            let (m, mut a) = experiment("latency", json!(0.15));
            alter(&mut a, "candidate", |rows| rows.iter_mut().for_each(|r| r["seconds"] = seconds.clone()));
            assert_eq!(evaluate(&m, &a).unwrap()["passed"], false);
        }
    }

    #[test]
    fn faster_but_worse_quality_loses() {
        let (m, mut a) = experiment("latency", json!(0.15));
        alter(&mut a, "candidate", |rows| fail(&mut rows[0], "missing 'required fact'", json!({"pass": false})));
        let result = evaluate(&m, &a).unwrap();
        assert_eq!(result["passed"], false);
        assert!(result["blockers"].as_array().unwrap().iter().any(|b| b.as_str().unwrap().contains("regressed")));
    }

    #[test]
    fn one_fallback_blocks_an_otherwise_winning_candidate() {
        let (m, mut a) = experiment("latency", json!(0.15));
        a[1].1["results"][0]["response"] = json!("Here are fallback links");
        a[1].1["results"][0]["reasons"] = json!(["answer did not finish (finish reason 'fallback')"]);
        let result = evaluate(&m, &a).unwrap();
        assert_eq!(result["passed"], false);
        assert_eq!(result["metrics"]["candidate"]["completion_failures"], 1);
    }

    #[test]
    fn tail_regression_blocks_median_gain() {
        let (m, mut a) = experiment("latency", json!(0.15));
        alter(&mut a, "candidate", |rows| rows.last_mut().unwrap()["seconds"] = json!(20));
        assert_eq!(evaluate(&m, &a).unwrap()["passed"], false);
    }

    #[test]
    fn benefit_must_hold_in_each_round() {
        let (m, mut a) = experiment("latency", json!(0.15));
        a[1].1["results"].as_array_mut().unwrap().iter_mut().for_each(|r| r["seconds"] = json!(10));
        let result = evaluate(&m, &a).unwrap();
        assert_eq!(result["passed"], false);
        assert!(result["blockers"].as_array().unwrap().iter().any(|b| b.as_str().unwrap().contains("every paired round")));
    }

    #[test]
    fn quality_gain_requires_semantic_improvement() {
        let (m, mut a) = experiment("quality", json!(1));
        assert_eq!(evaluate(&m, &a).unwrap()["passed"], false);
        alter(&mut a, "production", |rows| fail(&mut rows[0], "judge: incomplete explanation", json!({"pass": false})));
        assert_eq!(evaluate(&m, &a).unwrap()["passed"], true);
    }

    #[test]
    fn reliability_objective_counts_completions() {
        let (m, mut a) = experiment("reliability", json!(1));
        alter(&mut a, "production", |rows| fail(&mut rows[0], "answer did not finish (finish reason 'fallback')", Value::Null));
        assert_eq!(evaluate(&m, &a).unwrap()["passed"], true);
    }

    #[test]
    fn zero_or_missing_cost_telemetry_cannot_win_cost_comparison() {
        let (m, mut a) = experiment("cost", json!(0.15));
        alter(&mut a, "candidate", |rows| rows.iter_mut().for_each(|r| {
            r["serving_cost_usd"] = json!(0);
            r["serving_usage"] = json!({});
        }));
        assert!(evaluate(&m, &a).unwrap_err().contains("metered costs"));
        let (m, mut a) = experiment("cost", json!(0.15));
        alter(&mut a, "candidate", |rows| rows.iter_mut().for_each(|r| r["serving_cost_usd"] = json!(0.008)));
        assert_eq!(evaluate(&m, &a).unwrap()["passed"], true);
    }

    #[test]
    fn judge_cost_cannot_be_mistaken_for_serving_cost() {
        let (m, mut a) = experiment("cost", json!(0.15));
        for (_, artifact) in a.iter_mut() {
            for row in artifact["results"].as_array_mut().unwrap() {
                row.as_object_mut().unwrap().remove("serving_cost_usd");
                row.as_object_mut().unwrap().remove("serving_usage");
            }
        }
        assert!(evaluate(&m, &a).unwrap_err().contains("metered costs"));
    }

    #[test]
    fn missing_or_incomparable_evidence_is_rejected() {
        let changes: Vec<Box<dyn Fn(&mut Value, &mut Vec<(Value, Value)>)>> = vec![
            Box::new(|_, a| { a.pop(); }),
            Box::new(|_, a| a[1].0["arm"] = json!("production")),
            Box::new(|_, a| { a[1].1["results"].as_array_mut().unwrap().pop(); }),
            Box::new(|_, a| a[1].1["results"][0]["id"] = json!("q1")),
            Box::new(|_, a| a[1].1["meta"]["partial"] = json!(true)),
            Box::new(|_, a| a[1].1["meta"]["judge"] = json!(false)),
            Box::new(|_, a| a[1].1["results"][0]["judge"] = Value::Null),
            Box::new(|_, a| a[1].1["meta"]["prompt_sha256"] = json!("changed")),
            Box::new(|_, a| a[1].1["meta"]["model"] = json!("changed")),
            Box::new(|_, a| a[1].1["meta"]["judge_model"] = json!("claude-opus-5-5")),
            Box::new(|_, a| { a[1].1["meta"].as_object_mut().unwrap().remove("judge_model"); }),
            Box::new(|_, a| a[1].1["meta"]["started"] = json!("2026-10-26T10:00:00+00:00")),
            Box::new(|_, a| a[1].1["results"][0]["seconds"] = json!("nan")),
            Box::new(|_, a| a[1].1["summary"]["passed"] = json!(0)),
            Box::new(|m, _| { m.as_object_mut().unwrap().remove("version"); }),
        ];
        for (i, change) in changes.iter().enumerate() {
            let (mut m, mut a) = experiment("latency", json!(0.15));
            change(&mut m, &mut a);
            assert!(evaluate(&m, &a).is_err(), "change {i} was accepted");
        }
    }

    #[test]
    fn rounds_and_gain_cannot_be_zero_or_trivial() {
        let plan = experiment("latency", json!(0.15)).0["plan"].clone();
        for (k, v) in [("rounds", json!(2)), ("rounds", json!(3)), ("rounds", json!(true)),
                       ("minimum_gain", json!(0)), ("minimum_gain", json!(0.001)), ("minimum_gain", json!("nan"))] {
            let mut p = plan.clone();
            p[k] = v;
            assert!(validate_plan(&p).is_err(), "{k} {}", p[k]);
        }
    }

    #[test]
    fn unknown_error_is_not_treated_as_successful_completion() {
        let mut row = experiment("latency", json!(0.15)).1[0].1["results"][0].clone();
        row["reasons"] = json!(["upstream stream disconnected"]);
        assert!(!completed(&row).unwrap());
        row["reasons"] = json!(["missing 'word'", "search_sections not called (called: ['search'])"]);
        assert!(completed(&row).unwrap());
    }

    #[test]
    fn python_formats() {
        for (x, repr) in [(22.6, "22.6"), (10.0, "10.0"), (0.0001, "0.0001"), (0.00001, "1e-05"), (1e16, "1e+16"),
                          (1e15, "1000000000000000.0"), (26.610000000000003, "26.610000000000003"), (-0.5, "-0.5"), (0.0, "0.0")] {
            assert_eq!(float_repr(x), repr);
        }
        for (x, g) in [(0.2, "0.2"), (1.0, "1"), (0.15345, "0.1535"), (1.0625, "1.062"), (0.00001234, "1.234e-05"), (12345.6, "1.235e+04"), (0.0, "0")] {
            assert_eq!(general4(x), g);
        }
        assert_eq!(dumps(&json!({"a": [1, 2.5, "é\"\n"], "b": {}, "c": []})), "{\n  \"a\": [\n    1,\n    2.5,\n    \"\\u00e9\\\"\\n\"\n  ],\n  \"b\": {},\n  \"c\": []\n}");
        // statistics.mean([0.1, 0.2, 0.4]) is 0.23333333333333334; summing the floats gives ...36.
        assert_eq!(exact_mean(&[0.1, 0.2, 0.4]), 0.23333333333333334);
        assert_eq!(exact_mean(&[22.6]), 22.6);
    }

    // ---------- with files, git and a model directory ----------

    struct Temp(PathBuf);
    impl Temp {
        fn new(name: &str) -> Temp {
            let p = std::env::temp_dir().join(format!("gate-test-{name}-{}-{}", std::process::id(), random_hex8().unwrap()));
            fs::create_dir_all(&p).unwrap();
            Temp(fs::canonicalize(p).unwrap())
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sh(dir: &Path, args: &[&str]) {
        let ok = Command::new("git").arg("-C").arg(dir).args(["-c", "user.email=test@example.invalid", "-c", "user.name=Gate test",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]).args(args).stdout(Stdio::null()).stderr(Stdio::null()).status().unwrap();
        assert!(ok.success(), "git {args:?}");
    }

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    /// A retriever repo with production v2-qwen and a candidate reranker "new", and a RiverGuide
    /// checkout on main whose golden set declares semantic checks for `ids`.
    fn setup(t: &Temp, ids: &[String]) -> (PathBuf, PathBuf) {
        let (root, rg) = (t.0.join("retriever"), t.0.join("riverguide"));
        for model in ["v2-qwen", "new"] {
            write(&root.join(format!("models/{model}/config.json")), "{}");
            write(&root.join(format!("models/{model}/model.safetensors")), model);
        }
        std::os::unix::fs::symlink("v2-qwen", root.join("models/production")).unwrap();
        write(&root.join("index/meta.json"), r#"{"corpus_commit": "db17b0e"}"#);
        write(&root.join("lab/production.env"), "ENGINE=ollama-max\n");
        for p in SOURCES {
            write(&root.join(p), p);
        }
        let golden: Vec<Value> = ids.iter().map(|id| json!({"id": id, "expected_summary": "A supported answer"})).collect();
        write(&rg.join("evals/golden.yaml"), &serde_json::to_string(&golden).unwrap());
        write(&rg.join("prompt.md"), "prompt");
        sh(&rg, &["init", "-q", "-b", "main"]);
        sh(&rg, &["add", "."]);
        sh(&rg, &["commit", "-qm", "golden"]);
        sh(&root, &["init", "-q", "-b", "main"]);
        (root, rg)
    }

    fn commit_plan(root: &Path, plan: &Value) -> PathBuf {
        let path = root.join("plan.json");
        save(&path, plan).unwrap();
        sh(root, &["add", "plan.json"]);
        sh(root, &["commit", "-qm", "Declare target before measurement"]);
        path
    }

    #[test]
    fn committed_plan_to_switch_decision_without_model_calls() {
        let t = Temp::new("flow");
        let (template, artifacts) = experiment("latency", json!(0.15));
        let ids: Vec<String> = template["expected_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
        let (root, rg) = setup(&t, &ids);
        let plan = commit_plan(&root, &template["plan"]);
        let manifest = prepare(&root, &rg, &plan, "rerank-new").unwrap();
        let frozen = load(&manifest).unwrap();
        assert_eq!(frozen["plan"], template["plan"]);
        assert_eq!(frozen["runs"], json!([]));
        assert_eq!(frozen["semantic_ids"].as_array().unwrap().len(), 10);
        for (i, (spec, mut artifact)) in artifacts.into_iter().enumerate() {
            artifact["meta"]["started"] = json!(iso_now(Utc::now()));
            let path = root.join(format!("run{i}.json"));
            save(&path, &artifact).unwrap();
            add(&manifest, spec["arm"].as_str().unwrap(), spec["round"].as_i64().unwrap(), &path).unwrap();
        }
        let result = evaluate_file(&manifest).unwrap();
        assert_eq!(result["passed"], true);
        let verdict = root.join("verdict.json");
        save(&verdict, &result).unwrap();
        assert!(check_switch(&root, &rg, &verdict).unwrap().starts_with("predeclared latency benefit met"));
        // Any change to what was measured: measure again.
        write(&root.join("models/new/model.safetensors"), "changed weights");
        assert!(check_switch(&root, &rg, &verdict).unwrap_err().contains("changed; measure again"));
        save(&root.join("run1.json"), &json!({"edited": true})).unwrap();
        assert!(evaluate_file(&manifest).unwrap_err().contains("measurement changed"));
    }

    #[test]
    fn prepare_refuses_uncommitted_edits_and_posthoc_plans() {
        let t = Temp::new("plan");
        let (mut template, _) = experiment("latency", json!(0.15));
        let (root, rg) = setup(&t, &["q0".to_string()]);
        let plan = commit_plan(&root, &template["plan"]);
        let manifest = prepare(&root, &rg, &plan, "rerank-new").unwrap();
        assert_eq!(load(&manifest).unwrap()["semantic_ids"], json!(["q0"]));
        let mut edited = template["plan"].clone();
        edited["minimum_gain"] = json!(0.05);
        save(&plan, &edited).unwrap();
        assert!(prepare(&root, &rg, &plan, "rerank-new").unwrap_err().contains("commit the exact plan"));
        // A manifest whose plan was edited after the commit.
        template["plan"]["minimum_gain"] = json!(0.05);
        template["root"] = json!(root.display().to_string());
        template["plan_path"] = json!("plan.json");
        template["plan_commit"] = json!(git(&root, &["rev-parse", "HEAD"]).unwrap());
        template["runs"] = json!([]);
        let posthoc = root.join("posthoc.json");
        save(&posthoc, &template).unwrap();
        assert!(evaluate_file(&posthoc).unwrap_err().contains("committed declaration"));
    }

    #[test]
    fn legacy_pass_cannot_authorize_switch() {
        let t = Temp::new("legacy");
        let path = t.0.join("legacy.json");
        save(&path, &json!({"candidate": "rerank-old", "production": "v2-qwen", "passed": true})).unwrap();
        assert!(check_switch(&t.0, &t.0, &path).unwrap_err().contains("legacy tolerance-only"));
    }

    #[test]
    fn model_changes_are_part_of_switch_identity() {
        let t = Temp::new("model");
        write(&t.0.join("config.json"), "{}");
        write(&t.0.join("model.safetensors"), "before");
        let before = model_identity(&t.0).unwrap();
        write(&t.0.join("model.safetensors"), "changed weights");
        assert_ne!(before, model_identity(&t.0).unwrap());
    }
}
