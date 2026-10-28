// Pick the engine that survives the bake-off (ported from bakeoff_verdict.py, 2026-10-25).
//
//   riverguide-lab verdict ollama-max sglang-mtp sglang-dflash     # writes runs/bakeoff.json
//
// The incumbent (lab/production.env's ENGINE) keeps its place unless a challenger is clearly
// better where students feel it:
//   - its run was fair (fair.rs), and it started (fundraising's extraction call is tried and
//     reported, not required: the app is shelved as of 2026-10-24 and follows whatever engine wins
//     when it comes back);
//   - quality held: golden passes over both rounds at least the incumbent's minus 2 (run-to-run noise);
//   - it finished as many answers at 4 students (River Guide's real peak is 3) as the incumbent, give or take 1;
//   - p90 seconds at 4 students at least 15% lower, and one student's median no more than 10% higher.
// Among challengers that clear all of that, the lowest p90 at 4 students wins; questions a minute
// at 16 students breaks a tie. If the incumbent's own run isn't fair, nothing is decided.
//
// What may go (runs/bakeoff.json "remove_paths", under ~/engines): an engine family only when
// every one of its variants in this bake-off had a fair run and lost; a model file only when no
// winning or kept engine reads it. Ollama always stays (model_ab.sh tries new models in it first).
use crate::fair::{self, family};
use crate::{read_json, tail_chars, write_json};
use serde_json::{json, Map, Value};
use std::{collections::BTreeSet, fs, path::Path};

pub fn production_engine() -> String {
    fs::read_to_string("lab/production.env")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("ENGINE=").map(|v| v.trim().to_string())))
        .unwrap_or_else(|| "ollama-p1".into())
}

/// Python's repr() of what bakeoff.sh stored as "why" (the old verdict printed it with str()).
pub(crate) fn py_repr(v: &Value) -> String {
    fn s(t: &str) -> String {
        let q = if t.contains('\'') && !t.contains('"') { '"' } else { '\'' };
        let mut out = String::from(q);
        for c in t.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c == q => { out.push('\\'); out.push(c); }
                c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push(q);
        out
    }
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Number(n) => n.to_string(),
        Value::String(t) => s(t),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", s(k), py_repr(v))).collect::<Vec<_>>().join(", ")),
    }
}

fn result(engine: &str, meta: &Value) -> Result<Value, String> {
    if !meta["started"].as_bool().unwrap_or(false) {
        let why = match &meta["why"] {
            Value::Null => String::new(),
            Value::String(t) => t.clone(),
            other => py_repr(other),
        };
        return Ok(json!({"engine": engine, "started": false, "why": tail_chars(&why, 300)}));
    }
    let (mut passed, mut total) = (0i64, 0i64);
    for r in [1, 2] {
        let g = read_json(&format!("runs/golden-bake-{engine}-r{r}.json"))?;
        passed += g["summary"]["passed"].as_i64().ok_or("golden summary has no passed")?;
        total += g["summary"]["total"].as_i64().ok_or("golden summary has no total")?;
    }
    let load = read_json(&format!("runs/load-bake-{engine}.json"))?;
    let level = |n: i64| -> Result<Value, String> {
        load["levels"].as_array().and_then(|l| l.iter().find(|lv| lv["students"].as_i64() == Some(n))).cloned()
            .ok_or_else(|| format!("runs/load-bake-{engine}.json has no level at {n} students"))
    };
    let (l1, l4, l16) = (level(1)?, level(4)?, level(16)?);
    Ok(json!({
        "engine": engine, "started": true, "settings": meta["settings"], "memory_mib": meta["memory_mib"],
        "fundraising_ok": meta["fundraising_ok"], "golden": passed, "golden_of": total,
        "median_1": l1["median_seconds"], "p90_4": l4["p90_seconds"], "unfinished_4": l4["unfinished"],
        "qpm_4": l4["questions_per_minute"], "qpm_16": l16["questions_per_minute"], "p90_16": l16["p90_seconds"],
    }))
}

fn num(r: &Value, k: &str) -> f64 {
    r[k].as_f64().unwrap_or(f64::NAN)
}

pub fn run(engines: &[&str]) -> Result<u8, String> {
    let incumbent = production_engine();
    let slots = fair::incumbent_slots(&incumbent);
    let mut table = Map::new();
    for &e in engines {
        let meta = read_json(&format!("runs/bake-{e}.json"))?;
        let mut r = result(e, &meta)?;
        let f = fair::check(e, &meta, &incumbent, slots);
        r["fair"] = json!(f.problems.is_empty());
        r["problems"] = json!(f.problems);
        r["notes"] = json!(f.notes);
        r["checks"] = Value::Object(f.checks);
        table.insert(e.to_string(), r);
    }

    let fair_run = |r: &Value| r["fair"].as_bool().unwrap_or(false);
    // Decided: the incumbent ran fairly in this bake-off, so a loss to it means something.
    let decided = table.get(&incumbent).map_or(false, |b| b["started"].as_bool().unwrap_or(false) && fair_run(b));
    let (winner, reason) = match table.get(&incumbent) {
        Some(base) if base["started"].as_bool().unwrap_or(false) && !fair_run(base) => {
            println!("the incumbent {incumbent}'s run isn't fair; keeping it and deciding nothing");
            (incumbent.clone(), format!("the incumbent's own run isn't fair ({})", base["problems"][0].as_str().unwrap_or("")))
        }
        Some(base) if base["started"].as_bool().unwrap_or(false) => {
            let clears = |r: &Value| {
                fair_run(r) && r["started"].as_bool().unwrap_or(false)
                    && num(r, "golden") >= num(base, "golden") - 2.0
                    && num(r, "unfinished_4") <= num(base, "unfinished_4") + 1.0
                    && num(r, "p90_4") <= 0.85 * num(base, "p90_4")
                    && num(r, "median_1") <= 1.10 * num(base, "median_1")
            };
            let best = table.iter().filter(|(e, r)| **e != incumbent && clears(r)).map(|(_, r)| r)
                .min_by(|a, b| num(a, "p90_4").total_cmp(&num(b, "p90_4")).then(num(b, "qpm_16").total_cmp(&num(a, "qpm_16"))));
            match best {
                Some(b) => (b["engine"].as_str().unwrap_or_default().to_string(), format!(
                    "p90 at 4 students {} s vs {} s, golden {} vs {}, {} vs {} questions/min at 16",
                    b["p90_4"], base["p90_4"], b["golden"], base["golden"], b["qpm_16"], base["qpm_16"])),
                None => (incumbent.clone(), "no challenger cleared the bar".to_string()),
            }
        }
        _ => {
            println!("the incumbent {incumbent} has no result; keeping it");
            (incumbent.clone(), "incumbent unmeasured".to_string())
        }
    };

    // What promote_engine.sh may remove. A family stays while any of its variants is unresolved,
    // and nothing goes unless the incumbent was measured fairly in this same bake-off.
    let won = family(&winner).to_string();
    let mut remove: Vec<String> = vec![];
    let mut kept = Map::new();
    let families: BTreeSet<&str> = engines.iter().map(|e| family(e)).filter(|f| *f != "ollama").collect();
    for fam in families {
        if fam == won {
            continue;
        }
        let unfair: Vec<&str> = engines.iter().copied().filter(|e| family(e) == fam && !fair_run(&table[*e])).collect();
        if unfair.is_empty() && decided {
            remove.push(fam.to_string());
        } else {
            kept.insert(fam.to_string(), json!(unfair));
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let tested: BTreeSet<&str> = engines.iter().map(|e| family(e)).collect();
    // Still in use: a family kept for a re-run, or one installed but not in this bake-off; and the
    // winner's family, though of a winner's drafters only the one its variant reads.
    let held = |f: &str| kept.contains_key(f) || (!tested.contains(f) && Path::new(&format!("{home}/engines/{f}")).exists());
    let needed = |f: &str| f == won || held(f);
    let mut remove_paths = remove.clone();
    if !needed("vllm") && !needed("sglang") {
        remove_paths.extend(["models/qwen3.8-27b-w4a16".into(), "hf".into()]);
    }
    if !needed("llamacpp") {
        remove_paths.push("models/qwen3.8-27b-template".into());
    }
    if winner != "llamacpp-dflash" && !held("llamacpp") {
        remove_paths.push("models/qwen3.8-27b-dflash2-gguf".into());
    }
    if !["vllm-dflash", "sglang-dflash"].contains(&winner.as_str()) && !held("vllm") && !held("sglang") {
        remove_paths.extend(["models/qwen3.8-27b-dflash2-w4a16".into(), "models/qwen3.8-27b-dflash2".into()]);
    }
    let unresolved: Vec<Value> = table.iter().filter(|(_, r)| !fair_run(r))
        .map(|(e, r)| json!({"engine": e, "problems": r["problems"]})).collect();
    let summary = unresolved.iter()
        .map(|u| format!("{} ({})", u["engine"].as_str().unwrap_or(""), u["problems"].as_array().map(|p| p.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; ")).unwrap_or_default()))
        .collect::<Vec<_>>().join("; ");

    let kept_families: Vec<String> = kept.keys().cloned().collect();
    let out = json!({"incumbent": incumbent, "winner": winner, "reason": reason, "engines": table,
        "production_concurrency": slots, "remove": remove, "remove_paths": remove_paths, "kept": kept,
        "kept_families": kept_families, "unresolved": unresolved, "unresolved_summary": summary});
    write_json("runs/bakeoff.json", &out)?;

    println!("{:10} {:>7} {:>8} {:>6} {:>7} {:>8} {:>6} fundraising", "engine", "golden", "median@1", "p90@4", "q/min@4", "q/min@16", "MiB");
    let s = |v: &Value| match v { Value::String(t) => t.clone(), other => other.to_string() };
    for r in out["engines"].as_object().into_iter().flat_map(|m| m.values()) {
        if r["started"].as_bool().unwrap_or(false) {
            println!("{:10} {:>3}/{:<3} {:>8} {:>6} {:>7} {:>8} {:>6} {}", s(&r["engine"]), s(&r["golden"]), s(&r["golden_of"]),
                s(&r["median_1"]), s(&r["p90_4"]), s(&r["qpm_4"]), s(&r["qpm_16"]), s(&r["memory_mib"]),
                if r["fundraising_ok"].as_bool().unwrap_or(false) { "ok" } else { "FAILED" });
        } else {
            println!("{:10} did not start: {}", s(&r["engine"]), py_repr(&json!(tail_chars(r["why"].as_str().unwrap_or(""), 80))));
        }
        for p in r["problems"].as_array().into_iter().flatten() {
            println!("{:10}   not a fair run: {}", "", s(p));
        }
    }
    println!("winner: {winner} ({reason})");
    if !remove_paths.is_empty() {
        println!("may remove: {}", remove_paths.join(" "));
    }
    if !kept.is_empty() {
        println!("kept for a re-run: {}", kept.keys().cloned().collect::<Vec<_>>().join(" "));
    }
    Ok(0)
}
