// The fairness gate: whether a bake-off run measured an engine at its best, so that losing counts.
// A run that fails any check is unresolved, not lost: the verdict won't let it win, and
// promote_engine.sh won't remove its engine (verdict.rs). Each check reads the engine's own server
// log for the settings that were measured, up to the end of the load test:
//   - concurrency: its best settings (the first the bake-off tried) serve at least as many
//     students at once as production does. Fallbacks may give up slots; the first settings may not.
//     (SGLang's ladder of 2026-10-25 began at one conversation at a time, against Ollama's four.)
//   - speedup: a variant built on a drafter (MTP, DFlash2) shows its drafts accepted, at least 1.5
//     tokens per verify step (1.0 is none) or 30% of drafts. (SGLang's DFlash2 had 1.00.)
//   - context: every conversation can reach 32,768 tokens, as production Ollama allows.
//   - errors: the server logged no crash, CUDA error or out-of-memory during the test.
//   - prefix cache: where the engine reports it, the cache hit at least once.
use crate::{read_json, tail_chars};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::{fs, path::Path};

pub const CONTEXT: u64 = 32_768;
const MIN_ACCEPT_LEN: f64 = 1.5;
const MIN_ACCEPT_RATE: f64 = 0.3;
const ERRORS: &str = r"Scheduler hit an exception|EngineCore encountered a fatal error|CUDA error|NCCL error|CUDA out of memory|OutOfMemoryError|GGML_ASSERT|Segmentation fault|core dumped|illegal memory access";

pub struct Fairness {
    pub problems: Vec<String>,
    pub notes: Vec<String>,
    pub checks: Map<String, Value>,
}

pub enum Accept {
    Len(f64),
    Rate(f64),
}

pub fn family(engine: &str) -> &str {
    engine.split('-').next().unwrap_or(engine)
}

fn caps<'a>(text: &'a str, re: &str) -> impl Iterator<Item = regex::Captures<'a>> {
    Regex::new(re).expect("regex").captures_iter(text).collect::<Vec<_>>().into_iter()
}

fn first_u64(text: &str, re: &str) -> Option<u64> {
    caps(text, re).next().and_then(|c| c[1].replace(',', "").parse().ok())
}

fn mean(xs: &[f64]) -> Option<f64> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn count(n: u64) -> Value {
    if n == u64::MAX { json!("auto") } else { json!(n) }
}

/// How many conversations a command lets the engine serve at once: its slots, or its cap on
/// running requests (unset, SGLang and vLLM size it themselves, which never limits the bake-off).
pub fn concurrency(engine: &str, cmd: &str) -> u64 {
    let n = |re: &str| first_u64(cmd, re);
    match family(engine) {
        "ollama" => n(r"OLLAMA_NUM_PARALLEL=(\d+)").unwrap_or(1),
        "llamacpp" => n(r"(?:^|\s)(?:-np|--parallel)\s+(\d+)").unwrap_or(1),
        "vllm" => n(r"--max-num-seqs\s+(\d+)").unwrap_or(256),
        "sglang" => n(r"--max-running-requests\s+(\d+)").unwrap_or(u64::MAX),
        _ => 1,
    }
}

/// A variant named for its slot count (ollama-p1, ollama-p4) is measured at that count on purpose.
fn named_for_slots(engine: &str) -> bool {
    Regex::new(r"-p\d+$").expect("regex").is_match(engine)
}

/// Production's concurrency: the incumbent's measured command, or the production engine's script.
pub fn incumbent_slots(incumbent: &str) -> u64 {
    let home = std::env::var("HOME").unwrap_or_default();
    read_json(&format!("runs/bake-{incumbent}.json"))
        .ok()
        .and_then(|m| m["command"].as_str().map(String::from))
        .or_else(|| fs::read_to_string(format!("{home}/engines/serve-production.sh")).ok())
        .map(|c| concurrency(incumbent, &c))
        .unwrap_or(1)
}

/// The drafts' acceptance an engine logged: SGLang's and vLLM's mean accepted length per verify
/// step (it counts the target's own token, so 1.0 is no draft accepted), llama.cpp's share of
/// drafted tokens accepted.
pub fn acceptance(family: &str, log: &str) -> Option<Accept> {
    match family {
        "sglang" => {
            let xs: Vec<f64> = caps(log, r"Decode batch.*?accept len: ([\d.]+)").filter_map(|c| c[1].parse().ok()).collect();
            mean(&xs).map(Accept::Len)
        }
        "vllm" => {
            let xs: Vec<f64> = caps(log, r"Mean acceptance length: ([\d.]+)").filter_map(|c| c[1].parse().ok()).collect();
            mean(&xs).map(Accept::Len)
        }
        "llamacpp" => {
            let (mut acc, mut gen) = (0u64, 0u64);
            for c in caps(log, r"draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)") {
                acc += c[1].parse::<u64>().unwrap_or(0);
                gen += c[2].parse::<u64>().unwrap_or(0);
            }
            (gen > 0).then(|| Accept::Rate(acc as f64 / gen as f64))
        }
        _ => None,
    }
}

/// Whether the drafts were accepted often enough for the speedup to count as working.
pub fn accepted(a: &Accept) -> bool {
    match a {
        Accept::Len(x) => *x >= MIN_ACCEPT_LEN,
        Accept::Rate(r) => *r >= MIN_ACCEPT_RATE,
    }
}

/// The measured settings' server log, cut where the load test ended when bakeoff.sh noted that.
fn measured_log(engine: &str, meta: &Value) -> String {
    let settings = meta["settings"].as_u64().unwrap_or(1);
    let numbered = format!("logs/bake-{engine}-s{settings}.log");
    let path = if Path::new(&numbered).exists() { numbered } else { format!("logs/bake-{engine}.log") };
    let mut bytes = fs::read(&path).unwrap_or_default();
    if let Some(n) = meta["log_bytes"].as_u64() {
        bytes.truncate(n as usize);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn check(engine: &str, meta: &Value, incumbent: &str, slots: u64) -> Fairness {
    let mut f = Fairness { problems: vec![], notes: vec![], checks: Map::new() };
    let fam = family(engine);
    let started = meta["started"].as_bool().unwrap_or(false);

    // Concurrency of its best settings: the first attempt, or the measured command when that was
    // settings 1. Runs from before this gate didn't note their attempts.
    let first = meta["attempts"].as_array().and_then(|a| a.first()).and_then(|a| a["command"].as_str()).map(String::from)
        .or_else(|| (meta["settings"].as_u64() == Some(1)).then(|| meta["command"].as_str().map(String::from)).flatten());
    if engine != incumbent && !named_for_slots(engine) {
        match &first {
            Some(cmd) => {
                let n = concurrency(engine, cmd);
                f.checks.insert("best_settings_concurrency".into(), count(n));
                if n < slots {
                    f.problems.push(format!("its best settings served {n} conversation{} at once; production serves {slots}", plural(n)));
                }
            }
            None if started => f.notes.push(format!(
                "measured with fallback settings {}; the settings tried before them weren't noted (a run from before the fairness gate)",
                meta["settings"])),
            None => {}
        }
    }

    // A variant that never ran counts as a fair loss only with evidence that the fault isn't the
    // setup (bakeoff.sh's diagnose: the drafter's original release, at a context that fits):
    //   - every settings ran out of memory or couldn't hold 32K, and the original drafter works:
    //     it doesn't fit this card;
    //   - the original drafter accepts nothing either: the engine's support is broken.
    // Anything else (a crash, a bad conversion, no diagnosis) is unresolved.
    let diagnosis = meta["diagnosis"].as_str().or_else(|| meta["refused"].as_str().and_then(|w| w.split_once("; ").map(|(_, d)| d)));
    let original_broken = diagnosis.map_or(false, |d| d.contains("bf16 drafter") && d.contains("drafts low"));
    if let Some(why) = meta["refused"].as_str() {
        if why.starts_with("its drafter isn't working") && original_broken {
            f.notes.push(format!("its drafter's original accepted almost nothing too, so the engine's support is broken, not the setup: {why}"));
        } else {
            f.problems.push(format!("the bake-off refused to measure it: {why}"));
        }
        return f;
    }
    if !started {
        let crashed = meta["why"].as_object().map_or(false, |o| o.keys().any(|k| k.starts_with("crashed during")));
        let memory = Regex::new(r"out of memory|OutOfMemoryError|can't hold a 32,768-token conversation|too small to serve").expect("regex");
        let attempts = meta["attempts"].as_array().cloned().unwrap_or_default();
        let memory_bound = !attempts.is_empty() && attempts.iter().all(|a| a["outcome"].as_str().map_or(false, |o| memory.is_match(o)));
        let original_works = diagnosis.map_or(false, |d| d.contains("bf16 drafter") && d.contains("drafts ok"));
        if !crashed && memory_bound && original_works {
            f.notes.push(format!("all {} settings ran out of memory or couldn't hold a 32,768-token conversation, and its drafter's original works ({}): it doesn't fit this card",
                attempts.len(), diagnosis.unwrap_or("")));
        } else if !crashed && original_broken {
            f.notes.push(format!("its drafter's original accepted almost nothing too, so the engine's support is broken, not the setup ({})", diagnosis.unwrap_or("")));
        } else {
            let mut p = if crashed { "its server crashed during the test".to_string() } else { "it did not start".to_string() };
            if let Some(d) = diagnosis {
                p = format!("{p} ({d})");
            }
            f.problems.push(p);
        }
        return f;
    }

    let cmd = meta["command"].as_str().unwrap_or("");
    let log = measured_log(engine, meta);
    f.checks.insert("concurrency".into(), count(concurrency(engine, cmd)));

    let drafted = engine.ends_with("-mtp") || engine.ends_with("-dflash") || cmd.contains("--speculative") || cmd.contains("--spec-type");
    if drafted {
        match acceptance(fam, &log) {
            Some(a @ Accept::Len(x)) => {
                f.checks.insert("accept_len".into(), json!((x * 100.0).round() / 100.0));
                if !accepted(&a) {
                    f.problems.push(format!("its drafts averaged {x:.2} tokens per verify step (1.00 is none accepted): the speedup didn't work"));
                }
            }
            Some(a @ Accept::Rate(r)) => {
                f.checks.insert("accept_rate".into(), json!((r * 1000.0).round() / 1000.0));
                if !accepted(&a) {
                    f.problems.push(format!("{:.0}% of its drafted tokens were accepted: the speedup didn't work", r * 100.0));
                }
            }
            None => f.problems.push("its log shows no draft acceptance, so its speedup can't be confirmed".into()),
        }
    }

    let per_conversation = match fam {
        "sglang" => first_u64(&log, r"max_total_num_tokens=(\d+)"),
        "vllm" => first_u64(&log, r"GPU KV cache size: ([\d,]+) tokens"),
        "llamacpp" => first_u64(cmd, r"(?:^|\s)(?:-c|--ctx-size)\s+(\d+)").map(|c| c / concurrency(engine, cmd).max(1)),
        _ => None,
    };
    if let Some(n) = per_conversation {
        f.checks.insert("tokens_per_conversation".into(), json!(n));
        if n < CONTEXT {
            f.problems.push(format!("a conversation could reach only {n} tokens, not {CONTEXT}"));
        }
    }

    if let Some(line) = log.lines().find(|l| Regex::new(ERRORS).expect("regex").is_match(l)) {
        f.problems.push(format!("its server logged an error during the test: {}", tail_chars(line.trim(), 160)));
    }

    match fam {
        "sglang" => {
            let (mut new, mut cached) = (0u64, 0u64);
            for c in caps(&log, r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)") {
                new += c[1].parse::<u64>().unwrap_or(0);
                cached += c[2].parse::<u64>().unwrap_or(0);
            }
            if new + cached > 0 {
                f.checks.insert("prefix_cache_hit".into(), json!((cached as f64 / (new + cached) as f64 * 1000.0).round() / 1000.0));
                if cached == 0 {
                    f.problems.push("its prefix cache never hit".into());
                }
            }
        }
        "vllm" => {
            let rates: Vec<f64> = caps(&log, r"Prefix cache hit rate: ([\d.]+)%").filter_map(|c| c[1].parse().ok()).collect();
            if let Some(best) = rates.iter().cloned().reduce(f64::max) {
                f.checks.insert("prefix_cache_hit".into(), json!(best / 100.0));
                if best == 0.0 {
                    f.problems.push("its prefix cache never hit".into());
                }
            }
        }
        _ => {}
    }
    f
}

pub fn preflight(engine: &str, command: &str) -> Result<u8, String> {
    let incumbent = crate::verdict::production_engine();
    let slots = incumbent_slots(&incumbent);
    let n = concurrency(engine, command);
    if engine == incumbent || named_for_slots(engine) || n >= slots {
        return Ok(0);
    }
    println!("its best settings serve {n} conversation{} at once and production ({incumbent}) serves {slots}; \
              the bake-off's settings for an engine start at production's concurrency and fall back from there", plural(n));
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrency_from_commands() {
        assert_eq!(concurrency("ollama-max", "env OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve"), 4);
        assert_eq!(concurrency("ollama-p1", "(system Ollama, 1 slot)"), 1);
        assert_eq!(concurrency("llamacpp-mtp", "llama-server -m x -np 3 -c 98304"), 3);
        assert_eq!(concurrency("vllm-mtp", "vllm serve x --max-num-seqs 4 --kv-cache-dtype fp8"), 4);
        assert_eq!(concurrency("sglang-mtp", "python -m sglang.launch_server --max-running-requests 1 --max-mamba-cache-size 4"), 1);
        assert_eq!(concurrency("sglang-mtp", "python -m sglang.launch_server --context-length 32768"), u64::MAX);
    }

    fn attempt(n: u64, outcome: &str) -> Value {
        json!({"settings": n, "command": "python -m sglang.launch_server --max-running-requests 4", "outcome": outcome})
    }

    #[test]
    fn never_ran_needs_evidence() {
        let pool = "did not start: bakeoff: a 28814-token pool can't hold a 32,768-token conversation";
        // Out of memory everywhere, and the original drafter works: it doesn't fit this card (fair).
        let m = json!({"started": false, "attempts": [attempt(1, pool), attempt(2, "did not start: torch.OutOfMemoryError: CUDA out of memory")],
                       "diagnosis": "Inco AI's bf16 drafter at 8,192 tokens: drafts ok 3.40"});
        assert!(check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        // The original accepts nothing either: the engine's support is broken (fair).
        let m = json!({"started": false, "attempts": [attempt(1, pool)], "diagnosis": "Inco AI's bf16 drafter at 8,192 tokens: drafts low 1.00"});
        assert!(check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        // No diagnosis, or an attempt that failed some other way: unresolved.
        let m = json!({"started": false, "attempts": [attempt(1, pool)]});
        assert!(!check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        let m = json!({"started": false, "attempts": [attempt(1, "did not start: ValueError: bad flag")],
                       "diagnosis": "Inco AI's bf16 drafter at 8,192 tokens: drafts ok 3.40"});
        assert!(!check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        // A drafter that accepts nothing while its original works: a bad conversion, unresolved.
        let m = json!({"started": false, "attempts": [attempt(1, pool)],
                       "refused": "its drafter isn't working: drafts low 1.00 per verify step (1.00 is none accepted); Inco AI's bf16 drafter at 8,192 tokens: drafts ok 3.40"});
        assert!(!check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        let m = json!({"started": false, "attempts": [attempt(1, pool)],
                       "refused": "its drafter isn't working: drafts low 1.00 per verify step (1.00 is none accepted); Inco AI's bf16 drafter at 8,192 tokens: drafts low 1.01"});
        assert!(check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
        // A ladder that starts below production's concurrency is never fair.
        let m = json!({"started": false, "attempts": [json!({"settings": 1, "command": "--max-running-requests 1", "outcome": pool})],
                       "diagnosis": "Inco AI's bf16 drafter at 8,192 tokens: drafts ok 3.40"});
        assert!(!check("sglang-dflash", &m, "ollama-max", 4).problems.is_empty());
    }

    #[test]
    fn acceptance_from_logs() {
        let sg = "[x] Decode batch, #running-req: 1, accept len: 3.40, accept rate: 0.80, cuda graph: True\n\
                  [x] Decode batch, #running-req: 1, accept len: 2.60, accept rate: 0.55, cuda graph: True\n";
        assert!(matches!(acceptance("sglang", sg), Some(Accept::Len(x)) if (x - 3.0).abs() < 1e-9));
        let none = "[x] Decode batch, #running-req: 1, accept len: 1.00, accept rate: 0.00, cuda graph: True\n";
        assert!(matches!(acceptance("sglang", none), Some(Accept::Len(x)) if x == 1.0));
        let lc = "slot print_timing: id  1 | task 15830 | draft acceptance = 0.93750 (   90 accepted /    96 generated), mean len =  3.09\n\
                  slot print_timing: id  1 | task 15692 | draft acceptance = 0.84559 (  115 accepted /   136 generated), mean len =  2.77\n";
        assert!(matches!(acceptance("llamacpp", lc), Some(Accept::Rate(r)) if (r - 205.0 / 232.0).abs() < 1e-9));
        assert!(acceptance("vllm", "nothing here").is_none());
    }
}
