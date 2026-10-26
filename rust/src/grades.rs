// Grades for training RiverGuide's reranker, served to Claude Opus 5.5 graders and kept as they
// come (rerank_data.py writes the questions; train_reranker.py reads the grades):
//   riverguide-lab grades show BATCH [--start K]   five items from K: a made-up first message and
//                                                  v2's top 20 passages for it (the guide first)
//   riverguide-lab grades save BATCH               JSON lines on stdin, {"q": ID, "g": [20 grades]},
//                                                  checked and appended to data/rerank/opus/BATCH.jsonl
//   riverguide-lab grades status [GRADER]          items graded in each batch (opus, or qwen)
//   riverguide-lab grades qwen [--workers 4] [--limit N]
//                                                  production's Qwen grades every item not yet in
//                                                  data/rerank/qwen/, four at a time (Ollama's slots)
// Run from the repo root. A grade is 2 (answers what the person needs), 1 (related, not the
// answer) or 0 (doesn't help). Opus graded 125 items before it used up Joel's plan (2026-10-26);
// Qwen grades them all, costs no plan tokens, and the overlap measures how far it agrees with Opus.
use crate::read_lossy;
use serde_json::{json, Value};
use std::{collections::HashMap, fs, io::Read, io::Write, sync::{Arc, Mutex}, time::Duration};

const SHOW: usize = 5;
const CHARS: usize = 800;
const DIR: &str = "data/rerank";

const GRADING: &str = "You grade search results for RiverGuide, Indian River State College's student assistant, to train
the model that puts the best College page first. Each item is a first message a student or parent
might type (made up in their style, not a real person's) and the 20 College web page passages a
search returned for it, in the search's order.

Grade every passage for that person:
  2 = it answers what they need: landing on this page, they'd have their answer, or the exact next
      step for their situation (the form, the office, the page that does it).
  1 = related but not the answer: the right topic, or a general page that might lead there.
  0 = doesn't help them.
Several passages can be 2 (the College repeats itself across its site and catalog), or none.
Judge from the passages as shown, not from what you know about IRSC; when a passage is cut off,
judge what it shows. Don't let a passage's place in the list sway you.
";
const SAVE: &str = "
Save one JSON line per item, its 20 grades in the order shown:
  {\"q\": 17, \"g\": [2, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
";

struct Item {
    text: String,
    candidates: Vec<usize>,
}

fn questions() -> Result<HashMap<u64, Item>, String> {
    let mut out = HashMap::new();
    for line in read_lossy(&format!("{DIR}/questions.jsonl")).lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{DIR}/questions.jsonl: {e}"))?;
        let candidates = v["candidates"].as_array().ok_or("a question without candidates")?
            .iter().filter_map(|c| c.as_u64().map(|c| c as usize)).collect();
        out.insert(v["qid"].as_u64().ok_or("a question without a qid")?, Item { text: v["text"].as_str().unwrap_or("").into(), candidates });
    }
    if out.is_empty() {
        return Err(format!("no questions in {DIR}/questions.jsonl (rerank_data.py prep writes them)"));
    }
    Ok(out)
}

fn batch_ids(batch: &str) -> Result<Vec<u64>, String> {
    let s = fs::read_to_string(format!("{DIR}/{batch}.ids")).map_err(|e| format!("{DIR}/{batch}.ids: {e}"))?;
    Ok(s.split_whitespace().filter_map(|x| x.parse().ok()).collect())
}

fn graded(batch: &str) -> HashMap<u64, Value> {
    graded_by("opus", batch)
}

fn graded_by(grader: &str, batch: &str) -> HashMap<u64, Value> {
    read_lossy(&format!("{DIR}/{grader}/{batch}.jsonl")).lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["q"].as_u64().map(|q| (q, v)))
        .collect()
}

/// The first `n` characters, cut back to a word, as gen_student.py cuts passages for its writers.
fn cut(text: &str, n: usize) -> String {
    if text.chars().count() <= n {
        return text.to_string();
    }
    let head: String = text.chars().take(n).collect();
    let head = head.rfind(' ').map_or(head.as_str(), |i| &head[..i]);
    format!("{head} ...")
}

fn load_docs() -> Vec<Value> {
    read_lossy("data/docs.jsonl").lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// One item as every grader sees it: the message, then its passages numbered in the search's order.
fn render(qid: u64, item: &Item, docs: &[Value]) -> Result<String, String> {
    let mut out = format!("=== item {qid}\nMessage: {}\n\n", item.text);
    for (j, c) in item.candidates.iter().enumerate() {
        let d = docs.get(*c).ok_or_else(|| format!("chunk {c} isn't in data/docs.jsonl"))?;
        out.push_str(&format!("[{}] {}\n{}\n\n", j + 1, d["url"].as_str().unwrap_or(""), cut(d["text"].as_str().unwrap_or(""), CHARS)));
    }
    Ok(out)
}

fn show(batch: &str, start: usize) -> Result<u8, String> {
    let (items, ids) = (questions()?, batch_ids(batch)?);
    let docs = load_docs();
    let mut out = String::new();
    if start == 0 {
        out.push_str(GRADING);
        out.push_str(SAVE);
        out.push('\n');
    }
    let end = (start + SHOW).min(ids.len());
    for qid in &ids[start.min(end)..end] {
        let item = items.get(qid).ok_or_else(|| format!("item {qid} isn't in {DIR}/questions.jsonl"))?;
        out.push_str(&render(*qid, item, &docs)?);
    }
    out.push_str(&format!("--- items {}-{end} of {} in {batch}.", start + 1, ids.len()));
    if end < ids.len() {
        out.push_str(&format!(" Save their grades, then: grades show {batch} --start {end}\n"));
    } else {
        out.push_str(" That was the last part.\n");
    }
    print!("{out}");
    Ok(0)
}

fn save(batch: &str) -> Result<u8, String> {
    let (items, ids) = (questions()?, batch_ids(batch)?);
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).map_err(|e| e.to_string())?;
    let (mut saved, mut errors) = (Vec::new(), Vec::new());
    for (n, line) in input.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let problem = match serde_json::from_str::<Value>(line) {
            Err(_) => Some("not JSON".to_string()),
            Ok(v) => match (v["q"].as_u64(), v["g"].as_array()) {
                (Some(q), _) if !ids.contains(&q) => Some(format!("item {q} isn't in {batch}")),
                (Some(q), Some(g)) => {
                    let want = items.get(&q).map_or(0, |i| i.candidates.len());
                    let grades: Vec<u64> = g.iter().filter_map(Value::as_u64).filter(|x| *x <= 2).collect();
                    if grades.len() != g.len() || g.len() != want {
                        Some(format!("item {q} needs {want} grades, each 0, 1 or 2"))
                    } else {
                        saved.push(json!({"q": q, "g": grades}));
                        None
                    }
                }
                _ => Some("needs {\"q\": ID, \"g\": [grades]}".to_string()),
            },
        };
        if let Some(p) = problem {
            errors.push(format!("line {}: {p}", n + 1));
        }
    }
    fs::create_dir_all(format!("{DIR}/opus")).map_err(|e| e.to_string())?;
    let mut f = fs::OpenOptions::new().create(true).append(true).open(format!("{DIR}/opus/{batch}.jsonl")).map_err(|e| e.to_string())?;
    for v in &saved {
        writeln!(f, "{v}").map_err(|e| e.to_string())?;
    }
    let done = graded(batch);
    println!("saved {}; {} of {} in {batch} still to grade", saved.len(), ids.iter().filter(|q| !done.contains_key(q)).count(), ids.len());
    for e in errors.iter().take(10) {
        println!("  not saved: {e}");
    }
    Ok(0)
}

fn batches() -> Result<Vec<String>, String> {
    let mut out: Vec<String> = fs::read_dir(DIR).map_err(|e| format!("{DIR}: {e}"))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".ids").map(String::from)).collect();
    out.sort();
    Ok(out)
}

fn status(grader: &str) -> Result<u8, String> {
    let (mut all, mut done) = (0, 0);
    for b in &batches()? {
        let (ids, g) = (batch_ids(b)?, graded_by(grader, b));
        let n = ids.iter().filter(|q| g.contains_key(q)).count();
        println!("{b}: {n}/{}", ids.len());
        all += ids.len();
        done += n;
    }
    println!("graded {done}/{all}");
    Ok(0)
}

/// Production's Qwen grades the items it hasn't graded yet, `workers` at a time. Each call gets the
/// grading guide and one item as the agents saw it, and a JSON schema that allows exactly one grade
/// of 0, 1 or 2 per passage; it thinks first (Qwen3.8's reasoning), at Qwen's recommended sampling.
fn qwen(workers: usize, limit: Option<usize>) -> Result<u8, String> {
    let items = Arc::new(questions()?);
    let docs = Arc::new(load_docs());
    let mut todo = vec![];
    for b in batches()? {
        let done = graded_by("qwen", &b);
        todo.extend(batch_ids(&b)?.into_iter().filter(|q| !done.contains_key(q)).map(|q| (q, b.clone())));
    }
    todo.truncate(limit.unwrap_or(usize::MAX));
    let total = todo.len();
    println!("{total} items for Qwen to grade");
    fs::create_dir_all(format!("{DIR}/qwen")).map_err(|e| e.to_string())?;
    let base = std::env::var("PROD_BASE").unwrap_or_else(|_| "http://127.0.0.1:11440".into());
    // QWEN_CTX sets each slot's context (a dedicated Ollama with more, shorter slots grades faster).
    let ctx: Option<u64> = std::env::var("QWEN_CTX").ok().and_then(|c| c.parse().ok());
    let queue = Arc::new(Mutex::new(todo));
    let progress = Arc::new(Mutex::new((0usize, 0usize))); // (graded, failed); also the file lock
    let handles: Vec<_> = (0..workers).map(|_| {
        let (items, docs, queue, progress, base, ctx) = (items.clone(), docs.clone(), queue.clone(), progress.clone(), base.clone(), ctx);
        std::thread::spawn(move || loop {
            let Some((qid, batch)) = queue.lock().unwrap().pop() else { return };
            let item = &items[&qid];
            let n = item.candidates.len();
            let prompt = match render(qid, item, &docs) {
                Ok(r) => format!("{GRADING}\n{r}Return {{\"g\": [...]}} with exactly {n} grades, one per passage in the order shown."),
                Err(_) => continue,
            };
            let mut body = json!({"model": "qwen3.8:27b", "stream": false, "think": true,
                "messages": [{"role": "user", "content": prompt}],
                "format": {"type": "object", "required": ["g"], "properties": {"g": {"type": "array", "minItems": n, "maxItems": n,
                    "items": {"type": "integer", "enum": [0, 1, 2]}}}},
                "options": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_predict": 6144}});
            if let Some(c) = ctx {
                body["options"]["num_ctx"] = json!(c);
            }
            let mut grades = None;
            for _ in 0..2 {
                let reply: Option<Value> = ureq::post(&format!("{base}/api/chat")).timeout(Duration::from_secs(900))
                    .send_json(body.clone()).ok().and_then(|r| r.into_json().ok());
                let g: Option<Vec<u64>> = reply.as_ref().and_then(|r| serde_json::from_str::<Value>(r["message"]["content"].as_str()?).ok())
                    .and_then(|v| v["g"].as_array().map(|a| a.iter().filter_map(Value::as_u64).filter(|x| *x <= 2).collect()));
                if let Some(g) = g.filter(|g| g.len() == n) {
                    grades = Some(g);
                    break;
                }
            }
            let mut p = progress.lock().unwrap();
            match grades {
                Some(g) => {
                    let line = json!({"q": qid, "g": g}).to_string();
                    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(format!("{DIR}/qwen/{batch}.jsonl")) {
                        let _ = writeln!(f, "{line}");
                    }
                    p.0 += 1;
                }
                None => p.1 += 1,
            }
            if (p.0 + p.1) % 50 == 0 {
                println!("{}/{total} graded, {} failed", p.0, p.1);
            }
        })
    }).collect();
    for h in handles {
        let _ = h.join();
    }
    let p = progress.lock().unwrap();
    println!("qwen graded {} of {total}; {} failed (run again to retry them)", p.0, p.1);
    Ok(if p.0 > 0 || total == 0 { 0 } else { 1 })
}

pub fn run(a: &[&str]) -> Result<u8, String> {
    let number = |k: &str| k.parse::<usize>().map_err(|_| format!("{k:?} must be a number"));
    match a {
        ["show", batch] => show(batch, 0),
        ["show", batch, "--start", k] => show(batch, number(k)?),
        ["save", batch] => save(batch),
        ["status"] => status("opus"),
        ["status", grader] => status(grader),
        ["qwen"] => qwen(4, None),
        ["qwen", "--workers", w] => qwen(number(w)?, None),
        ["qwen", "--limit", n] => qwen(4, Some(number(n)?)),
        _ => Err("grades show BATCH [--start K] | save BATCH | status [GRADER] | qwen [--workers N | --limit N]".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::cut;

    #[test]
    fn cut_at_a_word() {
        assert_eq!(cut("short text", 800), "short text");
        assert_eq!(cut("one two three four", 10), "one two ...");
    }
}
