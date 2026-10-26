// Grades for training RiverGuide's reranker, served to Claude Opus 5.5 graders and kept as they
// come (rerank_data.py writes the questions; train_reranker.py reads the grades):
//   riverguide-lab grades show BATCH [--start K]   five items from K: a made-up first message and
//                                                  v2's top 20 passages for it (the guide first)
//   riverguide-lab grades save BATCH               JSON lines on stdin, {"q": ID, "g": [20 grades]},
//                                                  checked and appended to data/rerank/opus/BATCH.jsonl
//   riverguide-lab grades status                   items graded in each batch
// Run from the repo root. A grade is 2 (answers what the person needs), 1 (related, not the
// answer) or 0 (doesn't help).
use crate::read_lossy;
use serde_json::{json, Value};
use std::{collections::HashMap, fs, io::Read, io::Write};

const SHOW: usize = 5;
const CHARS: usize = 800;
const DIR: &str = "data/rerank";

const GUIDE: &str = "You grade search results for RiverGuide, Indian River State College's student assistant, to train
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
    read_lossy(&format!("{DIR}/opus/{batch}.jsonl")).lines()
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

fn show(batch: &str, start: usize) -> Result<u8, String> {
    let (items, ids) = (questions()?, batch_ids(batch)?);
    let docs: Vec<Value> = read_lossy("data/docs.jsonl").lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    let mut out = String::new();
    if start == 0 {
        out.push_str(GUIDE);
        out.push('\n');
    }
    let end = (start + SHOW).min(ids.len());
    for qid in &ids[start.min(end)..end] {
        let item = items.get(qid).ok_or_else(|| format!("item {qid} isn't in {DIR}/questions.jsonl"))?;
        out.push_str(&format!("=== item {qid}\nMessage: {}\n\n", item.text));
        for (j, c) in item.candidates.iter().enumerate() {
            let d = docs.get(*c).ok_or_else(|| format!("chunk {c} isn't in data/docs.jsonl"))?;
            out.push_str(&format!("[{}] {}\n{}\n\n", j + 1, d["url"].as_str().unwrap_or(""), cut(d["text"].as_str().unwrap_or(""), CHARS)));
        }
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

fn status() -> Result<u8, String> {
    let mut batches: Vec<String> = fs::read_dir(DIR).map_err(|e| format!("{DIR}: {e}"))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".ids").map(String::from)).collect();
    batches.sort();
    let (mut all, mut done) = (0, 0);
    for b in &batches {
        let (ids, g) = (batch_ids(b)?, graded(b));
        let n = ids.iter().filter(|q| g.contains_key(q)).count();
        println!("{b}: {n}/{}", ids.len());
        all += ids.len();
        done += n;
    }
    println!("graded {done}/{all}");
    Ok(0)
}

pub fn run(a: &[&str]) -> Result<u8, String> {
    match a {
        ["show", batch] => show(batch, 0),
        ["show", batch, "--start", k] => show(batch, k.parse().map_err(|_| format!("--start must be a number, not {k:?}"))?),
        ["save", batch] => save(batch),
        ["status"] => status(),
        _ => Err("grades show BATCH [--start K] | save BATCH | status".into()),
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
