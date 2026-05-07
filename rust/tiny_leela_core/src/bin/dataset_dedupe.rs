use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    max_rows: Option<usize>,
    max_rows_per_opening: Option<usize>,
    opening_prefix_plies: usize,
}

fn usage() -> ! {
    eprintln!("Usage: tiny-leela-rust-dataset-dedupe --input in.jsonl --output out.jsonl [--max-rows N] [--opening-prefix-plies N] [--max-rows-per-opening N]");
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut it = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut max_rows = None;
    let mut max_rows_per_opening = None;
    let mut opening_prefix_plies = 12usize;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--max-rows" => max_rows = it.next().and_then(|s| s.parse().ok()),
            "--max-rows-per-opening" => max_rows_per_opening = it.next().and_then(|s| s.parse().ok()),
            "--opening-prefix-plies" => opening_prefix_plies = it.next().and_then(|s| s.parse().ok()).unwrap_or(12),
            "--help" | "-h" => usage(),
            _ => usage(),
        }
    }
    Args { input: input.unwrap_or_else(|| usage()), output: output.unwrap_or_else(|| usage()), max_rows, max_rows_per_opening, opening_prefix_plies }
}

fn hash64<T: Hash>(x: &T) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

fn fen_key(fen: &str) -> Option<String> {
    let mut parts = fen.split_whitespace();
    let a = parts.next()?;
    let b = parts.next()?;
    let c = parts.next()?;
    let d = parts.next()?;
    Some(format!("{} {} {} {}", a, b, c, d))
}

fn ply(v: &Value) -> usize {
    for k in ["ply", "move_ply", "num_ply"] {
        if let Some(n) = v.get(k).and_then(|x| x.as_u64()) { return n as usize; }
    }
    0
}

fn game_id(v: &Value) -> String {
    let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("");
    if let Some((prefix, suffix)) = id.rsplit_once('_') {
        if suffix.chars().all(|c| c.is_ascii_digit()) { return prefix.to_string(); }
    }
    id.to_string()
}

fn opening_prefix(v: &Value, prefix_plies: usize) -> String {
    if let Some(arr) = v.get("moves").and_then(|x| x.as_array()) {
        let s: Vec<&str> = arr.iter().take(prefix_plies).filter_map(|x| x.as_str()).collect();
        if !s.is_empty() { return s.join("_"); }
    }
    // Fallback: cap by early game id bucket if full move list is unavailable.
    let p = ply(v).min(prefix_plies);
    format!("{}:{}", game_id(v), p)
}

fn main() -> io::Result<()> {
    let args = parse_args();
    let input = BufReader::new(File::open(&args.input)?);
    if let Some(parent) = args.output.parent() { std::fs::create_dir_all(parent)?; }
    let mut out = BufWriter::new(File::create(&args.output)?);

    let mut seen: HashSet<u64> = HashSet::new();
    let mut opening_counts: HashMap<u64, usize> = HashMap::new();
    let mut rows_in = 0usize;
    let mut rows_out = 0usize;
    let mut bad_json = 0usize;
    let mut missing_fen = 0usize;
    let mut duplicate_positions = 0usize;
    let mut skipped_opening_cap = 0usize;

    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        rows_in += 1;
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => { bad_json += 1; continue; }
        };
        let fen = match v.get("fen").and_then(|x| x.as_str()).and_then(fen_key) {
            Some(k) => k,
            None => { missing_fen += 1; continue; }
        };
        let h = hash64(&fen);
        if !seen.insert(h) { duplicate_positions += 1; continue; }
        if let Some(cap) = args.max_rows_per_opening {
            let oh = hash64(&opening_prefix(&v, args.opening_prefix_plies));
            let c = opening_counts.entry(oh).or_insert(0);
            if *c >= cap { skipped_opening_cap += 1; continue; }
            *c += 1;
        }
        writeln!(out, "{}", line)?;
        rows_out += 1;
        if args.max_rows.map_or(false, |m| rows_out >= m) { break; }
    }
    out.flush()?;
    eprintln!("METRIC dedupe_rows_in={}", rows_in);
    eprintln!("METRIC dedupe_rows_out={}", rows_out);
    eprintln!("METRIC dedupe_bad_json={}", bad_json);
    eprintln!("METRIC dedupe_missing_fen={}", missing_fen);
    eprintln!("METRIC dedupe_duplicate_positions={}", duplicate_positions);
    eprintln!("METRIC dedupe_skipped_opening_cap={}", skipped_opening_cap);
    eprintln!("METRIC dedupe_unique_positions={}", seen.len());
    Ok(())
}
