use serde_json::Value;
use std::{collections::BTreeMap, env, fs};
use tiny_leela_core::frozen_conv_student_features;

fn arg(name: &str, fallback: &str) -> String {
    let prefix = format!("{name}=");
    let args: Vec<String> = env::args().collect();
    for (i, value) in args.iter().enumerate() {
        if value.starts_with(&prefix) { return value[prefix.len()..].to_string(); }
        if value == name { if let Some(next) = args.get(i + 1) { return next.clone(); } }
    }
    fallback.to_string()
}

fn main() {
    let arch = arg("--arch", "64x6");
    let out = arg("--out", "artifacts/cache/conv_features_64x6.json");
    let inputs = arg("--inputs", "data/teacher_labels.jsonl,data/stockfish_teacher_labels.jsonl");
    let parts: Vec<_> = arch.split('x').collect();
    let channels: usize = parts.first().and_then(|v| v.parse().ok()).unwrap_or(64);
    let layers: usize = parts.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let mut cache: BTreeMap<String, Vec<f32>> = fs::read_to_string(&out)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let before = cache.len();
    for path in inputs.split(',').filter(|s| !s.is_empty()) {
        let Ok(text) = fs::read_to_string(path) else { continue; };
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(row) = serde_json::from_str::<Value>(line) else { continue; };
            let Some(fen) = row.get("fen").and_then(|v| v.as_str()) else { continue; };
            cache.entry(fen.to_string()).or_insert_with(|| frozen_conv_student_features(fen, channels, layers));
        }
    }
    if let Some(parent) = std::path::Path::new(&out).parent() { fs::create_dir_all(parent).expect("create cache dir"); }
    fs::write(&out, serde_json::to_string(&cache).expect("serialize cache")).expect("write cache");
    println!("METRIC rust_feature_cache_entries={}", cache.len());
    println!("METRIC rust_feature_cache_new_entries={}", cache.len().saturating_sub(before));
}
