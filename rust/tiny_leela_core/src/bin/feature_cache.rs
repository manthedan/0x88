use serde_json::{json, Value};
use std::{collections::BTreeMap, env, fs, path::Path};
use tiny_leela_core::frozen_conv_student_features;

fn arg(name: &str, fallback: &str) -> String {
    let prefix = format!("{name}=");
    let args: Vec<String> = env::args().collect();
    for (i, value) in args.iter().enumerate() {
        if value.starts_with(&prefix) {
            return value[prefix.len()..].to_string();
        }
        if value == name {
            if let Some(next) = args.get(i + 1) {
                return next.clone();
            }
        }
    }
    fallback.to_string()
}

fn write_atomic(path: &str, body: &str) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let tmp = format!("{path}.tmp-{}", std::process::id());
    fs::write(&tmp, body).expect("write temp output");
    fs::rename(&tmp, path).expect("rename temp output");
}

fn main() {
    let arch = arg("--arch", "64x6");
    let out = arg("--out", "artifacts/cache/conv_features_64x6.json");
    let manifest_out = arg("--manifest-out", "");
    let dataset_manifest = arg("--dataset-manifest", "unknown");
    let git_commit = arg("--git-commit", "unknown");
    let inputs = arg(
        "--inputs",
        "data/teacher_labels.jsonl,data/stockfish_teacher_labels.jsonl",
    );
    let parts: Vec<_> = arch.split('x').collect();
    let channels: usize = parts.first().and_then(|v| v.parse().ok()).unwrap_or(64);
    let layers: usize = parts.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let feature_dim = 2 + channels * 3;
    let mut cache: BTreeMap<String, Vec<f32>> = fs::read_to_string(&out)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let before = cache.len();
    for path in inputs.split(',').filter(|s| !s.is_empty()) {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(row) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let Some(fen) = row.get("fen").and_then(|v| v.as_str()) else {
                continue;
            };
            cache
                .entry(fen.to_string())
                .or_insert_with(|| frozen_conv_student_features(fen, channels, layers));
        }
    }
    write_atomic(
        &out,
        &serde_json::to_string(&cache).expect("serialize cache"),
    );
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "cache_manifest_v1",
            "format": "frozen_conv_features_json_v1",
            "source": {
                "dataset_manifest": dataset_manifest,
                "inputs": inputs.split(',').filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            },
            "producer": {
                "language": "rust",
                "binary": "tiny-leela-rust-feature-cache",
                "git_commit": git_commit,
                "contract_versions": ["cache_manifest_v1"],
            },
            "arrays": [{
                "path": out,
                "dtype": "f32",
                "shape": [cache.len(), feature_dim],
                "layout": "fen_to_feature_vector_json_object",
                "endianness": "native",
            }],
            "shards": [out],
            "validation": {
                "rows": { "total": cache.len(), "new": cache.len().saturating_sub(before) },
                "feature_dim": feature_dim,
                "arch": arch,
            },
        });
        write_atomic(
            &manifest_out,
            &(serde_json::to_string_pretty(&manifest).expect("serialize manifest") + "\n"),
        );
    }
    println!("METRIC rust_feature_cache_entries={}", cache.len());
    println!(
        "METRIC rust_feature_cache_new_entries={}",
        cache.len().saturating_sub(before)
    );
}
