use serde_json::{json, Value};
use std::{env, fs, path::Path, process::Command};

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

fn flag(name: &str) -> bool {
    env::args().any(|a| a == name)
}

fn sha256(path: &str) -> Option<String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .or_else(|_| {
            Command::new("shasum")
                .arg("-a")
                .arg("256")
                .arg(path)
                .output()
        })
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
}

fn meta_str<'a>(meta: &'a Value, key: &str) -> Option<&'a str> {
    meta.get(key).and_then(Value::as_str)
}
fn meta_u64(meta: &Value, key: &str) -> Option<u64> {
    meta.get(key).and_then(Value::as_u64)
}

fn model_arch(meta: &Value) -> &'static str {
    let arch = meta_str(meta, "architecture").unwrap_or("");
    let kind = meta_str(meta, "kind").unwrap_or("");
    if kind.contains("squareformer") || arch.contains("square") {
        "squareformer"
    } else if arch.contains("move_token")
        || arch.contains("move_transformer")
        || kind.contains("moveformer")
    {
        "moveformer"
    } else if arch.contains("residual") || arch.contains("cnn") || kind.contains("cnn") {
        "cnn"
    } else {
        "other"
    }
}

fn runtime_for(target: &str, threads: u64) -> Value {
    match target {
        "browser_webgpu" => {
            json!({"name":"onnxruntime-web","execution_providers":["webgpu","wasm"],"host_language":"typescript","threads":threads})
        }
        "browser_wasm" => {
            json!({"name":"onnxruntime-web","execution_providers":["wasm"],"host_language":"typescript","threads":threads})
        }
        "local_cuda" => {
            json!({"name":"onnxruntime-native","execution_providers":["CUDAExecutionProvider","CPUExecutionProvider"],"host_language":"rust","threads":threads})
        }
        "aws_batch_gpu" => {
            json!({"name":"onnxruntime-native","execution_providers":["CUDAExecutionProvider","CPUExecutionProvider"],"host_language":"mixed","threads":threads})
        }
        "mac_mini_native" | "native_cpu" | "aws_batch_cpu" => {
            json!({"name":"onnxruntime-native","execution_providers":["CPUExecutionProvider"],"host_language":"rust","threads":threads})
        }
        other => {
            json!({"name":"unknown","execution_providers":[other],"host_language":"mixed","threads":threads})
        }
    }
}

fn precision_for(model: &str, override_precision: &str) -> String {
    if !override_precision.is_empty() {
        return override_precision.to_string();
    }
    let lower = model.to_ascii_lowercase();
    if lower.contains("int8") {
        "dynamic_int8".to_string()
    } else if lower.contains("fp16") {
        "fp16".to_string()
    } else {
        "fp32".to_string()
    }
}

fn input_for(meta: &Value) -> Value {
    let arch = model_arch(meta);
    let encoding = match arch {
        "moveformer" => "board_planes_plus_legal_action_features",
        "squareformer" => {
            if meta_str(meta, "input_mode") == Some("embedding")
                || meta_str(meta, "input_format")
                    .unwrap_or("")
                    .contains("compact")
            {
                "compact_square_tokens"
            } else {
                "square_feature_tokens"
            }
        }
        "cnn" => "board_planes",
        _ => "unknown",
    };
    let legal_bucket = meta_u64(meta, "onnx_fixed_legal_moves")
        .or_else(|| meta_u64(meta, "max_legal_moves"))
        .map(|v| format!("k{v}"))
        .unwrap_or_else(|| {
            if meta.get("onnx_dynamic_legal").and_then(Value::as_bool) == Some(true) {
                "dynamic".to_string()
            } else {
                "none".to_string()
            }
        });
    json!({
        "encoding": encoding,
        "history_plies": meta_u64(meta, "history_plies").unwrap_or(0),
        "input_planes": meta_u64(meta, "input_planes"),
        "legal_bucket": legal_bucket,
        "batch_axis": meta.get("onnx_dynamic_batch").and_then(Value::as_bool).unwrap_or(true)
    })
}

fn outputs_for(meta: &Value) -> Vec<String> {
    if let Some(outputs) = meta.get("outputs").and_then(Value::as_array) {
        let vals: Vec<String> = outputs
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        if !vals.is_empty() {
            return vals;
        }
    }
    let mut out = match model_arch(meta) {
        "moveformer" => vec!["policy_logits_legal".to_string(), "wdl_logits".to_string()],
        "squareformer" => vec!["policy".to_string(), "wdl".to_string()],
        _ => vec!["policy".to_string(), "wdl".to_string()],
    };
    if meta.get("av_head_exported").and_then(Value::as_bool) == Some(true) {
        out.push("action_values".to_string());
    }
    if let Some(aux) = meta.get("aux_heads_exported").and_then(Value::as_array) {
        for name in aux.iter().filter_map(Value::as_str) {
            if !out.iter().any(|x| x == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn model_id_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string()
}

fn target_known_issues(target: &str) -> Vec<String> {
    match target {
        "browser_webgpu" => vec!["Requires real browser WebGPU smoke; Node ORT Web is not sufficient.".to_string()],
        "local_cuda" => vec!["Respect GPU-priority rules; do not contend with active training.".to_string()],
        "aws_batch_cpu" | "aws_batch_gpu" => vec!["Cloud smoke should emit compressed .jsonl.zst or a target-card artifact and stay inside budget guardrails.".to_string()],
        _ => Vec::new(),
    }
}

fn main() {
    let model = arg("--model", "");
    let meta_path = arg("--meta", "");
    let target = arg("--target", "native_cpu");
    let out = arg("--out", "");
    let label = arg("--label", "");
    let precision = precision_for(&model, &arg("--precision", ""));
    let threads: u64 = arg("--threads", "0").parse().unwrap_or(0);
    if model.is_empty() || meta_path.is_empty() {
        eprintln!("usage: tiny-leela-rust-onnx-card --model model.onnx --meta model.meta.json --target browser_wasm|browser_webgpu|local_cuda|mac_mini_native|aws_batch_cpu|native_cpu [--out card.json]");
        std::process::exit(2);
    }
    let meta_text = fs::read_to_string(&meta_path).expect("read meta json");
    let meta: Value = serde_json::from_str(&meta_text).expect("parse meta json");
    let model_bytes = fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
    let card = json!({
        "schema": "export_target_card_v1",
        "model": {
            "id": if label.is_empty() { model_id_from_path(&model) } else { label.clone() },
            "architecture": model_arch(&meta),
            "checkpoint": meta_str(&meta, "checkpoint").or_else(|| meta_str(&meta, "source_checkpoint")),
            "model_sha256": sha256(&model),
            "meta_sha256": sha256(&meta_path)
        },
        "artifact": {
            "onnx": model,
            "meta": meta_path,
            "onnx_bytes": model_bytes,
            "external_data": false,
            "simplified": model_id_from_path(&arg("--model", "")).contains("onnxsim"),
            "quantized_from": meta_str(&meta, "quantized_from")
        },
        "target": target,
        "runtime": runtime_for(&arg("--target", "native_cpu"), threads),
        "precision": precision,
        "input": input_for(&meta),
        "outputs": outputs_for(&meta),
        "parity": {"status":"pending", "required_before_promotion": true},
        "benchmarks": {"status":"pending", "required_before_promotion": true},
        "known_issues": target_known_issues(&arg("--target", "native_cpu"))
    });
    let text = serde_json::to_string_pretty(&card).expect("serialize card");
    if out.is_empty() || flag("--stdout") {
        println!("{text}");
    }
    if !out.is_empty() {
        if let Some(parent) = Path::new(&out).parent() {
            fs::create_dir_all(parent).expect("create output parent");
        }
        fs::write(&out, format!("{text}\n")).expect("write card");
        eprintln!("wrote {out}");
    }
}
