use serde_json::{json, Map, Value};
use std::{env, fs, path::Path};

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

fn multi_arg(name: &str) -> Vec<String> {
    let prefix = format!("{name}=");
    let args: Vec<String> = env::args().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let value = &args[i];
        if value.starts_with(&prefix) {
            out.push(value[prefix.len()..].to_string());
        } else if value == name {
            if let Some(next) = args.get(i + 1) {
                out.push(next.clone());
                i += 1;
            }
        }
        i += 1;
    }
    out
}

fn read_json(path: &str) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn artifact_exists(path: &str) -> bool {
    !path.is_empty() && Path::new(path).exists()
}

fn infer_family(meta: Option<&Value>, fallback: &str) -> String {
    if !fallback.is_empty() {
        return fallback.to_string();
    }
    let kind = meta
        .and_then(|m| m.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let arch = meta
        .and_then(|m| m.get("architecture"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if kind.contains("squareformer") || arch.contains("square") {
        "SquareFormer".to_string()
    } else if kind.contains("moveformer") || arch.contains("move") {
        "MoveFormer".to_string()
    } else if arch.contains("cnn") || arch.contains("residual") {
        "ResidualCNN".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn merge_object(dst: &mut Map<String, Value>, patch: Map<String, Value>) {
    for (key, value) in patch {
        if !value.is_null() {
            dst.insert(key, value);
        }
    }
}

fn main() {
    let overrides = arg("--overrides", "eval/model_manifest_overrides.json");
    let out = arg("--out", &overrides);
    let model_id = arg("--model-id", "");
    let onnx = arg("--onnx", "");
    let meta_path = arg("--meta", "");
    let display_name = arg("--display-name", &model_id);
    let family_arg = arg("--family", "");
    let status = arg("--status", "completed");
    let policy_dataset = arg("--policy-dataset", "");
    let av_dataset = arg("--av-dataset", "");
    let notes = arg("--notes", "");
    let allow_missing = flag("--allow-missing");
    if model_id.is_empty() || onnx.is_empty() || meta_path.is_empty() {
        eprintln!("usage: tiny-leela-rust-model-manifest-update --model-id ID --onnx path.onnx --meta path.meta.json [--overrides eval/model_manifest_overrides.json] [--out path]");
        std::process::exit(2);
    }
    if !allow_missing {
        for path in [&onnx, &meta_path] {
            if !artifact_exists(path) {
                eprintln!("missing artifact {path}; pass --allow-missing to record it anyway");
                std::process::exit(3);
            }
        }
    }
    let meta_json = if artifact_exists(&meta_path) {
        Some(read_json(&meta_path))
    } else {
        None
    };
    let family = infer_family(meta_json.as_ref(), &family_arg);
    let mut doc = read_json(&overrides);
    if !doc.is_object() {
        doc = json!({"schema_version": 1, "models": []});
    }
    let root = doc.as_object_mut().expect("overrides root object");
    root.entry("schema_version".to_string()).or_insert(json!(1));
    let models = root
        .entry("models".to_string())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("models must be an array");
    let tags: Vec<Value> = multi_arg("--tag").into_iter().map(Value::String).collect();
    let mut training = Map::new();
    if !policy_dataset.is_empty() {
        training.insert("policy_dataset".to_string(), json!(policy_dataset));
    }
    if !av_dataset.is_empty() {
        training.insert("av_dataset".to_string(), json!(av_dataset));
    }
    if !notes.is_empty() {
        training.insert("notes".to_string(), json!(notes));
    }
    let patch = json!({
        "model_id": model_id,
        "display_name": display_name,
        "family": family,
        "status": status,
        "artifacts": { "onnx": onnx, "meta": meta_path },
        "training": Value::Object(training),
        "tags": tags,
    });
    let patch_obj = patch.as_object().unwrap().clone();
    let mut updated = false;
    for model in models.iter_mut() {
        if model.get("model_id").and_then(Value::as_str) == Some(&model_id) {
            let obj = model.as_object_mut().expect("model entry object");
            merge_object(obj, patch_obj.clone());
            updated = true;
            break;
        }
    }
    if !updated {
        models.push(Value::Object(patch_obj));
    }
    if let Some(parent) = Path::new(&out).parent() {
        fs::create_dir_all(parent).expect("create output parent");
    }
    fs::write(&out, serde_json::to_string_pretty(&doc).unwrap() + "\n").expect("write overrides");
    println!("METRIC model_manifest_update_model_id={model_id}");
    println!(
        "METRIC model_manifest_update_updated={}",
        if updated { 1 } else { 0 }
    );
    println!("METRIC model_manifest_update_out={out}");
}
