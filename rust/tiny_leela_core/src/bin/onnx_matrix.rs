use serde_json::json;
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

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn card_cmd(model: &str, meta: &str, target: &str, label: &str, out: &str) -> String {
    format!(
        "cargo run --release --manifest-path rust/tiny_leela_core/Cargo.toml --bin tiny-leela-rust-onnx-card -- --model {} --meta {} --target {} --label {} --out {}",
        shell_quote(model), shell_quote(meta), shell_quote(target), shell_quote(label), shell_quote(out)
    )
}

fn smoke_cmd(
    model: &str,
    meta: &str,
    target: &str,
    positions: u32,
    repeats: u32,
    batches: &str,
    label: &str,
) -> String {
    match target {
        "browser_wasm" => format!(
            "TINY_LEELA_ORT_EP=wasm node --experimental-strip-types eval/onnx_inference_benchmark.mjs --model {} --meta {} --label {} --positions {} --repeats {} --batches {}",
            shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
        ),
        "browser_webgpu" => format!(
            "TINY_LEELA_ORT_EP=webgpu,wasm node --experimental-strip-types eval/onnx_inference_benchmark.mjs --model {} --meta {} --label {} --positions {} --repeats {} --batches {} # or run in agent_browser with a real WebGPU page",
            shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
        ),
        "local_cuda" => format!(
            ".venv-onnx/bin/python eval/onnx_native_inference_benchmark.py --model {} --meta {} --provider CUDAExecutionProvider --provider CPUExecutionProvider --require-provider CUDAExecutionProvider --label {} --positions {} --repeats {} --batches {}",
            shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
        ),
        "mac_mini_native" => {
            let remote = format!(
                "cd tiny_leela && .venv-onnx/bin/python eval/onnx_native_inference_benchmark.py --model {} --meta {} --provider CPUExecutionProvider --label {} --positions {} --repeats {} --batches {}",
                shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
            );
            format!("ssh mac-mini {}", shell_quote(&remote))
        },
        "aws_batch_cpu" => format!(
            "AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 scripts/submit_onnx_inference_smoke_aws.sh --model {} --meta {} --label {} --positions {} --repeats {} --batches {}",
            shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
        ),
        "native_cpu" => format!(
            ".venv-onnx/bin/python eval/onnx_native_inference_benchmark.py --model {} --meta {} --provider CPUExecutionProvider --label {} --positions {} --repeats {} --batches {}",
            shell_quote(model), shell_quote(meta), shell_quote(label), positions, repeats, shell_quote(batches)
        ),
        other => format!("# no smoke command template for target {other}"),
    }
}

fn main() {
    let model = arg("--model", "");
    let meta = arg("--meta", "");
    let out_dir = arg("--out-dir", "artifacts/inference_target_matrix/manual");
    let targets_s = arg(
        "--targets",
        "browser_webgpu,local_cuda,mac_mini_native,aws_batch_cpu",
    );
    let label = arg(
        "--label",
        Path::new(&model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
    );
    let positions: u32 = arg("--positions", "16").parse().unwrap_or(16);
    let repeats: u32 = arg("--repeats", "2").parse().unwrap_or(2);
    let batches = arg("--batches", "1,4,16");
    if model.is_empty() || meta.is_empty() {
        eprintln!("usage: tiny-leela-rust-onnx-matrix --model model.onnx --meta model.meta.json [--targets browser_webgpu,local_cuda,mac_mini_native,aws_batch_cpu] [--out-dir artifacts/...]");
        std::process::exit(2);
    }
    fs::create_dir_all(&out_dir).expect("create output dir");
    let mut commands = Vec::new();
    let mut target_cards = Vec::new();
    for target in targets_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let card_path = format!("{out_dir}/{label}.{target}.export_target_card.json");
        target_cards.push(json!({"target": target, "path": card_path}));
        commands.push(json!({"target": target, "kind": "card", "command": card_cmd(&model, &meta, target, &label, &card_path)}));
        commands.push(json!({"target": target, "kind": "smoke_bench", "command": smoke_cmd(&model, &meta, target, positions, repeats, &batches, &format!("{label}_{target}"))}));
    }
    let plan = json!({
        "schema": "onnx_export_target_matrix_plan_v1",
        "model": model,
        "meta": meta,
        "label": label,
        "out_dir": out_dir,
        "positions": positions,
        "repeats": repeats,
        "batches": batches,
        "target_cards": target_cards,
        "commands": commands,
        "notes": [
            "Generated by tiny-leela-rust-onnx-matrix; commands are explicit so GPU/cloud/browser smoke remains opt-in.",
            "Run target-card commands first, then smoke commands as each host is available.",
            "Browser WebGPU requires a real browser context; local Node ORT WebGPU may not be representative."
        ]
    });
    let plan_path = format!(
        "{}/plan.json",
        plan.get("out_dir").and_then(|v| v.as_str()).unwrap_or(".")
    );
    fs::write(
        &plan_path,
        format!("{}\n", serde_json::to_string_pretty(&plan).unwrap()),
    )
    .expect("write plan");
    let shell_path = format!(
        "{}/commands.sh",
        plan.get("out_dir").and_then(|v| v.as_str()).unwrap_or(".")
    );
    let mut shell = String::from("#!/usr/bin/env bash\nset -euo pipefail\n\n");
    shell.push_str("# Review before running; GPU/cloud/browser targets are opt-in.\n");
    for cmd in plan.get("commands").and_then(|v| v.as_array()).unwrap() {
        shell.push_str("\n# ");
        shell.push_str(
            cmd.get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("target"),
        );
        shell.push(' ');
        shell.push_str(cmd.get("kind").and_then(|v| v.as_str()).unwrap_or("cmd"));
        shell.push('\n');
        shell.push_str(
            cmd.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("# missing command"),
        );
        shell.push('\n');
    }
    fs::write(&shell_path, shell).expect("write shell commands");
    eprintln!("wrote {plan_path}");
    eprintln!("wrote {shell_path}");
    println!("METRIC onnx_matrix_targets={}", target_cards.len());
    println!("PLAN {plan_path}");
    println!("COMMANDS {shell_path}");
}
