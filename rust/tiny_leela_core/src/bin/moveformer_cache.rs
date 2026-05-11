use serde_json::{json, Value};
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};
use tiny_leela_core::{
    encode_moveformer_legal_inputs, for_each_jsonl_line, move_to_action_id, parse_fen,
};

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

fn square_index(s: &str) -> Option<u32> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some((b[0] - b'a') as u32 + ((b[1] - b'1') as u32) * 8)
}

fn action_id_from_uci(uci: &str) -> Option<i64> {
    if uci.len() < 4 {
        return None;
    }
    let from = square_index(&uci[0..2])?;
    let to = square_index(&uci[2..4])?;
    let promo = if uci.len() >= 5 {
        match uci.as_bytes()[4].to_ascii_lowercase() {
            b'n' => 1,
            b'b' => 2,
            b'r' => 3,
            b'q' => 4,
            _ => 0,
        }
    } else {
        0
    };
    Some(((from * 64 + to) * 5 + promo) as i64)
}

fn write_atomic(path: &str, body: &str) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let tmp = format!("{path}.tmp-{}", std::process::id());
    fs::write(&tmp, body).expect("write temp output");
    fs::rename(&tmp, path).expect("rename temp output");
}

fn parse_row(line: &str) -> Option<(String, i64, [f32; 3], f32)> {
    if line.trim().is_empty() {
        return None;
    }
    let row: Value = serde_json::from_str(line).ok()?;
    let fen = row.get("fen")?.as_str()?.to_string();
    let policy = row.get("policy")?.as_object()?;
    if policy.len() != 1 {
        return None;
    }
    let uci = policy.keys().next()?;
    let target = action_id_from_uci(uci)?;
    let mut wdl = [0.25f32, 0.5, 0.25];
    if let Some(xs) = row.get("wdl").and_then(|v| v.as_array()) {
        for i in 0..3.min(xs.len()) {
            if let Some(v) = xs[i].as_f64() {
                wdl[i] = v as f32;
            }
        }
    }
    let q = row
        .get("q")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(wdl[0] - wdl[2]);
    Some((fen, target, wdl, q))
}

fn main() {
    let input = arg("--input", "");
    let out = arg("--out", "artifacts/cache_moveformer_rust/shard_0000");
    let manifest_out = arg("--manifest-out", "");
    let dataset_manifest = arg("--dataset-manifest", "unknown");
    let max_rows: usize = arg("--max-rows", "0").parse().unwrap_or(0);
    let max_legal_moves: usize = arg("--max-legal-moves", "128").parse().unwrap_or(128);
    let num_features: usize = arg("--num-move-features", "20").parse().unwrap_or(20);
    if input.is_empty() {
        panic!("--input is required");
    }
    let inputs: Vec<String> = input
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let mut rows = 0usize;
    let mut bad = 0usize;
    for path in &inputs {
        for_each_jsonl_line(path, |line| {
            if max_rows > 0 && rows >= max_rows {
                return Ok(false);
            }
            if parse_row(line).is_some() {
                rows += 1;
            } else {
                bad += 1;
            }
            Ok(true)
        })
        .expect("stream MoveFormer input");
    }

    let out_path = PathBuf::from(&out);
    let tmp_path = out_path.with_file_name(format!(
        ".{}.tmp-{}",
        out_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("moveformer_cache"),
        std::process::id()
    ));
    if tmp_path.exists() {
        fs::remove_dir_all(&tmp_path).expect("remove stale temp output dir");
    }
    fs::create_dir_all(&tmp_path).expect("create temp output dir");
    let mut target_file =
        fs::File::create(tmp_path.join("target_action_id.int64")).expect("create target");
    let mut slot_file =
        fs::File::create(tmp_path.join("target_legal_slot.int16")).expect("create slot");
    let mut wdl_file = fs::File::create(tmp_path.join("wdl.float32")).expect("create wdl");
    let mut q_file = fs::File::create(tmp_path.join("q.float32")).expect("create q");
    let mut ids_file =
        fs::File::create(tmp_path.join("legal_action_ids.int64")).expect("create ids");
    let mut feat_file =
        fs::File::create(tmp_path.join("legal_features.float32")).expect("create features");
    let mut mask_file = fs::File::create(tmp_path.join("legal_mask.float32")).expect("create mask");
    let mut written = 0usize;
    let mut target_found = 0usize;
    let mut truncated = 0usize;
    for path in &inputs {
        for_each_jsonl_line(path, |line| {
            if written >= rows {
                return Ok(false);
            }
            let Some((fen, target, wdl, q)) = parse_row(line) else {
                return Ok(true);
            };
            let Ok(board) = parse_fen(&fen) else {
                return Ok(true);
            };
            let (moves, ids, features, mask) =
                encode_moveformer_legal_inputs(&board, max_legal_moves, num_features);
            if moves.len() > max_legal_moves {
                truncated += 1;
            }
            let mut slot = -1i16;
            for (j, mv) in moves.iter().take(max_legal_moves).enumerate() {
                if move_to_action_id(*mv) as i64 == target {
                    slot = j as i16;
                    target_found += 1;
                    break;
                }
            }
            target_file
                .write_all(&target.to_le_bytes())
                .expect("write target");
            slot_file
                .write_all(&slot.to_le_bytes())
                .expect("write slot");
            for v in wdl {
                wdl_file.write_all(&v.to_le_bytes()).expect("write wdl");
            }
            q_file.write_all(&q.to_le_bytes()).expect("write q");
            for id in ids {
                ids_file.write_all(&id.to_le_bytes()).expect("write ids");
            }
            for v in features {
                feat_file
                    .write_all(&v.to_le_bytes())
                    .expect("write features");
            }
            for v in mask {
                mask_file.write_all(&v.to_le_bytes()).expect("write mask");
            }
            written += 1;
            Ok(true)
        })
        .expect("stream MoveFormer input");
    }
    let meta = json!({
        "format": "moveformer_sidecar_cache_rust_v1",
        "rows": written,
        "max_legal_moves": max_legal_moves,
        "num_move_features": num_features,
        "policy_target_legal_rate": target_found as f64 / written.max(1) as f64,
        "legal_truncation_rate": truncated as f64 / written.max(1) as f64,
        "bad_or_skipped_rows": bad,
        "producer": { "language": "rust", "binary": "tiny-leela-rust-moveformer-cache" },
    });
    fs::write(
        tmp_path.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .expect("write meta");
    fs::write(tmp_path.join("_SUCCESS"), b"ok\n").expect("write success marker");
    drop((
        target_file,
        slot_file,
        wdl_file,
        q_file,
        ids_file,
        feat_file,
        mask_file,
    ));
    if out_path.exists() {
        fs::remove_dir_all(&out_path).expect("remove existing output dir");
    }
    fs::rename(&tmp_path, &out_path).expect("publish output dir");
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "cache_manifest_v1",
            "format": "moveformer_sidecar_cache_rust_v1",
            "source": { "dataset_manifest": dataset_manifest, "inputs": inputs },
            "producer": { "language": "rust", "binary": "tiny-leela-rust-moveformer-cache", "contract_versions": ["cache_manifest_v1"] },
            "arrays": [
                { "path": format!("{out}/legal_action_ids.int64"), "dtype": "int64", "shape": [written, max_legal_moves], "endianness": "little" },
                { "path": format!("{out}/legal_features.float32"), "dtype": "float32", "shape": [written, max_legal_moves, num_features], "endianness": "little" },
                { "path": format!("{out}/legal_mask.float32"), "dtype": "float32", "shape": [written, max_legal_moves], "endianness": "little" }
            ],
            "shards": [out],
            "validation": { "rows": { "total": written, "bad_or_skipped": bad }, "max_legal_moves": max_legal_moves },
        });
        write_atomic(
            &manifest_out,
            &(serde_json::to_string_pretty(&manifest).unwrap() + "\n"),
        );
    }
    println!("METRIC moveformer_cache_rows={written}");
    println!(
        "METRIC moveformer_cache_target_legal_rate={:.6}",
        target_found as f64 / written.max(1) as f64
    );
}
