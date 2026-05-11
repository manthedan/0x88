use serde_json::{json, Value};
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};
use tiny_leela_core::{encode_squareformer_compact_input, parse_fen, SquareFormerEvaluatorMeta};

const SCHEMA_ACTION_VALUE: &str = "teacher.action_value.v1";
const POLICY_SIZE: usize = 4096 + 4096 * 4;

#[derive(Clone, Debug)]
struct Candidate {
    move_class: i64,
    value: f32,
    regret: f32,
    rank: i64,
    source_order: usize,
}

#[derive(Clone, Debug)]
struct Group {
    key: String,
    fen: String,
    history_fens: Vec<String>,
    candidates: Vec<Candidate>,
}

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

fn square_index(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some((b[0] - b'a') as i64 + ((b[1] - b'1') as i64) * 8)
}

fn chessbench_move_class(uci: &str) -> Option<i64> {
    if uci.len() < 4 {
        return None;
    }
    let from = square_index(&uci[0..2])?;
    let to = square_index(&uci[2..4])?;
    let ft = from * 64 + to;
    if uci.len() >= 5 {
        let promo = match uci.as_bytes()[4].to_ascii_lowercase() {
            b'n' => Some(0),
            b'b' => Some(1),
            b'r' => Some(2),
            b'q' => Some(3),
            _ => None,
        }?;
        Some(4096 + ft * 4 + promo)
    } else {
        Some(ft)
    }
}

fn input_lines(path: &str) -> impl Iterator<Item = String> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("open input {path}: {e}"));
    BufReader::new(file)
        .lines()
        .map(|l| l.expect("read input line"))
}

fn write_atomic(path: &str, body: &str) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let tmp = format!("{path}.tmp-{}", std::process::id());
    fs::write(&tmp, body).expect("write temp output");
    fs::rename(&tmp, path).expect("rename temp output");
}

fn f64_field(row: &Value, name: &str, fallback: f64) -> f64 {
    row.get(name).and_then(|v| v.as_f64()).unwrap_or(fallback)
}

fn i64_field(row: &Value, name: &str, fallback: i64) -> i64 {
    row.get(name).and_then(|v| v.as_i64()).unwrap_or(fallback)
}

fn history_fens(row: &Value, history: usize) -> Vec<String> {
    row.get("history_fens")
        .and_then(|v| v.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .take(history)
                .collect()
        })
        .unwrap_or_default()
}

fn candidate_from_row(
    row: &Value,
    source_order: usize,
) -> Option<(String, String, Vec<String>, Candidate)> {
    if row.get("schema").and_then(|v| v.as_str()) != Some(SCHEMA_ACTION_VALUE) {
        return None;
    }
    let fen = row.get("fen")?.as_str()?.to_string();
    let key = row
        .get("position_key")
        .and_then(|v| v.as_str())
        .unwrap_or(&fen)
        .to_string();
    let move_uci = row.get("move")?.as_str()?;
    let move_class = chessbench_move_class(move_uci)?;
    let value = f64_field(row, "value", 0.0).clamp(-1.0, 1.0) as f32;
    let regret = (f64_field(row, "regret_cp", 0.0) / 400.0) as f32;
    let rank = i64_field(row, "rank", 0);
    let history = history_fens(row, usize::MAX);
    Some((
        key,
        fen,
        history,
        Candidate {
            move_class,
            value,
            regret,
            rank,
            source_order,
        },
    ))
}

fn write_i64s(file: &mut fs::File, values: impl IntoIterator<Item = i64>) {
    for value in values {
        file.write_all(&value.to_le_bytes()).expect("write i64");
    }
}

fn write_f32s(file: &mut fs::File, values: impl IntoIterator<Item = f32>) {
    for value in values {
        file.write_all(&value.to_le_bytes()).expect("write f32");
    }
}

fn write_group(
    group: Group,
    token_file: &mut fs::File,
    moves_file: &mut fs::File,
    values_file: &mut fs::File,
    regrets_file: &mut fs::File,
    mask_file: &mut fs::File,
    meta: &SquareFormerEvaluatorMeta,
    max_candidates: usize,
) -> Option<usize> {
    if group.candidates.len() < 2 {
        return None;
    }
    let board = parse_fen(&group.fen).ok()?;
    let tokens = encode_squareformer_compact_input(&board, meta, &group.history_fens);
    for tok in tokens {
        token_file.write_all(&[tok as u8]).expect("write token");
    }
    let mut candidates = group.candidates;
    candidates.sort_by(|a, b| {
        let arank = if a.rank > 0 { a.rank } else { i64::MAX };
        let brank = if b.rank > 0 { b.rank } else { i64::MAX };
        arank
            .cmp(&brank)
            .then_with(|| a.source_order.cmp(&b.source_order))
            .then_with(|| {
                b.value
                    .partial_cmp(&a.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let keep = candidates.len().min(max_candidates);
    let mut moves = vec![0i64; max_candidates];
    let mut values = vec![0.0f32; max_candidates];
    let mut regrets = vec![0.0f32; max_candidates];
    let mut mask = vec![0.0f32; max_candidates];
    for (j, cand) in candidates.into_iter().take(max_candidates).enumerate() {
        moves[j] = cand.move_class;
        values[j] = cand.value;
        regrets[j] = cand.regret;
        mask[j] = 1.0;
    }
    write_i64s(moves_file, moves);
    write_f32s(values_file, values);
    write_f32s(regrets_file, regrets);
    write_f32s(mask_file, mask);
    Some(keep)
}

fn main() {
    let input = arg("--input", "");
    let out = arg("--out", "artifacts/cache_action_value_rust/shard_0000");
    let manifest_out = arg("--manifest-out", "");
    let dataset_manifest = arg("--dataset-manifest", "unknown");
    let max_positions: usize = arg("--max-positions", "0").parse().unwrap_or(0);
    let max_candidates: usize = arg("--max-candidates", "8").parse().unwrap_or(8);
    let history_plies: usize = arg("--history-plies", "2").parse().unwrap_or(2);
    if input.is_empty() {
        panic!("--input is required");
    }
    let inputs: Vec<String> = input
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let out_path = PathBuf::from(&out);
    let tmp_path = out_path.with_file_name(format!(
        ".{}.tmp-{}",
        out_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("action_value_cache"),
        std::process::id()
    ));
    if tmp_path.exists() {
        fs::remove_dir_all(&tmp_path).expect("remove stale temp output dir");
    }
    fs::create_dir_all(&tmp_path).expect("create temp output dir");
    let mut token_file = fs::File::create(tmp_path.join("tokens.uint8")).expect("create tokens");
    let mut moves_file =
        fs::File::create(tmp_path.join("candidate_moves.int64")).expect("create moves");
    let mut values_file =
        fs::File::create(tmp_path.join("candidate_values.float32")).expect("create values");
    let mut regrets_file =
        fs::File::create(tmp_path.join("candidate_regrets.float32")).expect("create regrets");
    let mut mask_file =
        fs::File::create(tmp_path.join("candidate_mask.float32")).expect("create mask");
    let meta = SquareFormerEvaluatorMeta {
        kind: "squareformer".to_string(),
        input_dim: None,
        token_features: Some(history_plies + 9),
        input_mode: Some("compact".to_string()),
        input_format: Some("compact_uint8_tokens".to_string()),
        policy_size: POLICY_SIZE,
        history_plies,
        av_head_exported: true,
        max_legal_moves: None,
        onnx_fixed_legal_moves: None,
    };
    let mut current: Option<Group> = None;
    let mut source_order = 0usize;
    let mut positions = 0usize;
    let mut candidate_rows = 0usize;
    let mut scanned_rows = 0usize;
    let mut malformed_rows = 0usize;
    let mut skipped_rows = 0usize;
    let mut bad_position_groups = 0usize;
    for path in &inputs {
        for line in input_lines(path) {
            if max_positions > 0 && positions >= max_positions {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            scanned_rows += 1;
            let row: Value = match serde_json::from_str(&line) {
                Ok(row) => row,
                Err(_) => {
                    malformed_rows += 1;
                    continue;
                }
            };
            source_order += 1;
            let Some((key, fen, hist, cand)) = candidate_from_row(&row, source_order) else {
                skipped_rows += 1;
                continue;
            };
            if current.as_ref().map(|g| g.key.as_str()) != Some(key.as_str()) {
                if let Some(group) = current.take() {
                    match write_group(
                        group,
                        &mut token_file,
                        &mut moves_file,
                        &mut values_file,
                        &mut regrets_file,
                        &mut mask_file,
                        &meta,
                        max_candidates,
                    ) {
                        Some(n) => {
                            positions += 1;
                            candidate_rows += n;
                        }
                        None => bad_position_groups += 1,
                    }
                    if max_positions > 0 && positions >= max_positions {
                        break;
                    }
                }
                current = Some(Group {
                    key,
                    fen,
                    history_fens: hist.into_iter().take(history_plies).collect(),
                    candidates: Vec::new(),
                });
            }
            if let Some(group) = current.as_mut() {
                group.candidates.push(cand);
            }
        }
    }
    if max_positions == 0 || positions < max_positions {
        if let Some(group) = current.take() {
            match write_group(
                group,
                &mut token_file,
                &mut moves_file,
                &mut values_file,
                &mut regrets_file,
                &mut mask_file,
                &meta,
                max_candidates,
            ) {
                Some(n) => {
                    positions += 1;
                    candidate_rows += n;
                }
                None => bad_position_groups += 1,
            }
        }
    }
    let cache_meta = json!({
        "format": "compact_action_value_cache_rust_v1",
        "source_schema": SCHEMA_ACTION_VALUE,
        "rows": positions,
        "candidate_rows": candidate_rows,
        "max_candidates": max_candidates,
        "history_plies": history_plies,
        "token_features": history_plies + 9,
        "policy_size": POLICY_SIZE,
        "source_shards": inputs,
        "scanned_rows": scanned_rows,
        "malformed_rows": malformed_rows,
        "skipped_rows": skipped_rows,
        "bad_position_groups": bad_position_groups,
        "producer": { "language": "rust", "binary": "tiny-leela-rust-action-value-cache" },
        "files": {
            "tokens": "tokens.uint8",
            "candidate_moves": "candidate_moves.int64",
            "candidate_values": "candidate_values.float32",
            "candidate_regrets": "candidate_regrets.float32",
            "candidate_mask": "candidate_mask.float32"
        }
    });
    fs::write(
        tmp_path.join("meta.json"),
        serde_json::to_string_pretty(&cache_meta).unwrap(),
    )
    .expect("write meta");
    fs::write(tmp_path.join("_SUCCESS"), b"ok\n").expect("write success marker");
    drop((token_file, moves_file, values_file, regrets_file, mask_file));
    if out_path.exists() {
        fs::remove_dir_all(&out_path).expect("remove existing output dir");
    }
    fs::rename(&tmp_path, &out_path).expect("publish output dir");
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "cache_manifest_v1",
            "format": "compact_action_value_cache_rust_v1",
            "source": { "dataset_manifest": dataset_manifest, "inputs": inputs },
            "producer": { "language": "rust", "binary": "tiny-leela-rust-action-value-cache", "contract_versions": ["cache_manifest_v1"] },
            "arrays": [
                { "path": format!("{out}/tokens.uint8"), "dtype": "uint8", "shape": [positions, 64, history_plies + 9] },
                { "path": format!("{out}/candidate_moves.int64"), "dtype": "int64", "shape": [positions, max_candidates], "endianness": "little" },
                { "path": format!("{out}/candidate_values.float32"), "dtype": "float32", "shape": [positions, max_candidates], "endianness": "little" },
                { "path": format!("{out}/candidate_regrets.float32"), "dtype": "float32", "shape": [positions, max_candidates], "endianness": "little" },
                { "path": format!("{out}/candidate_mask.float32"), "dtype": "float32", "shape": [positions, max_candidates], "endianness": "little" }
            ],
            "shards": [out],
            "validation": { "rows": { "total": positions, "candidates": candidate_rows, "skipped": skipped_rows, "malformed": malformed_rows } }
        });
        write_atomic(
            &manifest_out,
            &(serde_json::to_string_pretty(&manifest).unwrap() + "\n"),
        );
    }
    println!("METRIC action_value_cache_positions={positions}");
    println!("METRIC action_value_cache_candidate_rows={candidate_rows}");
}
