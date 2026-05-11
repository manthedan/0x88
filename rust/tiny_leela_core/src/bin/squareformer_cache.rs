use serde_json::{json, Value};
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

const FILES: &str = "abcdefgh";
const PIECES: &str = ".PNBRQKpnbrqk";

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

fn square_index(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 || !FILES.as_bytes().contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some((b[0] - b'a') as usize + ((b[1] - b'1') as usize) * 8)
}

fn move_class(uci: &str) -> Option<i64> {
    if uci.len() < 4 {
        return None;
    }
    let from = square_index(&uci[0..2])?;
    let to = square_index(&uci[2..4])?;
    if uci.len() >= 5 {
        let promo = match uci.as_bytes()[4].to_ascii_lowercase() {
            b'n' => Some(0),
            b'b' => Some(1),
            b'r' => Some(2),
            b'q' => Some(3),
            _ => None,
        };
        if let Some(p) = promo {
            return Some(4096 + ((from * 64 + to) * 4 + p) as i64);
        }
    }
    Some((from * 64 + to) as i64)
}

fn parse_board(fen: &str) -> Option<[u8; 64]> {
    let mut board = [0u8; 64];
    let placement = fen.split_whitespace().next()?;
    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return None;
    }
    for (rr, rank_s) in ranks.iter().enumerate() {
        let mut f = 0usize;
        let r = 7usize.saturating_sub(rr);
        for ch in rank_s.chars() {
            if ch.is_ascii_digit() {
                f += ch.to_digit(10)? as usize;
            } else if let Some(pi) = PIECES.find(ch) {
                if f >= 8 {
                    return None;
                }
                board[r * 8 + f] = pi as u8;
                f += 1;
            } else {
                return None;
            }
        }
        if f != 8 {
            return None;
        }
    }
    Some(board)
}

fn encode(fen: &str, hist: &[String], history: usize) -> Option<Vec<u8>> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let stm = parts.get(1).copied().unwrap_or("w");
    let cast = parts.get(2).copied().unwrap_or("-");
    let ep = parts.get(3).copied().unwrap_or("-");
    let half = parts
        .get(4)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0)
        .min(255) as u8;
    let fdim = history + 9;
    let mut out = vec![0u8; 64 * fdim];
    let mut boards = Vec::with_capacity(history + 1);
    boards.push(parse_board(fen)?);
    for h in hist.iter().take(history) {
        boards.push(parse_board(h)?);
    }
    while boards.len() < history + 1 {
        boards.push([0u8; 64]);
    }
    for (i, board) in boards.iter().enumerate() {
        for sq in 0..64 {
            out[sq * fdim + i] = board[sq];
        }
    }
    for sq in 0..64 {
        out[sq * fdim + history + 1] = if stm == "w" { 1 } else { 2 };
        out[sq * fdim + history + 2] = ((cast.contains('K') as u8) << 0)
            | ((cast.contains('Q') as u8) << 1)
            | ((cast.contains('k') as u8) << 2)
            | ((cast.contains('q') as u8) << 3);
        out[sq * fdim + history + 4] = half;
        let r = (sq / 8) as u8;
        let f = (sq % 8) as u8;
        out[sq * fdim + history + 5] = r;
        out[sq * fdim + history + 6] = f;
        out[sq * fdim + history + 7] = if ((r + f) & 1) != 0 { 1 } else { 0 };
        out[sq * fdim + history + 8] = sq as u8;
    }
    if let Some(ep_sq) = square_index(ep) {
        out[ep_sq * fdim + history + 3] = 1;
    }
    Some(out)
}

fn valid_row(line: &str, history: usize) -> Option<(Vec<u8>, i64, [f32; 3])> {
    if line.trim().is_empty() {
        return None;
    }
    let row: Value = serde_json::from_str(line).ok()?;
    let fen = row.get("fen")?.as_str()?;
    let policy = row.get("policy")?.as_object()?;
    if policy.len() != 1 {
        return None;
    }
    let mv = policy.keys().next()?;
    let y = move_class(mv)?;
    let hist: Vec<String> = row
        .get("history_fens")
        .and_then(|v| v.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let x = encode(fen, &hist, history)?;
    let mut wdl = [0.25f32, 0.5, 0.25];
    if let Some(xs) = row.get("wdl").and_then(|v| v.as_array()) {
        for i in 0..3.min(xs.len()) {
            if let Some(v) = xs[i].as_f64() {
                wdl[i] = v as f32;
            }
        }
    }
    Some((x, y, wdl))
}

fn write_atomic(path: &str, body: &str) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let tmp = format!("{path}.tmp-{}", std::process::id());
    fs::write(&tmp, body).expect("write temp output");
    fs::rename(&tmp, path).expect("rename temp output");
}

fn input_lines(path: &str) -> impl Iterator<Item = String> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("open input {path}: {e}"));
    BufReader::new(file)
        .lines()
        .map(|l| l.expect("read input line"))
}

fn main() {
    let input = arg("--input", "");
    let out = arg("--out", "artifacts/cache_squareformer_rust/shard_0000");
    let manifest_out = arg("--manifest-out", "");
    let dataset_manifest = arg("--dataset-manifest", "unknown");
    let history: usize = arg("--history-plies", "2").parse().unwrap_or(2);
    let max_rows: usize = arg("--max-rows", "0").parse().unwrap_or(0);
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
        for line in input_lines(path) {
            if max_rows > 0 && rows >= max_rows {
                break;
            }
            if valid_row(&line, history).is_some() {
                rows += 1;
            } else {
                bad += 1;
            }
        }
    }
    let out_path = PathBuf::from(&out);
    let tmp_path = out_path.with_file_name(format!(
        ".{}.tmp-{}",
        out_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("squareformer_cache"),
        std::process::id()
    ));
    if tmp_path.exists() {
        fs::remove_dir_all(&tmp_path).expect("remove stale temp output dir");
    }
    fs::create_dir_all(&tmp_path).expect("create temp output dir");
    let fdim = history + 9;
    let mut tokens = fs::File::create(tmp_path.join("tokens.uint8")).expect("create tokens");
    let mut policy = fs::File::create(tmp_path.join("policy.int64")).expect("create policy");
    let mut wdl_file = fs::File::create(tmp_path.join("wdl.float32")).expect("create wdl");
    let mut written = 0usize;
    'outer: for path in &inputs {
        for line in input_lines(path) {
            if written >= rows {
                break 'outer;
            }
            let Some((x, y, wdl)) = valid_row(&line, history) else {
                continue;
            };
            tokens.write_all(&x).expect("write tokens");
            policy.write_all(&y.to_le_bytes()).expect("write policy");
            for v in wdl {
                wdl_file.write_all(&v.to_le_bytes()).expect("write wdl");
            }
            written += 1;
            if written % 100000 == 0 {
                println!("METRIC square_cache_rows_written={written}");
            }
        }
    }
    let meta = json!({
        "rows": rows,
        "token_features": fdim,
        "history_plies": history,
        "policy_size": 4096 + 4096 * 4,
        "format": "compact_square_tokens_v1",
        "bad_or_skipped_rows": bad,
        "producer": { "language": "rust", "binary": "tiny-leela-rust-squareformer-cache" },
    });
    fs::write(
        tmp_path.join("meta.json"),
        serde_json::to_string(&meta).unwrap(),
    )
    .expect("write meta");
    drop(tokens);
    drop(policy);
    drop(wdl_file);
    if out_path.exists() {
        fs::remove_dir_all(&out_path).expect("remove existing output dir");
    }
    fs::rename(&tmp_path, &out_path).expect("publish output dir");
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "cache_manifest_v1",
            "format": "compact_square_tokens_v1",
            "source": {
                "dataset_manifest": dataset_manifest,
                "inputs": inputs,
                "history_plies": history,
            },
            "producer": {
                "language": "rust",
                "binary": "tiny-leela-rust-squareformer-cache",
                "contract_versions": ["cache_manifest_v1", "squareformer_token_cache_v1"],
            },
            "arrays": [
                { "path": format!("{out}/tokens.uint8"), "dtype": "uint8", "shape": [rows, 64, fdim], "endianness": "native" },
                { "path": format!("{out}/policy.int64"), "dtype": "int64", "shape": [rows], "endianness": "little" },
                { "path": format!("{out}/wdl.float32"), "dtype": "float32", "shape": [rows, 3], "endianness": "little" }
            ],
            "shards": [out],
            "validation": {
                "rows": { "total": rows, "bad_or_skipped": bad },
                "token_features": fdim,
                "policy_size": 4096 + 4096 * 4,
            },
        });
        write_atomic(
            &manifest_out,
            &(serde_json::to_string_pretty(&manifest).expect("serialize manifest") + "\n"),
        );
    }
    println!("METRIC square_cache_rows={rows}");
    println!("METRIC square_cache_token_features={fdim}");
    println!("METRIC square_cache_bad_rows={bad}");
}
