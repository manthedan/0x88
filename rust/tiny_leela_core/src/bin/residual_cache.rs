use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, HashMap},
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};
use tiny_leela_core::for_each_jsonl_line;

const FILES: &str = "abcdefgh";
const PIECES: &str = "PNBRQKpnbrqk";

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

fn on(f: i32, r: i32) -> bool {
    (0..8).contains(&f) && (0..8).contains(&r)
}

fn sq(f: i32, r: i32) -> String {
    format!("{}{}", FILES.as_bytes()[f as usize] as char, r + 1)
}

fn fixed_policy_moves() -> Vec<String> {
    let mut out = BTreeSet::new();
    let dirs = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let knights = [
        (1, 2),
        (2, 1),
        (-1, 2),
        (-2, 1),
        (1, -2),
        (2, -1),
        (-1, -2),
        (-2, -1),
    ];
    for r in 0..8 {
        for f in 0..8 {
            let from = sq(f, r);
            for (df, dr) in dirs {
                for n in 1..8 {
                    let tf = f + df * n;
                    let tr = r + dr * n;
                    if !on(tf, tr) {
                        break;
                    }
                    out.insert(format!("{}{}", from, sq(tf, tr)));
                }
            }
            for (df, dr) in knights {
                if on(f + df, r + dr) {
                    out.insert(format!("{}{}", from, sq(f + df, r + dr)));
                }
            }
        }
    }
    for r in [1, 6] {
        let tr = if r == 6 { 7 } else { 0 };
        for f in 0..8 {
            for df in [-1, 0, 1] {
                if on(f + df, tr) {
                    for promo in ["q", "r", "b", "n"] {
                        out.insert(format!("{}{}{}", sq(f, r), sq(f + df, tr), promo));
                    }
                }
            }
        }
    }
    out.into_iter().collect()
}

fn input_plane_count(history_plies: usize, state_planes: bool) -> usize {
    12 * (history_plies + 1) + if state_planes { 10 } else { 2 }
}

fn add_piece_planes(x: &mut [i8], c: usize, fen: &str, offset: usize) -> Option<()> {
    let board = fen.split_whitespace().next()?;
    let mut r = 0usize;
    let mut f = 0usize;
    for ch in board.chars() {
        if ch == '/' {
            r += 1;
            f = 0;
        } else if ch.is_ascii_digit() {
            f += ch.to_digit(10)? as usize;
        } else if let Some(pi) = PIECES.find(ch) {
            if offset + pi >= c || r >= 8 || f >= 8 {
                return None;
            }
            x[(offset + pi) * 64 + r * 8 + f] = 1;
            f += 1;
        } else {
            return None;
        }
    }
    Some(())
}

fn current_board_18(fen: &str) -> Option<Vec<i8>> {
    let mut x = vec![0i8; 18 * 64];
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let side = parts.get(1).copied().unwrap_or("w");
    let castling = parts.get(2).copied().unwrap_or("-");
    let ep = parts.get(3).copied().unwrap_or("-");
    add_piece_planes(&mut x, 18, fen, 0)?;
    x[12 * 64..13 * 64].fill(if side == "w" { 1 } else { 0 });
    for (i, flag) in ["K", "Q", "k", "q"].iter().enumerate() {
        if castling.contains(flag) {
            x[(13 + i) * 64..(14 + i) * 64].fill(1);
        }
    }
    if ep.len() >= 2 {
        let b = ep.as_bytes();
        let ef = (b[0] as i32) - ('a' as i32);
        let er = 8 - ((b[1] - b'0') as i32);
        if on(ef, er) {
            x[17 * 64 + er as usize * 8 + ef as usize] = 1;
        }
    }
    Some(x)
}

fn history_planes(
    fen: &str,
    hist: &[String],
    history_plies: usize,
    state_planes: bool,
) -> Option<Vec<i8>> {
    let c = input_plane_count(history_plies, state_planes);
    let mut x = vec![0i8; c * 64];
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let side = parts.get(1).copied().unwrap_or("w");
    let castling = parts.get(2).copied().unwrap_or("-");
    let ep = parts.get(3).copied().unwrap_or("-");
    add_piece_planes(&mut x, c, fen, 0)?;
    for (h, hf) in hist.iter().take(history_plies).enumerate() {
        add_piece_planes(&mut x, c, hf, 12 * (h + 1))?;
    }
    let s0 = 12 * (history_plies + 1);
    x[s0 * 64..(s0 + 1) * 64].fill(if side == "w" { 1 } else { -1 });
    if state_planes {
        for (i, flag) in ["K", "Q", "k", "q"].iter().enumerate() {
            if castling.contains(flag) {
                x[(s0 + 1 + i) * 64..(s0 + 2 + i) * 64].fill(1);
            }
        }
        if ep.len() >= 2 {
            let b = ep.as_bytes();
            let ef = (b[0] as i32) - ('a' as i32);
            let er = 8 - ((b[1] - b'0') as i32);
            if on(ef, er) {
                x[(s0 + 5) * 64 + er as usize * 8 + ef as usize] = 1;
            }
        }
        x[(s0 + 6) * 64..(s0 + 7) * 64].fill(1);
        x[(s0 + 7) * 64..(s0 + 8) * 64].fill(if side == "w" { 1 } else { 0 });
    } else {
        x[(s0 + 1) * 64..(s0 + 2) * 64].fill(1);
    }
    Some(x)
}

#[derive(Clone)]
struct Row {
    x: Vec<i8>,
    y: i64,
    wdl: [f32; 3],
    weight: f32,
    stockfish_q: f32,
    stockfish_winrate_loss: f32,
    blunder: i64,
}

fn parse_row(
    line: &str,
    mid: &HashMap<String, usize>,
    history_plies: usize,
    state_planes: bool,
    current18: bool,
) -> Option<Row> {
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
    let y = *mid.get(mv)? as i64;
    let hist: Vec<String> = row
        .get("history_fens")
        .and_then(|v| v.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let x = if current18 {
        current_board_18(fen)?
    } else {
        history_planes(fen, &hist, history_plies, state_planes)?
    };
    let mut wdl = [0.25f32, 0.5, 0.25];
    if let Some(xs) = row.get("wdl").and_then(|v| v.as_array()) {
        for i in 0..3.min(xs.len()) {
            if let Some(v) = xs[i].as_f64() {
                wdl[i] = v as f32;
            }
        }
    }
    Some(Row {
        x,
        y,
        wdl,
        weight: row.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        stockfish_q: row
            .get("stockfish_q")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(f32::NAN),
        stockfish_winrate_loss: row
            .get("stockfish_winrate_loss")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(f32::NAN),
        blunder: row
            .get("stockfish_blunder_bucket")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1),
    })
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
    let input = arg("--input", "");
    let out = arg("--out", "artifacts/cache_residual_rust/shard_0000");
    let manifest_out = arg("--manifest-out", "");
    let dataset_manifest = arg("--dataset-manifest", "unknown");
    let history_plies: usize = arg("--history-plies", "2").parse().unwrap_or(2);
    let max_rows: usize = arg("--max-rows", "0").parse().unwrap_or(0);
    let state_planes = flag("--state-planes");
    let current18 = flag("--current-board-18");
    if input.is_empty() {
        panic!("--input is required");
    }
    let moves = fixed_policy_moves();
    let mid: HashMap<String, usize> = moves
        .iter()
        .enumerate()
        .map(|(i, m)| (m.clone(), i))
        .collect();
    let inputs: Vec<String> = input
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let mut rows = 0usize;
    let mut skipped_unknown_moves = 0usize;
    for path in &inputs {
        for_each_jsonl_line(path, |line| {
            if max_rows > 0 && rows >= max_rows {
                return Ok(false);
            }
            let row: Value = serde_json::from_str(line).unwrap_or(Value::Null);
            let mv = row.get("policy").and_then(|p| p.as_object()).and_then(|p| {
                if p.len() == 1 {
                    p.keys().next().cloned()
                } else {
                    None
                }
            });
            if let Some(mv) = mv {
                if !mid.contains_key(&mv) {
                    skipped_unknown_moves += 1;
                    return Ok(true);
                }
            }
            if parse_row(line, &mid, history_plies, state_planes, current18).is_some() {
                rows += 1;
            }
            Ok(true)
        })
        .expect("stream residual input");
    }
    let out_path = PathBuf::from(&out);
    let tmp_path = out_path.with_file_name(format!(
        ".{}.tmp-{}",
        out_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("residual_cache"),
        std::process::id()
    ));
    if tmp_path.exists() {
        fs::remove_dir_all(&tmp_path).expect("remove stale temp output dir");
    }
    fs::create_dir_all(&tmp_path).expect("create temp output dir");
    let c = if current18 {
        18
    } else {
        input_plane_count(history_plies, state_planes)
    };
    let mut x_file = fs::File::create(tmp_path.join("x.int8")).expect("create x");
    let mut y_file = fs::File::create(tmp_path.join("policy.int64")).expect("create policy");
    let mut wdl_file = fs::File::create(tmp_path.join("wdl.float32")).expect("create wdl");
    let mut weight_file = fs::File::create(tmp_path.join("weight.float32")).expect("create weight");
    let mut sfq_file =
        fs::File::create(tmp_path.join("stockfish_q.float32")).expect("create stockfish_q");
    let mut wr_file = fs::File::create(tmp_path.join("stockfish_winrate_loss.float32"))
        .expect("create winrate loss");
    let mut blunder_file =
        fs::File::create(tmp_path.join("stockfish_blunder_bucket.int64")).expect("create blunder");
    let mut written = 0usize;
    for path in &inputs {
        for_each_jsonl_line(path, |line| {
            if written >= rows {
                return Ok(false);
            }
            let Some(row) = parse_row(line, &mid, history_plies, state_planes, current18) else {
                return Ok(true);
            };
            x_file
                .write_all(unsafe {
                    std::slice::from_raw_parts(row.x.as_ptr() as *const u8, row.x.len())
                })
                .expect("write x");
            y_file.write_all(&row.y.to_le_bytes()).expect("write y");
            for v in row.wdl {
                wdl_file.write_all(&v.to_le_bytes()).expect("write wdl");
            }
            weight_file
                .write_all(&row.weight.to_le_bytes())
                .expect("write weight");
            sfq_file
                .write_all(&row.stockfish_q.to_le_bytes())
                .expect("write sfq");
            wr_file
                .write_all(&row.stockfish_winrate_loss.to_le_bytes())
                .expect("write wr");
            blunder_file
                .write_all(&row.blunder.to_le_bytes())
                .expect("write blunder");
            written += 1;
            if written % 100000 == 0 {
                println!("METRIC residual_cache_rows_written={written}");
            }
            Ok(true)
        })
        .expect("stream residual input");
    }
    let meta = json!({
        "rows": rows,
        "input_planes": c,
        "history_plies": if current18 { 0 } else { history_plies },
        "state_planes": if current18 { false } else { state_planes },
        "input_mode": if current18 { "current_board_18" } else { "history" },
        "policy_size": moves.len(),
        "moves": moves,
        "skipped_unknown_moves": skipped_unknown_moves,
        "has_stockfish_q": true,
        "has_stockfish_winrate_loss": true,
        "has_side_info": false,
        "producer": { "language": "rust", "binary": "tiny-leela-rust-residual-cache" },
    });
    fs::write(
        tmp_path.join("meta.json"),
        serde_json::to_string(&meta).unwrap(),
    )
    .expect("write meta");
    fs::write(tmp_path.join("_SUCCESS"), b"ok\n").expect("write success marker");
    drop((
        x_file,
        y_file,
        wdl_file,
        weight_file,
        sfq_file,
        wr_file,
        blunder_file,
    ));
    if out_path.exists() {
        fs::remove_dir_all(&out_path).expect("remove existing output dir");
    }
    fs::rename(&tmp_path, &out_path).expect("publish output dir");
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "cache_manifest_v1",
            "format": if current18 { "residual_current_board_18_v1" } else { "residual_history_planes_v1" },
            "source": { "dataset_manifest": dataset_manifest, "inputs": inputs, "history_plies": if current18 { 0 } else { history_plies } },
            "producer": { "language": "rust", "binary": "tiny-leela-rust-residual-cache", "contract_versions": ["cache_manifest_v1"] },
            "arrays": [
                { "path": format!("{out}/x.int8"), "dtype": "int8", "shape": [rows, c, 8, 8], "endianness": "native" },
                { "path": format!("{out}/policy.int64"), "dtype": "int64", "shape": [rows], "endianness": "little" },
                { "path": format!("{out}/wdl.float32"), "dtype": "float32", "shape": [rows, 3], "endianness": "little" }
            ],
            "shards": [out],
            "validation": { "rows": { "total": rows }, "policy_size": meta["policy_size"], "input_planes": c, "hashes": {} },
        });
        write_atomic(
            &manifest_out,
            &(serde_json::to_string_pretty(&manifest).unwrap() + "\n"),
        );
    }
    println!("METRIC residual_cache_rows={rows}");
    println!("METRIC residual_cache_input_planes={c}");
    println!("METRIC residual_cache_skipped_unknown_moves={skipped_unknown_moves}");
}
