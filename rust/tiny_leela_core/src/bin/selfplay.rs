use serde_json::json;
use std::{env, fs, path::Path, process::Command, time::Instant};
#[cfg(feature = "native-ort")]
use tiny_leela_core::OnnxEvaluator;
use tiny_leela_core::{
    board_to_fen, in_check, legal_moves, make_move, move_to_uci, parse_fen, search_root,
    sha256_file_hex, Color, PositionEvaluator, SearchOptions, SearchPolicyMode, StudentEvaluator,
    START_FEN,
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

fn flag(name: &str) -> bool {
    env::args().any(|a| a == name)
}

fn rng_next(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    (*s as f64 / 4294967296.0) as f32
}

fn terminal_white_score(board: &tiny_leela_core::Board) -> Option<f32> {
    if !legal_moves(board).is_empty() {
        return None;
    }
    if !in_check(board, board.turn) {
        return Some(0.5);
    }
    Some(if board.turn == Color::White { 0.0 } else { 1.0 })
}

fn write_selfplay_output(path: &str, body: &str) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let tmp = format!("{path}.tmp-{}", std::process::id());
    if path.ends_with(".zst") {
        let plain = format!("{tmp}.jsonl");
        fs::write(&plain, body).expect("write temp jsonl");
        let status = Command::new("zstd")
            .args(["-q", "-f", &plain, "-o", &tmp])
            .status()
            .expect("run zstd");
        if !status.success() {
            panic!("zstd compression failed for {path}: {status}");
        }
        let _ = fs::remove_file(&plain);
    } else {
        fs::write(&tmp, body).expect("write temp jsonl");
    }
    fs::rename(&tmp, path).expect("publish selfplay output");
}

fn result_for_turn(white_score: f32, turn: Color) -> [u8; 3] {
    if (white_score - 0.5).abs() < 1e-6 {
        return [0, 1, 0];
    }
    let side_won = (white_score == 1.0 && turn == Color::White)
        || (white_score == 0.0 && turn == Color::Black);
    if side_won {
        [1, 0, 0]
    } else {
        [0, 0, 1]
    }
}

fn adjudicated_white_score(
    board: &tiny_leela_core::Board,
    evaluator: &dyn PositionEvaluator,
    adjudicate: &str,
    threshold: f32,
) -> (f32, bool) {
    if let Some(score) = terminal_white_score(board) {
        return (score, false);
    }
    if adjudicate != "value" {
        return (0.5, false);
    }
    let evaln = evaluator.evaluate(board);
    let side_value = evaln.wdl[0] - evaln.wdl[2];
    if side_value.abs() <= threshold {
        return (0.5, true);
    }
    let side_to_move_wins = side_value > 0.0;
    let white_wins = (board.turn == Color::White && side_to_move_wins)
        || (board.turn == Color::Black && !side_to_move_wins);
    (if white_wins { 1.0 } else { 0.0 }, true)
}

fn main() {
    let model_path = arg("--model", "artifacts/student_distill_benchmark.json");
    let meta_path = arg("--meta", "");
    let out_path = arg("--out", "data/selfplay/bootstrap.jsonl");
    let manifest_out = arg("--manifest-out", "");
    let lane = arg("--lane", "sup_sp");
    let shard_id = arg("--shard-id", "");
    let model_id = arg("--model-id", "");
    let rules_only = flag("--rules-only");
    if lane == "gumbel_zero" {
        eprintln!("tiny-leela-rust-selfplay refuses --lane gumbel_zero because this binary uses model-guided PUCT; use the dedicated rules-only Gumbel-Zero generator instead");
        std::process::exit(2);
    }
    if !matches!(lane.as_str(), "sup_sp" | "eval_demo" | "other") {
        eprintln!("unsupported --lane {lane}; choose sup_sp, eval_demo, other (gumbel_zero is intentionally blocked here)");
        std::process::exit(2);
    }
    if rules_only {
        eprintln!("--rules-only is only valid for dedicated Gumbel-Zero tooling, not model-guided Rust selfplay");
        std::process::exit(2);
    }
    let games: usize = arg("--games", "2").parse().unwrap_or(2);
    let visits: u32 = arg("--visits", "4").parse().unwrap_or(4);
    let max_plies: usize = arg("--max-plies", "40").parse().unwrap_or(40);
    let temperature: f32 = arg("--temperature", "1").parse().unwrap_or(1.0);
    let policy_mode = SearchPolicyMode::from_name(&arg("--policy-mode", "classic"));
    let av_weight: f32 = arg("--av-weight", "0.25").parse().unwrap_or(0.25);
    let rank_weight: f32 = arg("--rank-weight", "0.0").parse().unwrap_or(0.0);
    let regret_weight: f32 = arg("--regret-weight", "0.0").parse().unwrap_or(0.0);
    let risk_weight: f32 = arg("--risk-weight", "0.0").parse().unwrap_or(0.0);
    let uncertainty_weight: f32 = arg("--uncertainty-weight", "0.0").parse().unwrap_or(0.0);
    let progress_every: usize = arg("--progress-every", "1").parse().unwrap_or(1).max(1);
    let adjudicate = arg("--adjudicate", "terminal");
    let adjudicate_threshold: f32 = arg("--adjudicate-threshold", "0.02")
        .parse()
        .unwrap_or(0.02);
    let opening_fens_path = arg("--opening-fens", "");
    let opening_fens: Vec<String> = if opening_fens_path.is_empty() {
        Vec::new()
    } else {
        fs::read_to_string(&opening_fens_path)
            .expect("read opening fens")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                line.split_once('\t')
                    .map(|(_, fen)| fen)
                    .unwrap_or(line)
                    .to_string()
            })
            .collect()
    };
    let mut rng: u32 = arg("--seed", "1").parse().unwrap_or(1);

    let evaluator: Box<dyn PositionEvaluator> = if !meta_path.is_empty()
        || model_path.to_ascii_lowercase().ends_with(".onnx")
    {
        #[cfg(feature = "native-ort")]
        {
            if meta_path.is_empty() {
                eprintln!("--meta is required when --model points at an ONNX file");
                std::process::exit(2);
            }
            Box::new(OnnxEvaluator::from_files(&model_path, &meta_path).expect("load ONNX model"))
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("ONNX self-play requires cargo feature native-ort");
            std::process::exit(2);
        }
    } else {
        let json_text = fs::read_to_string(&model_path).expect("read model");
        Box::new(StudentEvaluator::from_json(&json_text).expect("parse model"))
    };
    let started = Instant::now();
    eprintln!(
        "[rust-selfplay visits={visits}] start games={games} max_plies={max_plies} out={out_path}"
    );

    let mut rows = Vec::new();
    let mut completed_games = 0usize;
    let mut decisive_games = 0usize;
    let mut total_plies = 0usize;
    let mut policy_mass = 0f32;
    let mut adjudicated_games = 0usize;

    for game in 0..games {
        eprintln!(
            "[rust-selfplay visits={visits}] game {}/{} start positions={}",
            game + 1,
            games,
            rows.len()
        );
        let mut board = if opening_fens.is_empty() {
            parse_fen(START_FEN).expect("start fen")
        } else {
            let idx = ((rng_next(&mut rng) * opening_fens.len() as f32) as usize)
                .min(opening_fens.len() - 1);
            parse_fen(&opening_fens[idx]).expect("opening fen")
        };
        let mut pending: Vec<(serde_json::Value, Color)> = Vec::new();
        let mut white_score: Option<f32> = None;
        for ply in 0..max_plies {
            white_score = terminal_white_score(&board);
            if white_score.is_some() {
                break;
            }
            let legal_uci: Vec<String> = legal_moves(&board).into_iter().map(move_to_uci).collect();
            let result = search_root(
                &board,
                &*evaluator,
                SearchOptions {
                    visits,
                    cpuct: 1.5,
                    fpu: 0.0,
                    temperature,
                    policy_mode,
                    av_weight,
                    rank_weight,
                    regret_weight,
                    risk_weight,
                    uncertainty_weight,
                },
            );
            if result.mv.is_none() || result.policy.is_empty() {
                white_score = Some(0.5);
                break;
            }
            let mut policy_obj = serde_json::Map::new();
            let mut mass = 0f32;
            for entry in result.policy.iter().filter(|e| e.probability > 0.0) {
                let p = (entry.probability * 1e8).round() / 1e8;
                mass += p;
                policy_obj.insert(move_to_uci(entry.mv), json!(p));
            }
            policy_mass += mass;
            let fen = board_to_fen(&board);
            let turn = board.turn;
            let mut r = rng_next(&mut rng);
            let mut chosen = result.policy.last().map(|e| e.mv).unwrap();
            for entry in &result.policy {
                r -= entry.probability;
                if r <= 0.0 {
                    chosen = entry.mv;
                    break;
                }
            }
            let selected_uci = move_to_uci(chosen);
            pending.push((
                json!({
                    "schema": "selfplay_chunk_v1",
                    "lane": lane,
                    "game_id": format!("g{game:06}"),
                    "shard_id": if shard_id.is_empty() { serde_json::Value::Null } else { json!(shard_id) },
                    "ply": ply,
                    "fen": fen,
                    "turn": if turn == Color::White { "w" } else { "b" },
                    "legal_uci": legal_uci,
                    "selected_uci": selected_uci,
                    "visits": result.visits,
                    "policy": policy_obj,
                    "q": ((result.value * 1e8).round() / 1e8),
                    "root_value": ((result.value * 1e8).round() / 1e8),
                    "search": {
                        "visits": result.visits,
                        "policy_mode": format!("{policy_mode:?}"),
                        "cpuct": 1.5,
                        "temperature": temperature,
                    },
                    "provenance": {
                        "generator": "tiny-leela-rust-selfplay",
                        "seed": rng,
                        "model_id": if model_id.is_empty() { serde_json::Value::Null } else { json!(model_id) },
                        "model_sha256": sha256_file_hex(&model_path).ok(),
                        "rules_only": false,
                    },
                }),
                turn,
            ));
            board = make_move(&board, chosen);
            total_plies += 1;
            if (ply + 1) % progress_every == 0 {
                eprintln!("[rust-selfplay visits={visits}] game {}/{} ply={}/{} move={} rows_pending={} elapsed_s={:.1}", game + 1, games, ply + 1, max_plies, move_to_uci(chosen), pending.len(), started.elapsed().as_secs_f64());
            }
        }
        let (score, adjudicated) = if let Some(score) = white_score {
            (score, false)
        } else {
            adjudicated_white_score(&board, &*evaluator, &adjudicate, adjudicate_threshold)
        };
        if adjudicated {
            adjudicated_games += 1;
        }
        if (score - 0.5).abs() > 1e-6 {
            decisive_games += 1;
        }
        for (mut row, turn) in pending {
            let obj = row.as_object_mut().unwrap();
            let wdl = result_for_turn(score, turn);
            obj.insert("result".into(), json!(wdl));
            obj.insert("wdl".into(), json!(wdl));
            obj.insert("white_score".into(), json!(score));
            rows.push(row);
        }
        completed_games += 1;
        eprintln!("[rust-selfplay visits={visits}] game {}/{} done white_score={} rows_total={} adjudicated={} elapsed_s={:.1}", game + 1, games, score, rows.len(), adjudicated_games, started.elapsed().as_secs_f64());
    }

    let body = rows
        .iter()
        .map(|row| serde_json::to_string(row).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let output_body = if body.is_empty() {
        String::new()
    } else {
        format!("{body}\n")
    };
    write_selfplay_output(&out_path, &output_body);
    let bytes = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

    println!("METRIC selfplay_backend_rust=1");
    println!("METRIC selfplay_opening_fens={}", opening_fens.len());
    println!("METRIC selfplay_search_policy_mode={policy_mode:?}");
    println!("METRIC selfplay_games={completed_games}");
    println!("METRIC selfplay_positions={}", rows.len());
    println!(
        "METRIC selfplay_avg_plies={:.6}",
        total_plies as f64 / completed_games.max(1) as f64
    );
    println!(
        "METRIC selfplay_decisive_rate={:.6}",
        decisive_games as f64 / completed_games.max(1) as f64
    );
    println!(
        "METRIC selfplay_adjudicated_rate={:.6}",
        adjudicated_games as f64 / completed_games.max(1) as f64
    );
    println!(
        "METRIC selfplay_policy_mass={:.6}",
        policy_mass as f64 / rows.len().max(1) as f64
    );
    println!("METRIC selfplay_output_bytes={bytes}");
    if !manifest_out.is_empty() {
        let manifest = json!({
            "schema": "selfplay_chunk_manifest_v1",
            "chunk_schema": "selfplay_chunk_v1",
            "path": out_path,
            "rows": rows.len(),
            "games": completed_games,
            "lane": lane,
            "shard_id": shard_id,
            "sha256": sha256_file_hex(&out_path).ok(),
            "atomic_write": true,
            "compressed": out_path.ends_with(".zst"),
            "producer": { "language": "rust", "binary": "tiny-leela-rust-selfplay" },
        });
        if let Some(parent) = Path::new(&manifest_out).parent() {
            fs::create_dir_all(parent).expect("create manifest dir");
        }
        fs::write(
            &manifest_out,
            serde_json::to_string_pretty(&manifest).unwrap() + "\n",
        )
        .expect("write manifest");
    }
}
