use std::{fs, time::Instant};
use tiny_leela_core::{
    board_to_fen, legal_moves, make_move, move_to_action_id, move_to_uci, parse_fen, search_root,
    Board, PositionEvaluator, SearchOptions, StudentEvaluator, START_FEN,
};

#[cfg(feature = "native-ort")]
use tiny_leela_core::{
    encode_moveformer_legal_inputs, encode_onnx_input_planes, encode_squareformer_compact_input,
    encode_squareformer_float_input, encode_squareformer_legal_ids, is_squareformer_compact_meta,
    OnnxEvaluator, OnnxEvaluatorMeta, SquareFormerEvaluatorMeta,
};

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    for mv in legal_moves(board) {
        nodes += perft(&make_move(board, mv), depth - 1);
    }
    nodes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--onnx-bench-json") {
        #[cfg(feature = "native-ort")]
        {
            let model_path = arg_value(&args, "--onnx").expect("--onnx required");
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let repeats: usize = arg_value(&args, "--repeats")
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);
            let board = parse_fen(&fen).expect("parse fen");
            let evaluator =
                OnnxEvaluator::from_files(&model_path, &meta_path).expect("load onnx evaluator");
            for _ in 0..3 {
                let _ = evaluator.evaluate(&board);
            }
            let t0 = Instant::now();
            for _ in 0..repeats {
                let _ = evaluator.evaluate(&board);
            }
            let seconds = t0.elapsed().as_secs_f64().max(1e-9);
            println!(
                "{}",
                serde_json::json!({
                    "fen": board_to_fen(&board),
                    "repeats": repeats,
                    "seconds": seconds,
                    "msPerEval": seconds * 1000.0 / repeats.max(1) as f64,
                    "evalsPerSecond": repeats as f64 / seconds,
                })
            );
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--onnx-bench-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if args.iter().any(|a| a == "--onnx-eval-json") {
        #[cfg(feature = "native-ort")]
        {
            let model_path = arg_value(&args, "--onnx").expect("--onnx required");
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let history_fens = arg_value(&args, "--history-fens")
                .map(|s| s.split('|').map(|x| x.to_string()).collect::<Vec<_>>())
                .unwrap_or_default();
            let board = parse_fen(&fen).expect("parse fen");
            let evaluator =
                OnnxEvaluator::from_files(&model_path, &meta_path).expect("load onnx evaluator");
            let evaln = evaluator.evaluate_with_history(&board, &history_fens);
            let policy = evaln
                .policy
                .iter()
                .map(|(action_id, probability)| {
                    serde_json::json!({ "actionId": action_id, "probability": probability })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({
                    "fen": board_to_fen(&board),
                    "wdl": evaln.wdl,
                    "policy": policy,
                })
            );
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--onnx-eval-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if args.iter().any(|a| a == "--legal-json") {
        let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
        let board = parse_fen(&fen).expect("parse fen");
        let legal = legal_moves(&board)
            .into_iter()
            .map(|mv| {
                serde_json::json!({
                    "uci": move_to_uci(mv),
                    "actionId": move_to_action_id(mv),
                    "childFen": board_to_fen(&make_move(&board, mv)),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::json!({ "fen": board_to_fen(&board), "legal": legal })
        );
        return;
    }

    if args.iter().any(|a| a == "--encode-onnx-json") {
        #[cfg(feature = "native-ort")]
        {
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let history_fens = arg_value(&args, "--history-fens")
                .map(|s| s.split('|').map(|x| x.to_string()).collect::<Vec<_>>())
                .unwrap_or_default();
            let meta_json = fs::read_to_string(&meta_path).expect("read --meta");
            let meta: OnnxEvaluatorMeta = serde_json::from_str(&meta_json).expect("parse --meta");
            let board = parse_fen(&fen).expect("parse fen");
            let values = encode_onnx_input_planes(&board, &meta, &history_fens);
            println!(
                "{}",
                serde_json::json!({
                    "fen": board_to_fen(&board),
                    "inputPlanes": meta.input_planes,
                    "historyPlies": meta.history_plies,
                    "values": values,
                })
            );
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--encode-onnx-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if args.iter().any(|a| a == "--squareformer-encode-json") {
        #[cfg(feature = "native-ort")]
        {
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let history_fens = arg_value(&args, "--history-fens")
                .map(|s| s.split('|').map(|x| x.to_string()).collect::<Vec<_>>())
                .unwrap_or_default();
            let meta_json = fs::read_to_string(&meta_path).expect("read --meta");
            let meta: SquareFormerEvaluatorMeta =
                serde_json::from_str(&meta_json).expect("parse --meta");
            let board = parse_fen(&fen).expect("parse fen");
            let compact = is_squareformer_compact_meta(&meta);
            let stride = if compact {
                meta.token_features.unwrap_or(meta.history_plies + 9)
            } else {
                meta.input_dim.unwrap_or((meta.history_plies + 1) * 13 + 8)
            };
            if compact {
                let values = encode_squareformer_compact_input(&board, &meta, &history_fens);
                println!(
                    "{}",
                    serde_json::json!({
                        "fen": board_to_fen(&board),
                        "compact": true,
                        "stride": stride,
                        "values": values,
                    })
                );
            } else {
                let values = encode_squareformer_float_input(&board, &meta, &history_fens);
                println!(
                    "{}",
                    serde_json::json!({
                        "fen": board_to_fen(&board),
                        "compact": false,
                        "stride": stride,
                        "values": values,
                    })
                );
            }
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--squareformer-encode-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if args.iter().any(|a| a == "--squareformer-legal-json") {
        #[cfg(feature = "native-ort")]
        {
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let meta_json = fs::read_to_string(&meta_path).expect("read --meta");
            let meta: SquareFormerEvaluatorMeta =
                serde_json::from_str(&meta_json).expect("parse --meta");
            let width = meta
                .onnx_fixed_legal_moves
                .or(meta.max_legal_moves)
                .unwrap_or(128)
                .max(1);
            let board = parse_fen(&fen).expect("parse fen");
            let (moves, action_ids) = encode_squareformer_legal_ids(&board, width);
            let legal_uci = moves.into_iter().map(move_to_uci).collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({
                    "fen": board_to_fen(&board),
                    "width": width,
                    "legalUci": legal_uci,
                    "actionIds": action_ids,
                })
            );
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--squareformer-legal-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if args.iter().any(|a| a == "--moveformer-legal-json") {
        #[cfg(feature = "native-ort")]
        {
            let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
            let meta_path = arg_value(&args, "--meta").expect("--meta required");
            let meta_json = fs::read_to_string(&meta_path).expect("read --meta");
            let meta: OnnxEvaluatorMeta = serde_json::from_str(&meta_json).expect("parse --meta");
            let width = meta
                .onnx_fixed_legal_moves
                .or(meta.max_legal_moves)
                .unwrap_or(128)
                .max(1);
            let feature_count = meta.num_move_features.unwrap_or(20).max(1);
            let board = parse_fen(&fen).expect("parse fen");
            let (moves, action_ids, features, mask) =
                encode_moveformer_legal_inputs(&board, width, feature_count);
            let legal_uci = moves.into_iter().map(move_to_uci).collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({
                    "fen": board_to_fen(&board),
                    "width": width,
                    "featureCount": feature_count,
                    "legalUci": legal_uci,
                    "actionIds": action_ids,
                    "features": features,
                    "mask": mask,
                })
            );
            return;
        }
        #[cfg(not(feature = "native-ort"))]
        {
            eprintln!("--moveformer-legal-json requires cargo feature native-ort");
            std::process::exit(2);
        }
    }

    if let Some(depth_s) = arg_value(&args, "--perft") {
        let depth: u32 = depth_s.parse().expect("parse --perft depth");
        let fen = arg_value(&args, "--fen").unwrap_or_else(|| START_FEN.to_string());
        let board = parse_fen(&fen).expect("parse fen");
        let t0 = Instant::now();
        let nodes = perft(&board, depth);
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        println!("fen={}", board_to_fen(&board));
        println!("depth={depth}");
        println!("nodes={nodes}");
        println!("METRIC rust_perft_nodes={nodes}");
        println!("METRIC rust_perft_seconds={elapsed:.6}");
        return;
    }

    let mut positional = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += if args[i].contains('=') { 1 } else { 2 };
        } else {
            positional.push(args[i].clone());
            i += 1;
        }
    }
    let artifact_path = positional
        .first()
        .cloned()
        .unwrap_or_else(|| "artifacts/student_distill_benchmark.json".to_string());
    let fen = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| START_FEN.to_string());
    let visits: u32 = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let temperature: f32 = positional
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cpuct: f32 = arg_value(&args, "--cpuct")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5);
    let fpu: f32 = arg_value(&args, "--fpu")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let json = fs::read_to_string(&artifact_path).expect("read artifact json");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse student artifact");
    let board = parse_fen(&fen).expect("parse fen");
    let t0 = Instant::now();
    let evaln = evaluator.evaluate(&board);
    let result = search_root(
        &board,
        &evaluator,
        SearchOptions {
            visits,
            cpuct,
            fpu,
            temperature,
            ..SearchOptions::default()
        },
    );
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    println!("fen={}", board_to_fen(&board));
    println!(
        "best_move={}",
        result
            .mv
            .map(move_to_uci)
            .unwrap_or_else(|| "none".to_string())
    );
    let policy_json = result.policy.iter()
        .map(|entry| format!("{{\"move\":\"{}\",\"probability\":{:.9},\"visits\":{},\"prior\":{:.9},\"q\":{:.9}}}", move_to_uci(entry.mv), entry.probability, entry.visits, entry.prior, entry.q))
        .collect::<Vec<_>>()
        .join(",");
    println!("root_policy_json=[{}]", policy_json);
    println!(
        "wdl={:.6},{:.6},{:.6}",
        evaln.wdl[0], evaln.wdl[1], evaln.wdl[2]
    );
    println!("policy_legal_count={}", evaln.policy.len());
    println!("METRIC rust_student_search_visits={}", result.visits);
    println!("METRIC rust_student_eval_search_seconds={:.6}", elapsed);
    println!(
        "METRIC rust_student_visits_per_second={:.6}",
        result.visits as f64 / elapsed
    );
}
