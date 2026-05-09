use std::{fs, time::Instant};
use tiny_leela_core::{board_to_fen, legal_moves, make_move, move_to_uci, parse_fen, search_root, Board, StudentEvaluator, SearchOptions, PositionEvaluator, START_FEN};

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let mut nodes = 0u64;
    for mv in legal_moves(board) { nodes += perft(&make_move(board, mv), depth - 1); }
    nodes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
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

    let artifact_path = std::env::args().nth(1).unwrap_or_else(|| "artifacts/student_distill_benchmark.json".to_string());
    let fen = std::env::args().nth(2).unwrap_or_else(|| START_FEN.to_string());
    let visits: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let temperature: f32 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let json = fs::read_to_string(&artifact_path).expect("read artifact json");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse student artifact");
    let board = parse_fen(&fen).expect("parse fen");
    let t0 = Instant::now();
    let evaln = evaluator.evaluate(&board);
    let result = search_root(&board, &evaluator, SearchOptions { visits, cpuct: 1.5, temperature });
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    println!("fen={}", board_to_fen(&board));
    println!("best_move={}", result.mv.map(move_to_uci).unwrap_or_else(|| "none".to_string()));
    let policy_json = result.policy.iter()
        .map(|entry| format!("{{\"move\":\"{}\",\"probability\":{:.9},\"visits\":{},\"prior\":{:.9},\"q\":{:.9}}}", move_to_uci(entry.mv), entry.probability, entry.visits, entry.prior, entry.q))
        .collect::<Vec<_>>()
        .join(",");
    println!("root_policy_json=[{}]", policy_json);
    println!("wdl={:.6},{:.6},{:.6}", evaln.wdl[0], evaln.wdl[1], evaln.wdl[2]);
    println!("policy_legal_count={}", evaln.policy.len());
    println!("METRIC rust_student_search_visits={}", result.visits);
    println!("METRIC rust_student_eval_search_seconds={:.6}", elapsed);
    println!("METRIC rust_student_visits_per_second={:.6}", result.visits as f64 / elapsed);
}
