use std::{fs, time::Instant};
use tiny_leela_core::{board_to_fen, move_to_uci, parse_fen, search_root, StudentEvaluator, SearchOptions, PositionEvaluator, START_FEN};

fn main() {
    let artifact_path = std::env::args().nth(1).unwrap_or_else(|| "artifacts/student_distill_benchmark.json".to_string());
    let fen = std::env::args().nth(2).unwrap_or_else(|| START_FEN.to_string());
    let visits: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let json = fs::read_to_string(&artifact_path).expect("read artifact json");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse student artifact");
    let board = parse_fen(&fen).expect("parse fen");
    let t0 = Instant::now();
    let evaln = evaluator.evaluate(&board);
    let result = search_root(&board, &evaluator, SearchOptions { visits, cpuct: 1.5, temperature: 0.0 });
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    println!("fen={}", board_to_fen(&board));
    println!("best_move={}", result.mv.map(move_to_uci).unwrap_or_else(|| "none".to_string()));
    println!("wdl={:.6},{:.6},{:.6}", evaln.wdl[0], evaln.wdl[1], evaln.wdl[2]);
    println!("policy_legal_count={}", evaln.policy.len());
    println!("METRIC rust_student_search_visits={}", result.visits);
    println!("METRIC rust_student_eval_search_seconds={:.6}", elapsed);
    println!("METRIC rust_student_visits_per_second={:.6}", result.visits as f64 / elapsed);
}
