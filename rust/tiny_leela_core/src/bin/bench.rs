use std::time::Instant;
use tiny_leela_core::{legal_moves, parse_fen, START_FEN};

fn main() {
    let iterations: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let board = parse_fen(START_FEN).expect("valid start FEN");
    let t0 = Instant::now();
    let mut nodes = 0usize;
    for _ in 0..iterations {
        nodes += legal_moves(&board).len();
    }
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    println!("METRIC rust_legal_movegen_iterations={iterations}");
    println!("METRIC rust_legal_moves_total={nodes}");
    println!("METRIC rust_legal_movegen_positions_per_second={:.6}", iterations as f64 / elapsed);
}
