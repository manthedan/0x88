use std::{collections::HashMap, fs};
use tiny_leela_core::{
    legal_moves, make_move, move_to_action_id, move_to_uci, parse_fen, PositionEvaluator,
    StudentEvaluator, START_FEN,
};

struct Edge {
    mv: tiny_leela_core::Move,
    prior: f32,
    visits: u32,
    value_sum: f32,
}
fn q(e: &Edge) -> f32 {
    if e.visits > 0 {
        -e.value_sum / e.visits as f32
    } else {
        0.0
    }
}
fn value(wdl: [f32; 3]) -> f32 {
    wdl[0] - wdl[2]
}

fn main() {
    let artifact_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts/student_distill_benchmark.json".to_string());
    let fen = std::env::args()
        .nth(2)
        .unwrap_or_else(|| START_FEN.to_string());
    let sims: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let json = fs::read_to_string(&artifact_path).expect("read artifact");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse artifact");
    let board = parse_fen(&fen).expect("parse fen");
    let moves = legal_moves(&board);
    let evaln = evaluator.evaluate(&board);
    let map: HashMap<u32, f32> = evaln.policy.into_iter().collect();
    let raw: Vec<f32> = moves
        .iter()
        .map(|&m| {
            map.get(&move_to_action_id(m))
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    let total: f32 = raw.iter().sum();
    let fallback = if moves.is_empty() {
        0.0
    } else {
        1.0 / moves.len() as f32
    };
    let mut edges: Vec<Edge> = moves
        .into_iter()
        .enumerate()
        .map(|(i, mv)| Edge {
            mv,
            prior: if total > 0.0 {
                raw[i] / total
            } else {
                fallback
            },
            visits: 0,
            value_sum: 0.0,
        })
        .collect();
    println!("root_value={:.9}", value(evaln.wdl));
    for (i, e) in edges.iter().enumerate().take(8) {
        println!("PRIOR {i} {} {:.9}", move_to_uci(e.mv), e.prior);
    }
    for sim in 0..sims {
        let parent_visits: u32 = edges.iter().map(|e| e.visits).sum();
        let sqrt_parent = ((parent_visits + 1) as f32).sqrt();
        let mut best_i = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in edges.iter().enumerate() {
            let score = q(e) + 1.5 * e.prior * sqrt_parent / (1.0 + e.visits as f32);
            if score > best_score {
                best_i = i;
                best_score = score;
            }
        }
        let child = make_move(&board, edges[best_i].mv);
        let child_value = value(evaluator.evaluate(&child).wdl);
        let before = edges[best_i].visits;
        edges[best_i].visits += 1;
        edges[best_i].value_sum += child_value;
        println!("TRACE sim={} move={} prior={:.9} q_before={:.9} score={:.9} visits_before={} child_value={:.9} q_after={:.9}", sim, move_to_uci(edges[best_i].mv), edges[best_i].prior, if before>0 { -(edges[best_i].value_sum-child_value)/before as f32 } else { 0.0 }, best_score, before, child_value, q(&edges[best_i]));
    }
}
