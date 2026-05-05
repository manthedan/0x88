use std::{env, fs, time::Instant};
use tiny_leela_core::{in_check, legal_moves, make_move, move_to_uci, parse_fen, search_root, Color, PositionEvaluator, SearchOptions, StudentEvaluator, START_FEN};

fn arg(name: &str, fallback: &str) -> String {
    let prefix = format!("{name}=");
    let args: Vec<String> = env::args().collect();
    for (i, value) in args.iter().enumerate() {
        if value.starts_with(&prefix) { return value[prefix.len()..].to_string(); }
        if value == name { if let Some(next) = args.get(i + 1) { return next.clone(); } }
    }
    fallback.to_string()
}

fn terminal_white_score(board: &tiny_leela_core::Board) -> Option<f32> {
    if !legal_moves(board).is_empty() { return None; }
    if !in_check(board, board.turn) { return Some(0.5); }
    Some(if board.turn == Color::White { 0.0 } else { 1.0 })
}

fn candidate_score(white_score: f32, candidate_color: Color) -> f32 {
    if (white_score - 0.5).abs() < 1e-6 { 0.5 }
    else if (white_score == 1.0 && candidate_color == Color::White) || (white_score == 0.0 && candidate_color == Color::Black) { 1.0 }
    else { 0.0 }
}

fn adjudicated_white_score(board: &tiny_leela_core::Board, evaluator: &StudentEvaluator, adjudicate: &str, threshold: f32) -> (f32, bool) {
    if let Some(score) = terminal_white_score(board) { return (score, false); }
    if adjudicate != "value" { return (0.5, false); }
    let evaln = evaluator.evaluate(board);
    let side_value = evaln.wdl[0] - evaln.wdl[2];
    if side_value.abs() <= threshold { return (0.5, true); }
    let side_to_move_wins = side_value > 0.0;
    let white_wins = (board.turn == Color::White && side_to_move_wins) || (board.turn == Color::Black && !side_to_move_wins);
    (if white_wins { 1.0 } else { 0.0 }, true)
}

fn main() {
    let candidate_path = arg("--candidate", "artifacts/student_distill_benchmark.json");
    let baseline_path = arg("--baseline", "artifacts/student_distill_benchmark.json");
    let games: usize = arg("--games", "4").parse().unwrap_or(4);
    let visits: u32 = arg("--visits", "8").parse().unwrap_or(8);
    let max_plies: usize = arg("--max-plies", "40").parse().unwrap_or(40);
    let progress_every: usize = arg("--progress-every", "4").parse().unwrap_or(4).max(1);
    let adjudicate = arg("--adjudicate", "terminal");
    let adjudicate_threshold: f32 = arg("--adjudicate-threshold", "0.02").parse().unwrap_or(0.02);
    let openings_text = arg("--openings", START_FEN);
    let openings: Vec<String> = openings_text.split('|').map(|s| s.to_string()).collect();

    let candidate_json = fs::read_to_string(&candidate_path).expect("read candidate");
    let baseline_json = fs::read_to_string(&baseline_path).expect("read baseline");
    let candidate = StudentEvaluator::from_json(&candidate_json).expect("parse candidate");
    let baseline = StudentEvaluator::from_json(&baseline_json).expect("parse baseline");
    let started = Instant::now();
    eprintln!("[rust-arena visits={visits}] start games={games} max_plies={max_plies} candidate={candidate_path} baseline={baseline_path}");

    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;
    let mut illegal_losses = 0usize;
    let mut plies_total = 0usize;
    let mut adjudicated_games = 0usize;
    let mut true_play_wins = 0usize;
    let mut true_play_draws = 0usize;
    let mut true_play_losses = 0usize;
    let mut adjudicated_score_sum = 0.0f64;

    for game in 0..games {
        let candidate_color = if game % 2 == 0 { Color::White } else { Color::Black };
        eprintln!("[rust-arena visits={visits}] game {}/{} start candidate_color={}", game + 1, games, if candidate_color == Color::White { "w" } else { "b" });
        let mut board = parse_fen(&openings[game % openings.len()]).expect("parse opening");
        let mut white_score = terminal_white_score(&board);
        let mut plies = 0usize;
        while white_score.is_none() && plies < max_plies {
            let side_is_candidate = board.turn == candidate_color;
            let evaluator = if side_is_candidate { &candidate } else { &baseline };
            let legal_uci: Vec<String> = legal_moves(&board).iter().map(|&m| move_to_uci(m)).collect();
            let result = search_root(&board, evaluator, SearchOptions { visits, cpuct: 1.5, temperature: 0.0 });
            let Some(mv) = result.mv else {
                white_score = Some(if in_check(&board, board.turn) { if board.turn == Color::White { 0.0 } else { 1.0 } } else { 0.5 });
                break;
            };
            let uci = move_to_uci(mv);
            if !legal_uci.contains(&uci) {
                if side_is_candidate { illegal_losses += 1; }
                white_score = Some(if side_is_candidate { if candidate_color == Color::White { 0.0 } else { 1.0 } } else { if candidate_color == Color::White { 1.0 } else { 0.0 } });
                break;
            }
            board = make_move(&board, mv);
            white_score = terminal_white_score(&board);
            plies += 1;
            if plies % progress_every == 0 {
                eprintln!("[rust-arena visits={visits}] game {}/{} ply={}/{} move={} elapsed_s={:.1}", game + 1, games, plies, max_plies, uci, started.elapsed().as_secs_f64());
            }
        }
        let true_white_score = white_score.unwrap_or(0.5);
        let true_score = candidate_score(true_white_score, candidate_color);
        if true_score == 1.0 { true_play_wins += 1; } else if true_score == 0.0 { true_play_losses += 1; } else { true_play_draws += 1; }
        let (white_score, adjudicated) = if let Some(score) = white_score { (score, false) } else { adjudicated_white_score(&board, if board.turn == candidate_color { &candidate } else { &baseline }, &adjudicate, adjudicate_threshold) };
        let score = candidate_score(white_score, candidate_color);
        if adjudicated {
            adjudicated_games += 1;
            adjudicated_score_sum += score as f64;
        }
        if score == 1.0 { wins += 1; } else if score == 0.0 { losses += 1; } else { draws += 1; }
        plies_total += plies;
        eprintln!("[rust-arena visits={visits}] game {}/{} done score={} wdl={}/{}/{} illegal={} elapsed_s={:.1}", game + 1, games, score, wins, draws, losses, illegal_losses, started.elapsed().as_secs_f64());
    }

    let score_rate = (wins as f64 + 0.5 * draws as f64) / games.max(1) as f64;
    let true_play_score_rate = (true_play_wins as f64 + 0.5 * true_play_draws as f64) / games.max(1) as f64;
    let adjudicated_score_rate = if adjudicated_games == 0 { 0.5 } else { adjudicated_score_sum / adjudicated_games.max(1) as f64 };
    let elo = if score_rate <= 0.0 { -999.0 } else if score_rate >= 1.0 { 999.0 } else { -400.0 * ((1.0 / score_rate) - 1.0).log10() };
    let promotion_ready = if score_rate > 0.55 && illegal_losses == 0 { 1 } else { 0 };
    println!("METRIC arena_backend_rust=1");
    println!("METRIC arena_score_rate={score_rate:.6}");
    println!("METRIC arena_candidate_elo_estimate={elo:.6}");
    println!("METRIC arena_games={games}");
    println!("METRIC arena_wins={wins}");
    println!("METRIC arena_draws={draws}");
    println!("METRIC arena_losses={losses}");
    println!("METRIC arena_illegal_losses={illegal_losses}");
    println!("METRIC arena_adjudicated_rate={:.6}", adjudicated_games as f64 / games.max(1) as f64);
    println!("METRIC arena_true_play_score_rate={true_play_score_rate:.6}");
    println!("METRIC arena_adjudicated_score_rate={adjudicated_score_rate:.6}");
    println!("METRIC arena_true_play_wins={true_play_wins}");
    println!("METRIC arena_true_play_draws={true_play_draws}");
    println!("METRIC arena_true_play_losses={true_play_losses}");
    println!("METRIC arena_avg_plies={:.6}", plies_total as f64 / games.max(1) as f64);
    println!("METRIC arena_promotion_ready={promotion_ready}");
}
