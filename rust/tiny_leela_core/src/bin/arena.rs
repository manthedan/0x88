use serde::Serialize;
use std::io::Write;
use std::{
    env, fs,
    process::{Command, Stdio},
    time::Instant,
};
#[cfg(feature = "native-ort")]
use tiny_leela_core::OnnxEvaluator;
use tiny_leela_core::{
    board_to_fen, in_check, legal_moves, make_move, move_to_uci, parse_fen,
    search_root_with_history, Color, PositionEvaluator, SearchOptions, StudentEvaluator, START_FEN,
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

fn terminal_white_score(board: &tiny_leela_core::Board) -> Option<f32> {
    if !legal_moves(board).is_empty() {
        return None;
    }
    if !in_check(board, board.turn) {
        return Some(0.5);
    }
    Some(if board.turn == Color::White { 0.0 } else { 1.0 })
}

fn candidate_score(white_score: f32, candidate_color: Color) -> f32 {
    if (white_score - 0.5).abs() < 1e-6 {
        0.5
    } else if (white_score == 1.0 && candidate_color == Color::White)
        || (white_score == 0.0 && candidate_color == Color::Black)
    {
        1.0
    } else {
        0.0
    }
}

fn stockfish_white_score(
    board: &tiny_leela_core::Board,
    stockfish: &str,
    depth: u32,
    draw_cp: i32,
) -> Option<f32> {
    let fen = board_to_fen(board);
    let mut child = Command::new(stockfish)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    {
        let stdin = child.stdin.as_mut()?;
        writeln!(stdin, "uci").ok()?;
        writeln!(stdin, "isready").ok()?;
        writeln!(stdin, "position fen {fen}").ok()?;
        writeln!(stdin, "go depth {depth}").ok()?;
        writeln!(stdin, "quit").ok()?;
    }
    let output = child.wait_with_output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut side_cp: Option<i32> = None;
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        for i in 0..parts.len().saturating_sub(2) {
            if parts[i] == "score" && parts[i + 1] == "cp" {
                side_cp = parts[i + 2].parse::<i32>().ok();
            }
            if parts[i] == "score" && parts[i + 1] == "mate" {
                if let Ok(m) = parts[i + 2].parse::<i32>() {
                    side_cp = Some(if m > 0 { 100000 } else { -100000 });
                }
            }
        }
    }
    let side_cp = side_cp?;
    let white_cp = if board.turn == Color::White {
        side_cp
    } else {
        -side_cp
    };
    if white_cp.abs() <= draw_cp {
        Some(0.5)
    } else if white_cp > 0 {
        Some(1.0)
    } else {
        Some(0.0)
    }
}

fn adjudicated_white_score(
    board: &tiny_leela_core::Board,
    history: &[String],
    evaluator: &dyn PositionEvaluator,
    adjudicate: &str,
    threshold: f32,
    stockfish: &str,
    stockfish_depth: u32,
    stockfish_draw_cp: i32,
) -> (f32, bool) {
    if let Some(score) = terminal_white_score(board) {
        return (score, false);
    }
    if adjudicate == "stockfish" {
        return (
            stockfish_white_score(board, stockfish, stockfish_depth, stockfish_draw_cp)
                .unwrap_or(0.5),
            true,
        );
    }
    if adjudicate != "value" {
        return (0.5, false);
    }
    let evaln = evaluator.evaluate_with_history(board, history);
    let side_value = evaln.wdl[0] - evaln.wdl[2];
    if side_value.abs() <= threshold {
        return (0.5, true);
    }
    let side_to_move_wins = side_value > 0.0;
    let white_wins = (board.turn == Color::White && side_to_move_wins)
        || (board.turn == Color::Black && !side_to_move_wins);
    (if white_wins { 1.0 } else { 0.0 }, true)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameRecord {
    game: usize,
    candidate_color: String,
    opening: String,
    score: f32,
    true_score: f32,
    plies: usize,
    final_fen: String,
    adjudicated: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ArenaProtocol {
    backend: String,
    candidate_name: String,
    baseline_name: String,
    candidate: String,
    baseline: String,
    visits: u32,
    cpuct: f32,
    fpu: f32,
    games: usize,
    max_plies: usize,
    adjudicate: String,
    openings: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StandingRecord {
    name: String,
    wins: usize,
    draws: usize,
    losses: usize,
    score: f64,
    games: usize,
    score_rate: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PairRecord {
    a: String,
    b: String,
    a_score: f64,
    games: usize,
    a_wdl: [usize; 3],
    a_score_rate: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ArenaOutput {
    protocol: ArenaProtocol,
    standings: Vec<StandingRecord>,
    pairs: Vec<PairRecord>,
    score_rate: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    illegal_losses: usize,
    games: Vec<GameRecord>,
}

fn load_evaluator(
    json_path: &str,
    onnx_path: &str,
    meta_path: &str,
    label: &str,
) -> Box<dyn PositionEvaluator> {
    if !onnx_path.is_empty() || !meta_path.is_empty() || json_path.ends_with(".onnx") {
        #[cfg(feature = "native-ort")]
        {
            let model = if onnx_path.is_empty() {
                json_path
            } else {
                onnx_path
            };
            let meta = if meta_path.is_empty() {
                panic!("{label} ONNX requested but meta path is empty; pass --{label}-meta");
            } else {
                meta_path
            };
            return Box::new(OnnxEvaluator::from_files(model, meta).expect("load ONNX evaluator"));
        }
        #[cfg(not(feature = "native-ort"))]
        panic!("{label} ONNX requested but tiny-leela-rust-arena was built without --features native-ort");
    }
    let json = fs::read_to_string(json_path).expect("read student artifact");
    Box::new(StudentEvaluator::from_json(&json).expect("parse student artifact"))
}

fn main() {
    let candidate_path = arg("--candidate", "artifacts/student_distill_benchmark.json");
    let baseline_path = arg("--baseline", "artifacts/student_distill_benchmark.json");
    let candidate_onnx = arg("--candidate-onnx", "");
    let candidate_meta = arg("--candidate-meta", "");
    let baseline_onnx = arg("--baseline-onnx", "");
    let baseline_meta = arg("--baseline-meta", "");
    let candidate_name = arg("--candidate-name", "candidate");
    let baseline_name = arg("--baseline-name", "baseline");
    let games: usize = arg("--games", "4").parse().unwrap_or(4);
    let start_game: usize = arg("--start-game", "0").parse().unwrap_or(0);
    let visits: u32 = arg("--visits", "8").parse().unwrap_or(8);
    let cpuct: f32 = arg("--cpuct", "1.5").parse().unwrap_or(1.5);
    let fpu: f32 = arg("--fpu", "0").parse().unwrap_or(0.0);
    let max_plies: usize = arg("--max-plies", "40").parse().unwrap_or(40);
    let progress_every: usize = arg("--progress-every", "4").parse().unwrap_or(4).max(1);
    let adjudicate = arg("--adjudicate", "terminal");
    let adjudicate_threshold: f32 = arg("--adjudicate-threshold", "0.02")
        .parse()
        .unwrap_or(0.02);
    let stockfish = arg(
        "--stockfish",
        &env::var("STOCKFISH_BIN").unwrap_or_else(|_| "stockfish".to_string()),
    );
    let stockfish_depth: u32 = arg("--stockfish-depth", "8").parse().unwrap_or(8);
    let stockfish_draw_cp: i32 = arg("--stockfish-draw-cp", "50").parse().unwrap_or(50);
    let openings_text = arg("--openings", START_FEN);
    let openings: Vec<String> = openings_text.split('|').map(|s| s.to_string()).collect();
    let out_path = arg("--out", "");

    let candidate = load_evaluator(
        &candidate_path,
        &candidate_onnx,
        &candidate_meta,
        "candidate",
    );
    let baseline = load_evaluator(&baseline_path, &baseline_onnx, &baseline_meta, "baseline");
    let candidate_display = if candidate_onnx.is_empty() {
        candidate_path.as_str()
    } else {
        candidate_onnx.as_str()
    };
    let baseline_display = if baseline_onnx.is_empty() {
        baseline_path.as_str()
    } else {
        baseline_onnx.as_str()
    };
    let started = Instant::now();
    eprintln!("[rust-arena visits={visits} cpuct={cpuct} fpu={fpu}] start games={games} start_game={start_game} max_plies={max_plies} candidate={candidate_display} baseline={baseline_display}");

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
    let mut max_ply_draws = 0usize;

    let mut game_records = Vec::new();
    for local_game in 0..games {
        let game = start_game + local_game;
        let candidate_color = if game % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };
        eprintln!(
            "[rust-arena visits={visits}] game {}/{} global={} start candidate_color={}",
            local_game + 1,
            games,
            game,
            if candidate_color == Color::White {
                "w"
            } else {
                "b"
            }
        );
        let opening = openings[(game / 2) % openings.len()].clone();
        let mut board = parse_fen(&opening).expect("parse opening");
        let mut history: Vec<String> = Vec::new();
        let mut white_score = terminal_white_score(&board);
        let mut plies = 0usize;
        while white_score.is_none() && plies < max_plies {
            let side_is_candidate = board.turn == candidate_color;
            let evaluator: &dyn PositionEvaluator = if side_is_candidate {
                candidate.as_ref()
            } else {
                baseline.as_ref()
            };
            let legal_uci: Vec<String> = legal_moves(&board)
                .iter()
                .map(|&m| move_to_uci(m))
                .collect();
            let history_fens: Vec<String> = history.iter().rev().take(2).cloned().collect();
            let result = search_root_with_history(
                &board,
                evaluator,
                SearchOptions {
                    visits,
                    cpuct,
                    fpu,
                    temperature: 0.0,
                },
                &history_fens,
            );
            let Some(mv) = result.mv else {
                white_score = Some(if in_check(&board, board.turn) {
                    if board.turn == Color::White {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    0.5
                });
                break;
            };
            let uci = move_to_uci(mv);
            if !legal_uci.contains(&uci) {
                if side_is_candidate {
                    illegal_losses += 1;
                }
                white_score = Some(if side_is_candidate {
                    if candidate_color == Color::White {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    if candidate_color == Color::White {
                        1.0
                    } else {
                        0.0
                    }
                });
                break;
            }
            history.push(board_to_fen(&board));
            board = make_move(&board, mv);
            white_score = terminal_white_score(&board);
            plies += 1;
            if plies % progress_every == 0 {
                eprintln!(
                    "[rust-arena visits={visits}] game {}/{} ply={}/{} move={} elapsed_s={:.1}",
                    local_game + 1,
                    games,
                    plies,
                    max_plies,
                    uci,
                    started.elapsed().as_secs_f64()
                );
            }
        }
        let hit_max_ply_draw = white_score.is_none() && plies >= max_plies;
        if hit_max_ply_draw {
            max_ply_draws += 1;
        }
        let true_white_score = white_score.unwrap_or(0.5);
        let true_score = candidate_score(true_white_score, candidate_color);
        if true_score == 1.0 {
            true_play_wins += 1;
        } else if true_score == 0.0 {
            true_play_losses += 1;
        } else {
            true_play_draws += 1;
        }
        let history_fens: Vec<String> = history.iter().rev().take(2).cloned().collect();
        let adjudicator: &dyn PositionEvaluator = if board.turn == candidate_color {
            candidate.as_ref()
        } else {
            baseline.as_ref()
        };
        let (white_score, adjudicated) = if let Some(score) = white_score {
            (score, false)
        } else {
            adjudicated_white_score(
                &board,
                &history_fens,
                adjudicator,
                &adjudicate,
                adjudicate_threshold,
                &stockfish,
                stockfish_depth,
                stockfish_draw_cp,
            )
        };
        let score = candidate_score(white_score, candidate_color);
        if adjudicated {
            adjudicated_games += 1;
            adjudicated_score_sum += score as f64;
        }
        if score == 1.0 {
            wins += 1;
        } else if score == 0.0 {
            losses += 1;
        } else {
            draws += 1;
        }
        plies_total += plies;
        game_records.push(GameRecord {
            game,
            candidate_color: if candidate_color == Color::White {
                "white".to_string()
            } else {
                "black".to_string()
            },
            opening,
            score,
            true_score,
            plies,
            final_fen: board_to_fen(&board),
            adjudicated,
        });
        eprintln!("[rust-arena visits={visits}] game {}/{} done score={} wdl={}/{}/{} illegal={} elapsed_s={:.1}", local_game + 1, games, score, wins, draws, losses, illegal_losses, started.elapsed().as_secs_f64());
    }

    let score_rate = (wins as f64 + 0.5 * draws as f64) / games.max(1) as f64;
    let true_play_score_rate =
        (true_play_wins as f64 + 0.5 * true_play_draws as f64) / games.max(1) as f64;
    let adjudicated_score_rate = if adjudicated_games == 0 {
        0.5
    } else {
        adjudicated_score_sum / adjudicated_games.max(1) as f64
    };
    let elo = if score_rate <= 0.0 {
        -999.0
    } else if score_rate >= 1.0 {
        999.0
    } else {
        -400.0 * ((1.0 / score_rate) - 1.0).log10()
    };
    let promotion_ready = if score_rate > 0.55 && illegal_losses == 0 {
        1
    } else {
        0
    };
    if !out_path.is_empty() {
        if let Some(parent) = std::path::Path::new(&out_path).parent() {
            fs::create_dir_all(parent).expect("create arena output dir");
        }
        let cand_score = wins as f64 + 0.5 * draws as f64;
        let base_score = losses as f64 + 0.5 * draws as f64;
        let output = ArenaOutput {
            protocol: ArenaProtocol {
                backend: if candidate_onnx.is_empty()
                    && baseline_onnx.is_empty()
                    && !candidate_path.ends_with(".onnx")
                    && !baseline_path.ends_with(".onnx")
                {
                    "rust-student-json".to_string()
                } else {
                    "rust-native-ort".to_string()
                },
                candidate_name: candidate_name.clone(),
                baseline_name: baseline_name.clone(),
                candidate: if candidate_onnx.is_empty() {
                    candidate_path.clone()
                } else {
                    candidate_onnx.clone()
                },
                baseline: if baseline_onnx.is_empty() {
                    baseline_path.clone()
                } else {
                    baseline_onnx.clone()
                },
                visits,
                cpuct,
                fpu,
                games,
                max_plies,
                adjudicate: adjudicate.clone(),
                openings: openings.len(),
            },
            standings: vec![
                StandingRecord {
                    name: candidate_name.clone(),
                    wins,
                    draws,
                    losses,
                    score: cand_score,
                    games,
                    score_rate,
                },
                StandingRecord {
                    name: baseline_name.clone(),
                    wins: losses,
                    draws,
                    losses: wins,
                    score: base_score,
                    games,
                    score_rate: base_score / games.max(1) as f64,
                },
            ],
            pairs: vec![PairRecord {
                a: candidate_name.clone(),
                b: baseline_name.clone(),
                a_score: cand_score,
                games,
                a_wdl: [wins, draws, losses],
                a_score_rate: score_rate,
            }],
            score_rate,
            wins,
            draws,
            losses,
            illegal_losses,
            games: game_records,
        };
        fs::write(
            &out_path,
            serde_json::to_string_pretty(&output).expect("serialize arena output"),
        )
        .expect("write arena output");
    }
    println!("METRIC arena_backend_rust=1");
    println!("METRIC arena_score_rate={score_rate:.6}");
    println!("METRIC arena_candidate_elo_estimate={elo:.6}");
    println!("METRIC arena_games={games}");
    println!("METRIC arena_wins={wins}");
    println!("METRIC arena_draws={draws}");
    println!("METRIC arena_losses={losses}");
    println!("METRIC arena_illegal_losses={illegal_losses}");
    println!(
        "METRIC arena_adjudicated_rate={:.6}",
        adjudicated_games as f64 / games.max(1) as f64
    );
    println!("METRIC arena_true_play_score_rate={true_play_score_rate:.6}");
    println!("METRIC arena_adjudicated_score_rate={adjudicated_score_rate:.6}");
    println!("METRIC arena_true_play_wins={true_play_wins}");
    println!("METRIC arena_true_play_draws={true_play_draws}");
    println!("METRIC arena_true_play_losses={true_play_losses}");
    println!("METRIC arena_max_ply_draws={max_ply_draws}");
    println!(
        "METRIC arena_max_ply_draw_rate={:.6}",
        max_ply_draws as f64 / games.max(1) as f64
    );
    println!(
        "METRIC arena_avg_plies={:.6}",
        plies_total as f64 / games.max(1) as f64
    );
    println!("METRIC arena_promotion_ready={promotion_ready}");
}
