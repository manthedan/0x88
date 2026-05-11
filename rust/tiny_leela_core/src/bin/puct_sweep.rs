use serde::Serialize;
use std::{fs, path::Path, time::Instant};
use tiny_leela_core::{
    legal_moves, move_to_uci, parse_fen, search_root, SearchOptions, StudentEvaluator,
};

#[derive(Clone, Copy)]
struct SuiteCase {
    id: &'static str,
    fen: &'static str,
    best: &'static [&'static str],
    legal_only: bool,
}

#[derive(Serialize)]
struct DetailRow {
    id: String,
    fen: String,
    best_move: String,
    ok: bool,
    value: f32,
    visits_completed: u32,
}

#[derive(Serialize)]
struct SweepRow {
    mode: String,
    visits: u32,
    cpuct: f32,
    fpu: f32,
    pass: usize,
    total: usize,
    pass_rate: f32,
    elapsed_ms: f64,
    positions_per_second: f64,
    details: Vec<DetailRow>,
}

#[derive(Serialize)]
struct SweepOutput {
    artifact: String,
    protocol: SweepProtocol,
    rows: Vec<SweepRow>,
}

#[derive(Serialize)]
struct SweepProtocol {
    visits: Vec<u32>,
    cpucts: Vec<f32>,
    fpus: Vec<f32>,
    temperature: f32,
    suite: String,
}

const SUITE: &[SuiteCase] = &[
    SuiteCase {
        id: "mate_in_1_back_rank",
        fen: "6k1/5ppp/8/8/8/8/8/6KQ w - - 0 1",
        best: &["h1a8"],
        legal_only: false,
    },
    SuiteCase {
        id: "mate_in_1_rook",
        fen: "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        best: &["a1a8"],
        legal_only: false,
    },
    SuiteCase {
        id: "mate_in_1_queen_file",
        fen: "6k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1",
        best: &["d1d8"],
        legal_only: false,
    },
    SuiteCase {
        id: "mate_in_1_smother_net",
        fen: "6k1/6pp/8/8/8/5N2/6PP/6K1 w - - 0 1",
        best: &["f3g5"],
        legal_only: false,
    },
    SuiteCase {
        id: "win_hanging_queen",
        fen: "4k3/8/8/8/3q4/8/4Q3/4K3 w - - 0 1",
        best: &["e2d3", "e2d2", "e2e8"],
        legal_only: false,
    },
    SuiteCase {
        id: "must_recapture_queen",
        fen: "4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1",
        best: &["d2d4"],
        legal_only: false,
    },
    SuiteCase {
        id: "recapture_rook",
        fen: "4k3/8/8/8/4r3/8/4R3/4K3 w - - 0 1",
        best: &["e2e4"],
        legal_only: false,
    },
    SuiteCase {
        id: "knight_fork_king_queen",
        fen: "4k3/8/8/3q4/5N2/8/8/4K3 w - - 0 1",
        best: &["f4d5"],
        legal_only: false,
    },
    SuiteCase {
        id: "bishop_skewer_queen",
        fen: "4k3/6q1/8/8/8/2B5/8/4K3 w - - 0 1",
        best: &["c3g7"],
        legal_only: false,
    },
    SuiteCase {
        id: "promote_queen",
        fen: "6k1/P7/8/8/8/8/8/6K1 w - - 0 1",
        best: &["a7a8q"],
        legal_only: false,
    },
    SuiteCase {
        id: "underpromotion_knight_check",
        fen: "6k1/P7/8/8/8/8/5PPP/6K1 w - - 0 1",
        best: &["a7a8q", "a7a8n"],
        legal_only: false,
    },
    SuiteCase {
        id: "avoid_illegal_in_check_rook",
        fen: "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
        best: &[],
        legal_only: true,
    },
    SuiteCase {
        id: "avoid_illegal_in_check_bishop",
        fen: "4k3/8/8/8/8/8/3b4/4K3 w - - 0 1",
        best: &[],
        legal_only: true,
    },
    SuiteCase {
        id: "terminal_stalemate_sanity",
        fen: "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
        best: &[],
        legal_only: true,
    },
];

fn arg(args: &[String], name: &str, fallback: &str) -> String {
    let prefix = format!("{name}=");
    if let Some(v) = args.iter().find_map(|a| a.strip_prefix(&prefix)) {
        return v.to_string();
    }
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| fallback.to_string())
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}
fn parse_f32_list(s: &str) -> Vec<f32> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn eval_setting(
    evaluator: &StudentEvaluator,
    visits: u32,
    cpuct: f32,
    fpu: f32,
    temperature: f32,
) -> SweepRow {
    let started = Instant::now();
    let mut pass = 0usize;
    let mut details = Vec::with_capacity(SUITE.len());
    for case in SUITE {
        let board = parse_fen(case.fen).expect("parse built-in suite fen");
        let legal: Vec<String> = legal_moves(&board).into_iter().map(move_to_uci).collect();
        let result = search_root(
            &board,
            evaluator,
            SearchOptions {
                visits: visits.max(1),
                cpuct,
                fpu,
                temperature,
                ..SearchOptions::default()
            },
        );
        let uci = result
            .mv
            .map(move_to_uci)
            .unwrap_or_else(|| "none".to_string());
        let is_legal = legal.iter().any(|m| m == &uci);
        let ok = if case.legal_only {
            is_legal || (legal.is_empty() && uci == "none")
        } else {
            is_legal && case.best.iter().any(|m| *m == uci)
        };
        if ok {
            pass += 1;
        }
        details.push(DetailRow {
            id: case.id.to_string(),
            fen: case.fen.to_string(),
            best_move: uci,
            ok,
            value: result.value,
            visits_completed: result.visits,
        });
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    SweepRow {
        mode: "puct".to_string(),
        visits,
        cpuct,
        fpu,
        pass,
        total: SUITE.len(),
        pass_rate: pass as f32 / SUITE.len() as f32,
        elapsed_ms,
        positions_per_second: SUITE.len() as f64 / (elapsed_ms / 1000.0).max(1e-9),
        details,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let artifact = args
        .get(1)
        .filter(|s| !s.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| {
            arg(
                &args,
                "--artifact",
                "artifacts/student_distill_benchmark.json",
            )
        });
    let out = arg(&args, "--out", "artifacts/rust_puct_visit_sweep/sweep.json");
    let visits = parse_u32_list(&arg(&args, "--visits", "1,2,4,8,16,32,64"));
    let cpucts = parse_f32_list(&arg(&args, "--cpucts", "1.0,1.25,1.5,1.75,2.0"));
    let fpus = parse_f32_list(&arg(&args, "--fpus", "0"));
    let temperature: f32 = arg(&args, "--temperature", "0")
        .parse()
        .expect("parse --temperature");

    let json = fs::read_to_string(&artifact).expect("read student artifact");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse student artifact");
    let mut rows = Vec::new();
    for &v in &visits {
        for &c in &cpucts {
            for &f in &fpus {
                let row = eval_setting(&evaluator, v, c, f, temperature);
                println!("RESULT mode=puct visits={} cpuct={} fpu={} pass={}/{} rate={:.6} positions_per_second={:.3}", row.visits, row.cpuct, row.fpu, row.pass, row.total, row.pass_rate, row.positions_per_second);
                rows.push(row);
            }
        }
    }
    let best = rows
        .iter()
        .max_by(|a, b| {
            a.pass_rate
                .total_cmp(&b.pass_rate)
                .then_with(|| b.elapsed_ms.total_cmp(&a.elapsed_ms))
        })
        .expect("at least one row");
    println!(
        "METRIC rust_puct_sweep_best_pass_rate={:.6}",
        best.pass_rate
    );
    println!("METRIC rust_puct_sweep_best_visits={}", best.visits);
    println!("METRIC rust_puct_sweep_best_cpuct={}", best.cpuct);
    println!("METRIC rust_puct_sweep_best_fpu={}", best.fpu);

    if let Some(parent) = Path::new(&out).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    let output = SweepOutput {
        artifact,
        protocol: SweepProtocol {
            visits,
            cpucts,
            fpus,
            temperature,
            suite: "rust_builtin_tactical_v1".to_string(),
        },
        rows,
    };
    fs::write(
        &out,
        serde_json::to_string_pretty(&output).expect("serialize sweep json"),
    )
    .expect("write sweep json");
    println!("wrote={out}");
}
