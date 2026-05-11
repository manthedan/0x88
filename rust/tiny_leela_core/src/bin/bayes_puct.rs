use serde::Serialize;
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};
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

#[derive(Clone, Copy, Debug, Serialize)]
struct Weights {
    cpuct: f32,
    fpu: f32,
}

#[derive(Clone, Debug, Serialize)]
struct Observation {
    iter: usize,
    name: String,
    weights: Weights,
    score_rate: f32,
    score: usize,
    total: usize,
    wall_seconds: f64,
    posterior_mean: f32,
    posterior_std: f32,
    posterior_acq: f32,
}

#[derive(Clone, Copy)]
struct Posterior {
    mean: f32,
    std: f32,
    acq: f32,
    pseudo_games: f32,
}

#[derive(Serialize)]
struct State<'a> {
    protocol: Protocol<'a>,
    best: Option<&'a Observation>,
    observations: &'a [Observation],
}

#[derive(Serialize)]
struct Protocol<'a> {
    artifact: &'a str,
    visits: u32,
    iterations: usize,
    pool_size: usize,
    seed: u64,
    beta: f32,
    length_scale: f32,
    cpuct_min: f32,
    cpuct_max: f32,
    fpu_min: f32,
    fpu_max: f32,
    suite: &'a str,
}

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

fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) as f64 + 0.5) / (u64::MAX as f64 + 1.0)
}

fn key(w: Weights) -> String {
    format!("{:.5}_{:.5}", w.cpuct, w.fpu)
}
fn name(iter: usize, w: Weights) -> String {
    format!(
        "cand{iter:03}_c{:04}_f{}{:03}",
        (w.cpuct * 1000.0).round() as i32,
        if w.fpu < 0.0 { "m" } else { "p" },
        (w.fpu.abs() * 1000.0).round() as i32
    )
}
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}
fn round_step(x: f32, step: f32) -> f32 {
    (x / step).round() * step
}

fn normalize_weight(
    w: Weights,
    cpuct_min: f32,
    cpuct_max: f32,
    fpu_min: f32,
    fpu_max: f32,
) -> Weights {
    Weights {
        cpuct: round_step(clamp(w.cpuct, cpuct_min, cpuct_max), 0.025),
        fpu: round_step(clamp(w.fpu, fpu_min, fpu_max), 0.025),
    }
}

fn make_pool(
    pool_size: usize,
    seed: u64,
    cpuct_min: f32,
    cpuct_max: f32,
    fpu_min: f32,
    fpu_max: f32,
) -> Vec<Weights> {
    let mut pool = Vec::new();
    for &c in &[1.0, 1.25, 1.5, 1.75, 2.0] {
        pool.push(normalize_weight(
            Weights { cpuct: c, fpu: 0.0 },
            cpuct_min,
            cpuct_max,
            fpu_min,
            fpu_max,
        ));
    }
    for &f in &[-0.20, -0.10, 0.0, 0.10, 0.20] {
        pool.push(normalize_weight(
            Weights { cpuct: 1.5, fpu: f },
            cpuct_min,
            cpuct_max,
            fpu_min,
            fpu_max,
        ));
    }
    for &c in &[cpuct_min, (cpuct_min + cpuct_max) * 0.5, cpuct_max] {
        for &f in &[fpu_min, 0.0, fpu_max] {
            pool.push(normalize_weight(
                Weights { cpuct: c, fpu: f },
                cpuct_min,
                cpuct_max,
                fpu_min,
                fpu_max,
            ));
        }
    }
    let mut rng = seed;
    while pool.len() < pool_size {
        let c = cpuct_min + (cpuct_max - cpuct_min) * splitmix64(&mut rng) as f32;
        let f = fpu_min + (fpu_max - fpu_min) * splitmix64(&mut rng) as f32;
        pool.push(normalize_weight(
            Weights { cpuct: c, fpu: f },
            cpuct_min,
            cpuct_max,
            fpu_min,
            fpu_max,
        ));
    }
    let mut seen = HashSet::new();
    pool.into_iter().filter(|w| seen.insert(key(*w))).collect()
}

fn dist2(a: Weights, b: Weights, cpuct_scale: f32, fpu_scale: f32) -> f32 {
    ((a.cpuct - b.cpuct) / cpuct_scale).powi(2) + ((a.fpu - b.fpu) / fpu_scale).powi(2)
}

fn surrogate(
    w: Weights,
    obs: &[Observation],
    beta: f32,
    length_scale: f32,
    cpuct_scale: f32,
    fpu_scale: f32,
) -> Posterior {
    if obs.is_empty() {
        return Posterior {
            mean: 0.5,
            std: 0.25,
            acq: 0.5 + beta * 0.25,
            pseudo_games: 0.0,
        };
    }
    let mut sw = 0.0f32;
    let mut sy = 0.0f32;
    for o in obs {
        let k = (-dist2(w, o.weights, cpuct_scale, fpu_scale)
            / (2.0 * length_scale * length_scale))
            .exp();
        sw += k * o.total as f32;
        sy += k * o.score as f32;
    }
    let mean = if sw > 1e-9 { sy / sw } else { 0.5 };
    let mut rv = 0.0f32;
    let mut rw = 0.0f32;
    for o in obs {
        let k = (-dist2(w, o.weights, cpuct_scale, fpu_scale)
            / (2.0 * length_scale * length_scale))
            .exp();
        rv += k * o.total as f32 * (o.score_rate - mean).powi(2);
        rw += k * o.total as f32;
    }
    let residual = if rw > 1e-9 { rv / rw } else { 0.03 };
    let binom = (mean * (1.0 - mean)).max(0.0025) / (sw + 2.0).max(1.0);
    let prior = 0.02 / (1.0 + sw).sqrt();
    let std = (binom + residual + prior * prior).sqrt();
    Posterior {
        mean,
        std,
        acq: mean + beta * std,
        pseudo_games: sw,
    }
}

fn eval_candidate(evaluator: &StudentEvaluator, visits: u32, w: Weights) -> (usize, usize) {
    let mut pass = 0usize;
    for case in SUITE {
        let _case_id = case.id;
        let board = parse_fen(case.fen).expect("parse built-in suite fen");
        let legal: Vec<String> = legal_moves(&board).into_iter().map(move_to_uci).collect();
        let result = search_root(
            &board,
            evaluator,
            SearchOptions {
                visits: visits.max(1),
                cpuct: w.cpuct,
                fpu: w.fpu,
                temperature: 0.0,
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
    }
    (pass, SUITE.len())
}

fn append_line(path: &Path, line: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open append file");
    writeln!(file, "{line}").expect("append line");
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
    let out_dir = arg(&args, "--out-dir", "artifacts/rust_bayes_puct");
    let visits: u32 = arg(&args, "--visits", "32")
        .parse()
        .expect("parse --visits");
    let iterations: usize = arg(&args, "--iterations", "24")
        .parse()
        .expect("parse --iterations");
    let pool_size: usize = arg(&args, "--pool-size", "160")
        .parse()
        .expect("parse --pool-size");
    let seed: u64 = arg(&args, "--seed", "23").parse().expect("parse --seed");
    let beta: f32 = arg(&args, "--beta", "0.9").parse().expect("parse --beta");
    let length_scale: f32 = arg(&args, "--length-scale", "0.45")
        .parse()
        .expect("parse --length-scale");
    let cpuct_min: f32 = arg(&args, "--cpuct-min", "0.75")
        .parse()
        .expect("parse --cpuct-min");
    let cpuct_max: f32 = arg(&args, "--cpuct-max", "2.5")
        .parse()
        .expect("parse --cpuct-max");
    let fpu_min: f32 = arg(&args, "--fpu-min", "-0.25")
        .parse()
        .expect("parse --fpu-min");
    let fpu_max: f32 = arg(&args, "--fpu-max", "0.25")
        .parse()
        .expect("parse --fpu-max");
    let cpuct_scale: f32 = arg(&args, "--cpuct-scale", "0.35")
        .parse()
        .expect("parse --cpuct-scale");
    let fpu_scale: f32 = arg(&args, "--fpu-scale", "0.15")
        .parse()
        .expect("parse --fpu-scale");

    fs::create_dir_all(&out_dir).expect("create out dir");
    let obs_path = Path::new(&out_dir).join("observations.jsonl");
    let ledger_path = Path::new(&out_dir).join("ledger.jsonl");
    let summary_path = Path::new(&out_dir).join("summary.tsv");
    let state_path = Path::new(&out_dir).join("state.json");
    if !summary_path.exists() {
        fs::write(&summary_path, "iter\tname\tcpuct\tfpu\tscore_rate\tscore\ttotal\twall_seconds\tposterior_mean\tposterior_std\tposterior_acq\n").expect("write summary header");
    }

    let json = fs::read_to_string(&artifact).expect("read student artifact");
    let evaluator = StudentEvaluator::from_json(&json).expect("parse student artifact");
    let pool = make_pool(pool_size, seed, cpuct_min, cpuct_max, fpu_min, fpu_max);
    let mut obs: Vec<Observation> = Vec::new();
    let mut evaluated = HashSet::new();
    println!("[rust-bayes-puct] out_dir={out_dir} visits={visits} iterations={iterations} pool={} suite={} artifact={artifact}", pool.len(), SUITE.len());

    for iter in 1..=iterations {
        let mut chosen = None;
        let mut chosen_post = None;
        for &w in &pool {
            if evaluated.contains(&key(w)) {
                continue;
            }
            let p = surrogate(w, &obs, beta, length_scale, cpuct_scale, fpu_scale);
            if chosen_post
                .map(|old: Posterior| {
                    p.acq > old.acq || (p.acq == old.acq && p.pseudo_games < old.pseudo_games)
                })
                .unwrap_or(true)
            {
                chosen = Some(w);
                chosen_post = Some(p);
            }
        }
        let Some(w) = chosen else {
            break;
        };
        let p = chosen_post.expect("posterior exists");
        let nm = name(iter, w);
        let started = Instant::now();
        let (score, total) = eval_candidate(&evaluator, visits, w);
        let wall_seconds = started.elapsed().as_secs_f64();
        let score_rate = score as f32 / total as f32;
        let o = Observation {
            iter,
            name: nm.clone(),
            weights: w,
            score_rate,
            score,
            total,
            wall_seconds,
            posterior_mean: p.mean,
            posterior_std: p.std,
            posterior_acq: p.acq,
        };
        append_line(
            &obs_path,
            &serde_json::to_string(&o).expect("serialize observation"),
        );
        append_line(&ledger_path, &serde_json::json!({
            "trial_id": nm,
            "status": "succeeded",
            "params": { "visits": visits, "cpuct": w.cpuct, "fpu": w.fpu },
            "score": score_rate,
            "raw_metrics": { "score": score, "total": total, "suite": "rust_builtin_tactical_v1" },
            "cost": { "wall_seconds": wall_seconds, "games": 0, "positions": total, "visits": visits }
        }).to_string());
        append_line(
            &summary_path,
            &format!(
                "{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.3}\t{:.6}\t{:.6}\t{:.6}",
                iter,
                nm,
                w.cpuct,
                w.fpu,
                score_rate,
                score,
                total,
                wall_seconds,
                p.mean,
                p.std,
                p.acq
            ),
        );
        println!("RESULT iter={iter} name={} cpuct={} fpu={} score={}/{} rate={:.6} posterior_mean={:.4} posterior_std={:.4} acq={:.4}", o.name, w.cpuct, w.fpu, score, total, score_rate, p.mean, p.std, p.acq);
        evaluated.insert(key(w));
        obs.push(o);
        let mut ranked: Vec<&Observation> = obs.iter().collect();
        ranked.sort_by(|a, b| {
            b.score_rate
                .total_cmp(&a.score_rate)
                .then_with(|| a.wall_seconds.total_cmp(&b.wall_seconds))
        });
        let state = State {
            protocol: Protocol {
                artifact: &artifact,
                visits,
                iterations,
                pool_size,
                seed,
                beta,
                length_scale,
                cpuct_min,
                cpuct_max,
                fpu_min,
                fpu_max,
                suite: "rust_builtin_tactical_v1",
            },
            best: ranked.first().copied(),
            observations: &obs,
        };
        fs::write(
            &state_path,
            serde_json::to_string_pretty(&state).expect("serialize state"),
        )
        .expect("write state");
    }

    let mut ranked: Vec<&Observation> = obs.iter().collect();
    ranked.sort_by(|a, b| {
        b.score_rate
            .total_cmp(&a.score_rate)
            .then_with(|| a.wall_seconds.total_cmp(&b.wall_seconds))
    });
    println!("RUST_BAYES_PUCT_TOP");
    for (rank, o) in ranked.iter().take(10).enumerate() {
        println!(
            "rank={} iter={} name={} scoreRate={:.6} score={}/{} cpuct={} fpu={}",
            rank + 1,
            o.iter,
            o.name,
            o.score_rate,
            o.score,
            o.total,
            o.weights.cpuct,
            o.weights.fpu
        );
    }
    if let Some(best) = ranked.first() {
        println!(
            "METRIC rust_bayes_puct_best_pass_rate={:.6}",
            best.score_rate
        );
        println!("METRIC rust_bayes_puct_best_cpuct={}", best.weights.cpuct);
        println!("METRIC rust_bayes_puct_best_fpu={}", best.weights.fpu);
    }
}
