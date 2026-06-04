#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const source = resolve(process.env.RECKLESS_SOURCE_DIR ?? '.local_engines/reckless-wasi-src');
const workdir = resolve(process.env.RECKLESS_API_PROBE_DIR ?? '.local_engines/reckless-browser-api-probe');
const fen = process.argv[2] ?? 'startpos';
const depth = Number.parseInt(process.argv[3] ?? '7', 10);
const evalfile = process.env.RECKLESS_EVALFILE ? resolve(process.env.RECKLESS_EVALFILE) : '';
if (!Number.isFinite(depth) || depth < 1) throw new Error(`depth must be a positive integer, got ${process.argv[3]}`);

function run(cmd, args, options = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  return execFileSync(cmd, args, { stdio: 'inherit', ...options });
}

if (!existsSync(`${source}/Cargo.toml`)) {
  throw new Error(`Reckless source missing at ${source}; run npm run reckless:build-wasi first or set RECKLESS_SOURCE_DIR`);
}
const nnueSource = readFileSync(`${source}/src/nnue.rs`, 'utf8');
const l1Size = Number(nnueSource.match(/const L1_SIZE: usize = (\d+);/)?.[1] ?? '768');
if (l1Size !== 768 && !evalfile) {
  throw new Error(`Reckless source has custom L1_SIZE=${l1Size}; set RECKLESS_EVALFILE so the probe builds against the matching NNUE file`);
}

rmSync(workdir, { recursive: true, force: true });
mkdirSync(dirname(workdir), { recursive: true });
run('rsync', ['-a', '--exclude', '.git', '--exclude', 'target', `${source}/`, `${workdir}/`]);

writeFileSync(`${workdir}/src/lib.rs`, `#![allow(unsafe_op_in_unsafe_fn)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::redundant_clone)]

pub mod board;
pub mod browser_api;
pub mod evaluation;
pub mod history;
pub mod lookup;
pub mod misc;
pub mod movepick;
pub mod nnue;
pub mod numa;
pub mod parameters;
pub mod search;
pub mod setwise;
pub mod stack;
pub mod thread;
pub mod threadpool;
pub mod time;
pub mod tools;
pub mod transposition;
pub mod types;
pub mod uci;

#[cfg(feature = "syzygy")]
pub mod tb;

#[cfg(feature = "syzygy")]
#[allow(warnings)]
pub mod bindings;
`);

writeFileSync(`${workdir}/src/browser_api.rs`, `use std::{sync::{Arc, Once}, time::Duration};

use crate::{
    board::{Board, NullBoardObserver},
    search::Report,
    thread::SharedContext,
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    transposition::DEFAULT_TT_SIZE,
    types::{Move, Score, normalize_to_cp},
};

#[derive(Clone, Debug)]
pub struct SearchLine {
    pub multipv: usize,
    pub depth: i32,
    pub score_cp: Option<i32>,
    pub mate_in: Option<i32>,
    pub nodes: u64,
    pub nps: u64,
    pub pv: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub elapsed_ms: u128,
    pub lines: Vec<SearchLine>,
}

static INIT: Once = Once::new();

pub struct BrowserEngine {
    shared: Arc<SharedContext>,
    threads: ThreadPool,
    board: Board,
    hash_mb: usize,
    multi_pv: usize,
    move_overhead_ms: u64,
}

impl BrowserEngine {
    pub fn new(hash_mb: usize) -> Self {
        INIT.call_once(|| {
            crate::lookup::initialize();
            crate::nnue::initialize();
        });
        let shared = Arc::new(SharedContext::default());
        let threads = ThreadPool::new(shared.clone());
        let mut engine = Self {
            shared,
            threads,
            board: Board::starting_position(),
            hash_mb: hash_mb.max(1),
            multi_pv: 1,
            move_overhead_ms: 100,
        };
        engine.resize_hash(engine.hash_mb);
        engine
    }

    pub fn set_fen(&mut self, fen: &str) -> Result<(), String> {
        self.board = if fen.trim().eq_ignore_ascii_case("startpos") {
            Board::starting_position()
        } else {
            Board::from_fen(fen).map_err(|error| format!("{error:?}"))?
        };
        Ok(())
    }

    pub fn make_uci_move(&mut self, uci_move: &str) -> bool {
        let moves = self.board.generate_all_moves();
        if let Some(mv) = moves.iter().map(|entry| entry.mv).find(|mv| mv.to_uci(&self.board) == uci_move) {
            self.board.make_move(mv, &mut NullBoardObserver);
            true
        } else {
            false
        }
    }

    pub fn set_multipv(&mut self, multi_pv: usize) {
        self.multi_pv = multi_pv.max(1);
    }

    pub fn resize_hash(&mut self, hash_mb: usize) {
        self.hash_mb = hash_mb.max(1).min(DEFAULT_TT_SIZE.max(1) * 16384);
        self.shared.tt.resize(self.threads.len(), self.hash_mb);
    }

    pub fn new_game(&mut self) {
        self.threads.clear();
        self.shared.tt.clear(self.threads.len());
    }

    pub fn search_depth(&mut self, depth: i32) -> SearchResult {
        self.search(Limits::Depth(depth.max(1)))
    }

    pub fn search_movetime(&mut self, ms: u64) -> SearchResult {
        self.search(Limits::Time(ms.max(1)))
    }

    fn search(&mut self, limits: Limits) -> SearchResult {
        let time_manager = TimeManager::new(limits, self.board.fullmove_number(), self.move_overhead_ms);
        self.threads.execute_searches(time_manager, Report::None, self.multi_pv, &self.board, &self.shared);
        let td = &self.threads[0];
        let elapsed = td.time_manager.elapsed();
        if td.root_moves.is_empty() {
            return SearchResult { bestmove: None, elapsed_ms: elapsed.as_millis(), lines: Vec::new() };
        }
        let nodes = self.shared.nodes.aggregate();
        let nps = nps(nodes, elapsed);
        let lines = td.root_moves.iter().take(td.multi_pv).enumerate().map(|(index, root_move)| {
            let mut pv = Vec::new();
            if root_move.mv != Move::NULL {
                pv.push(root_move.mv.to_uci(&self.board));
            }
            pv.extend(root_move.pv.line().iter().map(|mv| mv.to_uci(&self.board)));
            let (score_cp, mate_in) = score_fields(root_move.display_score, &self.board);
            SearchLine {
                multipv: index + 1,
                depth: td.completed_depth,
                score_cp,
                mate_in,
                nodes,
                nps,
                pv,
            }
        }).collect::<Vec<_>>();
        let bestmove = lines.first().and_then(|line| line.pv.first()).cloned();
        SearchResult { bestmove, elapsed_ms: elapsed.as_millis(), lines }
    }
}

fn nps(nodes: u64, elapsed: Duration) -> u64 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 { 0 } else { (nodes as f64 / seconds).round() as u64 }
}

fn score_fields(score: i32, board: &Board) -> (Option<i32>, Option<i32>) {
    match score.abs() {
        s if s < Score::TB_WIN_IN_MAX => (Some(normalize_to_cp(score, board)), None),
        s if s <= Score::TB_WIN => {
            let cp = 20_000 - Score::TB_WIN + score.abs();
            (Some(if score.is_positive() { cp } else { -cp }), None)
        }
        _ => {
            let mate = (Score::MATE - score.abs() + score.is_positive() as i32) / 2;
            (None, Some(if score.is_positive() { mate } else { -mate }))
        }
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\\\"");
    for ch in value.chars() {
        match ch {
            '\\\\' => out.push_str("\\\\\\\\"),
            '\"' => out.push_str("\\\\\\\""),
            '\\n' => out.push_str("\\\\n"),
            '\\r' => out.push_str("\\\\r"),
            '\\t' => out.push_str("\\\\t"),
            c if c.is_control() => out.push_str(&format!("\\\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('\"');
    out
}

impl SearchResult {
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\\\"bestmove\\\":");
        match &self.bestmove {
            Some(bestmove) => out.push_str(&json_string(bestmove)),
            None => out.push_str("null"),
        }
        out.push_str(&format!(",\\\"elapsedMs\\\":{},\\\"lines\\\":[", self.elapsed_ms));
        for (index, line) in self.lines.iter().enumerate() {
            if index > 0 { out.push(','); }
            out.push_str(&format!("{{\\\"multipv\\\":{},\\\"depth\\\":{}", line.multipv, line.depth));
            match line.score_cp {
                Some(score_cp) => out.push_str(&format!(",\\\"scoreCp\\\":{}", score_cp)),
                None => out.push_str(",\\\"scoreCp\\\":null"),
            }
            match line.mate_in {
                Some(mate_in) => out.push_str(&format!(",\\\"mateIn\\\":{}", mate_in)),
                None => out.push_str(",\\\"mateIn\\\":null"),
            }
            out.push_str(&format!(",\\\"nodes\\\":{},\\\"nps\\\":{},\\\"pv\\\":[", line.nodes, line.nps));
            for (pv_index, mv) in line.pv.iter().enumerate() {
                if pv_index > 0 { out.push(','); }
                out.push_str(&json_string(mv));
            }
            out.push_str("]}");
        }
        out.push_str("]}");
        out
    }
}
`);

mkdirSync(`${workdir}/src/bin`, { recursive: true });
writeFileSync(`${workdir}/src/bin/browser_api_probe.rs`, `use reckless::browser_api::BrowserEngine;

fn main() {
    let mut args = std::env::args().skip(1);
    let fen = args.next().unwrap_or_else(|| "startpos".to_string());
    let depth = args.next().and_then(|value| value.parse::<i32>().ok()).unwrap_or(7);
    let mut engine = BrowserEngine::new(16);
    engine.set_fen(&fen).expect("valid FEN");
    let result = engine.search_depth(depth);
    println!("{}", result.to_json());
}
`);

run('cargo', ['run', '--release', '--no-default-features', '--bin', 'browser_api_probe', '--', fen, String(depth)], {
  cwd: workdir,
  env: evalfile ? { ...process.env, EVALFILE: evalfile } : process.env,
});
