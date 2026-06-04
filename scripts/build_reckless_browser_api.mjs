#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const source = resolve(process.env.RECKLESS_SOURCE_DIR ?? '.local_engines/reckless-wasi-src');
const workdir = resolve(process.env.RECKLESS_BROWSER_API_BUILD_DIR ?? '.local_engines/reckless-browser-api-build');
const out = resolve(process.env.RECKLESS_BROWSER_API_WASM_OUT ?? 'public/reckless/reckless-browser-api.wasm');
const evalfile = process.env.RECKLESS_EVALFILE ? resolve(process.env.RECKLESS_EVALFILE) : '';

function run(cmd, args, options = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  execFileSync(cmd, args, { stdio: 'inherit', ...options });
}

function replace(path, oldText, newText) {
  const before = readFileSync(path, 'utf8');
  if (!before.includes(oldText)) throw new Error(`patch anchor not found in ${path}`);
  writeFileSync(path, before.replace(oldText, newText));
}

if (!existsSync(`${source}/Cargo.toml`)) {
  throw new Error(`Reckless source missing at ${source}; run npm run reckless:build-wasi first or set RECKLESS_SOURCE_DIR`);
}

rmSync(workdir, { recursive: true, force: true });
mkdirSync(dirname(workdir), { recursive: true });
run('rsync', ['-a', '--exclude', '.git', '--exclude', 'target', `${source}/`, `${workdir}/`]);

replace(
  `${workdir}/Cargo.toml`,
  `[features]\ndefault = ["syzygy"]\nsyzygy = []\nspsa = []\n`,
  `[features]\ndefault = ["syzygy"]\nsyzygy = []\nspsa = []\n\n[lib]\ncrate-type = ["cdylib", "rlib"]\n`,
);

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

writeFileSync(`${workdir}/src/browser_api.rs`, String.raw`use std::{sync::{Arc, Mutex, OnceLock}, time::Duration};

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

pub struct BrowserEngine {
    shared: Arc<SharedContext>,
    threads: ThreadPool,
    board: Board,
    hash_mb: usize,
    multi_pv: usize,
    move_overhead_ms: u64,
}

struct Handle {
    engine: BrowserEngine,
    result_json: Vec<u8>,
    error: Vec<u8>,
}

static HANDLES: OnceLock<Mutex<Vec<Option<Handle>>>> = OnceLock::new();
static INIT: OnceLock<()> = OnceLock::new();

impl BrowserEngine {
    pub fn new(hash_mb: usize) -> Self {
        INIT.get_or_init(|| {
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
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl SearchResult {
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\"bestmove\":");
        match &self.bestmove {
            Some(bestmove) => out.push_str(&json_string(bestmove)),
            None => out.push_str("null"),
        }
        out.push_str(&format!(",\"elapsedMs\":{},\"lines\":[", self.elapsed_ms));
        for (index, line) in self.lines.iter().enumerate() {
            if index > 0 { out.push(','); }
            out.push_str(&format!("{{\"multipv\":{},\"depth\":{}", line.multipv, line.depth));
            match line.score_cp {
                Some(score_cp) => out.push_str(&format!(",\"scoreCp\":{}", score_cp)),
                None => out.push_str(",\"scoreCp\":null"),
            }
            match line.mate_in {
                Some(mate_in) => out.push_str(&format!(",\"mateIn\":{}", mate_in)),
                None => out.push_str(",\"mateIn\":null"),
            }
            out.push_str(&format!(",\"nodes\":{},\"nps\":{},\"pv\":[", line.nodes, line.nps));
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

fn handles() -> &'static Mutex<Vec<Option<Handle>>> {
    HANDLES.get_or_init(|| Mutex::new(Vec::new()))
}

fn with_handle<R>(handle: u32, f: impl FnOnce(&mut Handle) -> R) -> Option<R> {
    if handle == 0 { return None; }
    let mut guard = handles().lock().ok()?;
    guard.get_mut(handle as usize - 1)?.as_mut().map(f)
}

fn set_error(handle: u32, error: impl AsRef<str>) -> i32 {
    let _ = with_handle(handle, |h| h.error = error.as_ref().as_bytes().to_vec());
    -1
}

unsafe fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).map_err(|error| error.to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn reckless_api_free_bytes(ptr: *mut u8, len: usize, capacity: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, len, capacity));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_new(hash_mb: usize) -> u32 {
    let mut guard = handles().lock().expect("browser API handle lock");
    let handle = Handle { engine: BrowserEngine::new(hash_mb), result_json: Vec::new(), error: Vec::new() };
    if let Some((index, slot)) = guard.iter_mut().enumerate().find(|(_, slot)| slot.is_none()) {
        *slot = Some(handle);
        index as u32 + 1
    } else {
        guard.push(Some(handle));
        guard.len() as u32
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_free(handle: u32) {
    if handle == 0 { return; }
    if let Ok(mut guard) = handles().lock() {
        if let Some(slot) = guard.get_mut(handle as usize - 1) {
            *slot = None;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn reckless_api_set_fen(handle: u32, ptr: *const u8, len: usize) -> i32 {
    let fen = match read_str(ptr, len) {
        Ok(fen) => fen,
        Err(error) => return set_error(handle, error),
    };
    with_handle(handle, |h| match h.engine.set_fen(fen) {
        Ok(()) => 0,
        Err(error) => set_error(handle, error),
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_set_multipv(handle: u32, multi_pv: usize) -> i32 {
    with_handle(handle, |h| {
        h.engine.set_multipv(multi_pv);
        0
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_resize_hash(handle: u32, hash_mb: usize) -> i32 {
    with_handle(handle, |h| {
        h.engine.resize_hash(hash_mb);
        0
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_new_game(handle: u32) -> i32 {
    with_handle(handle, |h| {
        h.engine.new_game();
        h.result_json.clear();
        0
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_search_depth(handle: u32, depth: i32) -> i32 {
    with_handle(handle, |h| {
        let result = h.engine.search_depth(depth);
        h.result_json = result.to_json().into_bytes();
        0
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_search_movetime(handle: u32, ms: u64) -> i32 {
    with_handle(handle, |h| {
        let result = h.engine.search_movetime(ms);
        h.result_json = result.to_json().into_bytes();
        0
    }).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_result_json_ptr(handle: u32) -> *const u8 {
    with_handle(handle, |h| h.result_json.as_ptr()).unwrap_or(std::ptr::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_result_json_len(handle: u32) -> usize {
    with_handle(handle, |h| h.result_json.len()).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_error_ptr(handle: u32) -> *const u8 {
    with_handle(handle, |h| h.error.as_ptr()).unwrap_or(std::ptr::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_error_len(handle: u32) -> usize {
    with_handle(handle, |h| h.error.len()).unwrap_or(0)
}
`);

run('rustup', ['target', 'add', 'wasm32-wasip1']);
run('cargo', ['build', '--release', '--no-default-features', '--target', 'wasm32-wasip1', '--lib'], {
  cwd: workdir,
  env: evalfile ? { ...process.env, EVALFILE: evalfile } : process.env,
});

mkdirSync(dirname(out), { recursive: true });
const built = `${workdir}/target/wasm32-wasip1/release/reckless.wasm`;
if (!existsSync(built)) throw new Error(`expected build artifact missing: ${built}`);
cpSync(built, out);
console.log(`Wrote ${out}`);
