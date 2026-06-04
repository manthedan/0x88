#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const source = resolve(process.env.RECKLESS_SOURCE_DIR ?? '.local_engines/reckless-wasi-src');
const workdir = resolve(process.env.RECKLESS_BROWSER_API_BUILD_DIR ?? '.local_engines/reckless-browser-api-build');
const out = resolve(process.env.RECKLESS_BROWSER_API_WASM_OUT ?? 'public/reckless/reckless-browser-api.wasm');
const evalfile = process.env.RECKLESS_EVALFILE ? resolve(process.env.RECKLESS_EVALFILE) : '';
const enableWasmSimdNnue = process.env.RECKLESS_WASM_SIMD_NNUE === '1';
const externalNnue = process.env.RECKLESS_BROWSER_API_EXTERNAL_NNUE === '1';

function run(cmd, args, options = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  execFileSync(cmd, args, { stdio: 'inherit', ...options });
}

function replace(path, oldText, newText) {
  const before = readFileSync(path, 'utf8');
  if (!before.includes(oldText)) throw new Error(`patch anchor not found in ${path}`);
  writeFileSync(path, before.replace(oldText, newText));
}

function replaceIfPresent(path, oldText, newText) {
  const before = readFileSync(path, 'utf8');
  if (before.includes(newText)) return;
  if (!before.includes(oldText)) throw new Error(`patch anchor not found in ${path}`);
  writeFileSync(path, before.replace(oldText, newText));
}

function patchRecklessForWasmSimdNnue(root) {
  replaceIfPresent(
    `${root}/src/nnue.rs`,
    `mod forward {\n    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]\n    mod vectorized;\n    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]\n    pub use vectorized::*;\n\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]\n    pub use scalar::*;\n}\n`,
    `mod forward {\n    #[cfg(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128")))]\n    mod vectorized;\n    #[cfg(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128")))]\n    pub use vectorized::*;\n\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    pub use scalar::*;\n}\n`,
  );
  replaceIfPresent(
    `${root}/src/nnue.rs`,
    `    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    mod neon;\n    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    pub use neon::*;\n\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]\n    pub use scalar::*;\n}\n`,
    `    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    mod neon;\n    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    pub use neon::*;\n\n    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]\n    mod wasm32;\n    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]\n    pub use wasm32::*;\n\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    pub use scalar::*;\n}\n`,
  );
  replaceIfPresent(
    `${root}/src/nnue/forward/vectorized.rs`,
    `#[cfg(all(not(target_feature = "neon"), not(target_feature = "avx512vbmi2")))]\npub unsafe fn find_nnz(\n`,
    `#[cfg(all(not(target_arch = "wasm32"), not(target_feature = "neon"), not(target_feature = "avx512vbmi2")))]\npub unsafe fn find_nnz(\n`,
  );
  writeFileSync(`${root}/src/nnue/simd/wasm32.rs`, String.raw`use std::{arch::wasm32::*, mem::size_of};

pub const F32_LANES: usize = size_of::<v128>() / size_of::<f32>();
pub const I32_LANES: usize = size_of::<v128>() / size_of::<i32>();
pub const I16_LANES: usize = size_of::<v128>() / size_of::<i16>();
pub const MUL_HI_SHIFT: i32 = 0;

pub fn add_i16(a: v128, b: v128) -> v128 { i16x8_add(a, b) }
pub fn sub_i16(a: v128, b: v128) -> v128 { i16x8_sub(a, b) }

pub unsafe fn zeroed() -> v128 { i32x4_splat(0) }
pub unsafe fn splat_i16(a: i16) -> v128 { i16x8_splat(a) }
pub unsafe fn clamp_i16(x: v128, min: v128, max: v128) -> v128 { i16x8_max(i16x8_min(x, max), min) }
pub unsafe fn min_i16(a: v128, b: v128) -> v128 { i16x8_min(a, b) }
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: v128) -> v128 { i16x8_shl(a, SHIFT as u32) }

pub unsafe fn mul_high_i16(a: v128, b: v128) -> v128 {
    let lo = i32x4_shr(i32x4_extmul_low_i16x8(a, b), 16);
    let hi = i32x4_shr(i32x4_extmul_high_i16x8(a, b), 16);
    i16x8_narrow_i32x4(lo, hi)
}

pub unsafe fn convert_i8_i16(a: i64) -> v128 { i16x8_extend_low_i8x16(i64x2(a, 0)) }
pub unsafe fn packus(a: v128, b: v128) -> v128 { u8x16_narrow_i16x8(a, b) }
pub unsafe fn permute(a: v128) -> v128 { a }
pub unsafe fn splat_i32(a: i32) -> v128 { i32x4_splat(a) }
pub unsafe fn zero_f32() -> v128 { f32x4_splat(0.0) }
pub unsafe fn splat_f32(a: f32) -> v128 { f32x4_splat(a) }
pub unsafe fn mul_add_f32(a: v128, b: v128, c: v128) -> v128 { f32x4_add(f32x4_mul(a, b), c) }
pub unsafe fn convert_to_f32(a: v128) -> v128 { f32x4_convert_i32x4(a) }
pub unsafe fn clamp_f32(x: v128, min: v128, max: v128) -> v128 { f32x4_max(f32x4_min(x, max), min) }

unsafe fn dpbusd_once(i32s: v128, u8s: v128, i8s: v128) -> v128 {
    let prod_lo = i16x8_mul(u16x8_extend_low_u8x16(u8s), i16x8_extend_low_i8x16(i8s));
    let prod_hi = i16x8_mul(u16x8_extend_high_u8x16(u8s), i16x8_extend_high_i8x16(i8s));
    let pair_lo = i32x4_extadd_pairwise_i16x8(prod_lo);
    let pair_hi = i32x4_extadd_pairwise_i16x8(prod_hi);
    let sums = i32x4(
        i32x4_extract_lane::<0>(pair_lo) + i32x4_extract_lane::<1>(pair_lo),
        i32x4_extract_lane::<2>(pair_lo) + i32x4_extract_lane::<3>(pair_lo),
        i32x4_extract_lane::<0>(pair_hi) + i32x4_extract_lane::<1>(pair_hi),
        i32x4_extract_lane::<2>(pair_hi) + i32x4_extract_lane::<3>(pair_hi),
    );
    i32x4_add(i32s, sums)
}

pub unsafe fn dpbusd(i32s: v128, u8s: v128, i8s: v128) -> v128 { dpbusd_once(i32s, u8s, i8s) }
pub unsafe fn double_dpbusd(i32s: v128, u8s1: v128, i8s1: v128, u8s2: v128, i8s2: v128) -> v128 {
    dpbusd_once(dpbusd_once(i32s, u8s1, i8s1), u8s2, i8s2)
}

pub unsafe fn horizontal_sum(x: [v128; 4]) -> f32 {
    let mut sum = 0.0;
    for vector in x {
        sum += f32x4_extract_lane::<0>(vector);
        sum += f32x4_extract_lane::<1>(vector);
        sum += f32x4_extract_lane::<2>(vector);
        sum += f32x4_extract_lane::<3>(vector);
    }
    sum
}

pub unsafe fn nnz_bitmask(x: v128) -> u16 { u8x16_bitmask(i32x4_gt(x, i32x4_splat(0))) }
`);
  const vectorizedPath = `${root}/src/nnue/forward/vectorized.rs`;
  const vectorized = readFileSync(vectorizedPath, 'utf8');
  if (!vectorized.includes('target_arch = "wasm32", target_feature = "simd128"))]\npub unsafe fn find_nnz')) {
    writeFileSync(vectorizedPath, `${vectorized}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, _: &[SparseEntry]) -> (Aligned<[u16; L1_SIZE / 4]>, usize) {
    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;
    for i in 0..L1_SIZE / 4 {
        let mut nonzero = 0;
        for j in 0..4 {
            nonzero |= ft_out[i * 4 + j];
        }
        if nonzero != 0 {
            indexes[count] = i as u16;
            count += 1;
        }
    }
    (indexes, count)
}
`);
  }
}

function patchRecklessForBrowserApiNnue(root) {
  replace(
    `${root}/src/nnue.rs`,
    `use std::sync::Arc;`,
    `use std::sync::{Arc, OnceLock};`,
  );
  replace(
    `${root}/src/nnue.rs`,
    `const DEQUANT_MULTIPLIER: f32 = (1 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT * L1_QUANT) as f32;
`,
    `const DEQUANT_MULTIPLIER: f32 = (1 << FT_SHIFT) as f32 / (FT_QUANT * FT_QUANT * L1_QUANT) as f32;

static EXTERNAL_PARAMETERS: OnceLock<Arc<Parameters>> = OnceLock::new();
`,
  );
  const embeddedBody = externalNnue
    ? `        panic!("external NNUE parameters were not loaded");`
    : `        static EMBEDDED: Parameters = unsafe { std::mem::transmute(*include_bytes!(env!("MODEL"))) };
        &EMBEDDED`;
  replace(
    `${root}/src/nnue.rs`,
    `impl Parameters {
    fn embedded() -> &'static Self {
        static EMBEDDED: Parameters = unsafe { std::mem::transmute(*include_bytes!(env!("MODEL"))) };
        &EMBEDDED
    }

    fn allocate_owned() -> Arc<Self> {
        let mut boxed = Box::<std::mem::MaybeUninit<Self>>::new(std::mem::MaybeUninit::uninit());
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);

        unsafe {
            std::ptr::copy_nonoverlapping(Self::embedded() as *const Self, ptr, 1);
            Arc::from(Box::from_raw(ptr))
        }
    }
}
`,
    `impl Parameters {
    fn embedded() -> &'static Self {
${embeddedBody}
    }

    fn active() -> &'static Self {
        EXTERNAL_PARAMETERS.get().map(|parameters| parameters.as_ref()).unwrap_or_else(Self::embedded)
    }

    fn from_bytes_owned(bytes: &[u8]) -> Result<Arc<Self>, String> {
        let expected = std::mem::size_of::<Self>();
        if bytes.len() != expected {
            return Err(format!("invalid Reckless NNUE byte length: got {}, expected {}", bytes.len(), expected));
        }
        let boxed = Box::<std::mem::MaybeUninit<Self>>::new(std::mem::MaybeUninit::uninit());
        let ptr = Box::into_raw(boxed) as *mut Self;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            Ok(Arc::from(Box::from_raw(ptr)))
        }
    }

    fn allocate_owned() -> Arc<Self> {
        let mut boxed = Box::<std::mem::MaybeUninit<Self>>::new(std::mem::MaybeUninit::uninit());
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);

        unsafe {
            std::ptr::copy_nonoverlapping(Self::active() as *const Self, ptr, 1);
            Arc::from(Box::from_raw(ptr))
        }
    }
}

pub fn set_external_parameters_from_bytes(bytes: &[u8]) -> Result<(), String> {
    let parameters = Parameters::from_bytes_owned(bytes)?;
    EXTERNAL_PARAMETERS.set(parameters).map_err(|_| "external Reckless NNUE parameters are already loaded".to_owned())
}
`,
  );
  replace(
    `${root}/src/nnue.rs`,
    `        Self { inner: ParametersStorage::Embedded(Parameters::embedded()) }`,
    `        Self { inner: ParametersStorage::Embedded(Parameters::active()) }`,
  );
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
    types::{Color, Move, Score, normalize_to_cp},
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
static GLOBAL_ERROR: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

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

        for corrhist in self.shared.history.all() {
            corrhist.pawn.clear();
            corrhist.non_pawn[Color::White].clear();
            corrhist.non_pawn[Color::Black].clear();
        }
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

fn global_error() -> &'static Mutex<Vec<u8>> {
    GLOBAL_ERROR.get_or_init(|| Mutex::new(Vec::new()))
}

fn set_global_error(error: impl AsRef<str>) {
    if let Ok(mut global) = global_error().lock() {
        *global = error.as_ref().as_bytes().to_vec();
    }
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
pub unsafe extern "C" fn reckless_api_new_with_network(hash_mb: usize, ptr: *const u8, len: usize) -> u32 {
    let bytes = std::slice::from_raw_parts(ptr, len);
    if let Err(error) = crate::nnue::set_external_parameters_from_bytes(bytes) {
        set_global_error(error);
        return 0;
    }
    reckless_api_new(hash_mb)
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_global_error_ptr() -> *const u8 {
    global_error().lock().map(|error| error.as_ptr()).unwrap_or(std::ptr::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn reckless_api_global_error_len() -> usize {
    global_error().lock().map(|error| error.len()).unwrap_or(0)
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

if (enableWasmSimdNnue) patchRecklessForWasmSimdNnue(workdir);
patchRecklessForBrowserApiNnue(workdir);

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
