#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

const INPUT_CAP: usize = 32768;
const PLANES: usize = 112;
const SQUARES: usize = 64;
const HISTORY: usize = 8;
const PLANES_PER_HISTORY: usize = 13;
const AUX_BASE: usize = HISTORY * PLANES_PER_HISTORY;

static mut INPUT_BUF: [u8; INPUT_CAP] = [0; INPUT_CAP];
static mut PLANES_BUF: [f32; PLANES * SQUARES] = [0.0; PLANES * SQUARES];
static mut MASKS_BUF: [u64; PLANES] = [0; PLANES];
static mut VALUES_BUF: [f64; PLANES] = [1.0; PLANES];
static mut LAST_ERROR: i32 = 0;

#[derive(Clone, Copy)]
struct Board {
    squares: [u8; 64],
    turn: u8,
    castling: u8,
    ep: i16,
    halfmove: f64,
    fullmove: u64,
}

const EMPTY: u8 = 0;
const WP: u8 = 1; const WN: u8 = 2; const WB: u8 = 3; const WR: u8 = 4; const WQ: u8 = 5; const WK: u8 = 6;
const BP: u8 = 7; const BN: u8 = 8; const BB: u8 = 9; const BR: u8 = 10; const BQ: u8 = 11; const BK: u8 = 12;

#[no_mangle]
pub extern "C" fn lc0_input_buffer_ptr() -> *mut u8 { core::ptr::addr_of_mut!(INPUT_BUF) as *mut u8 }

#[no_mangle]
pub extern "C" fn lc0_input_buffer_len() -> usize { INPUT_CAP }

#[no_mangle]
pub extern "C" fn lc0_planes_buffer_ptr() -> *const f32 { core::ptr::addr_of!(PLANES_BUF) as *const f32 }

#[no_mangle]
pub extern "C" fn lc0_planes_len() -> usize { PLANES * SQUARES }

#[no_mangle]
pub extern "C" fn lc0_masks_buffer_ptr() -> *const u64 { core::ptr::addr_of!(MASKS_BUF) as *const u64 }

#[no_mangle]
pub extern "C" fn lc0_values_buffer_ptr() -> *const f64 { core::ptr::addr_of!(VALUES_BUF) as *const f64 }

#[no_mangle]
pub extern "C" fn lc0_last_error() -> i32 { unsafe { LAST_ERROR } }

fn set_error(code: i32) -> i32 { unsafe { LAST_ERROR = code; } code }

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn zero_planes() {
    use core::arch::wasm32::{f32x4_splat, v128_store};
    let zero = f32x4_splat(0.0);
    let ptr = core::ptr::addr_of_mut!(PLANES_BUF) as *mut f32;
    let mut i = 0usize;
    while i < PLANES * SQUARES {
        v128_store(ptr.add(i) as *mut core::arch::wasm32::v128, zero);
        i += 4;
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn zero_planes() {
    let ptr = core::ptr::addr_of_mut!(PLANES_BUF) as *mut f32;
    for i in 0..PLANES * SQUARES { *ptr.add(i) = 0.0; }
}

unsafe fn reset_output_buffers() {
    zero_planes();
    let masks = core::ptr::addr_of_mut!(MASKS_BUF) as *mut u64;
    let values = core::ptr::addr_of_mut!(VALUES_BUF) as *mut f64;
    for i in 0..PLANES {
        *masks.add(i) = 0;
        *values.add(i) = 1.0;
    }
}

fn piece_code(byte: u8) -> Option<u8> {
    Some(match byte {
        b'P' => WP, b'N' => WN, b'B' => WB, b'R' => WR, b'Q' => WQ, b'K' => WK,
        b'p' => BP, b'n' => BN, b'b' => BB, b'r' => BR, b'q' => BQ, b'k' => BK,
        _ => return None,
    })
}

fn role_offset(piece: u8) -> usize {
    match piece {
        WP | BP => 0,
        WN | BN => 1,
        WB | BB => 2,
        WR | BR => 3,
        WQ | BQ => 4,
        WK | BK => 5,
        _ => 0,
    }
}

fn color(piece: u8) -> u8 { if piece >= BP { 1 } else { 0 } }
fn file_of(sq: usize) -> usize { sq & 7 }
fn rank_of(sq: usize) -> usize { sq >> 3 }
fn mirror_rank(sq: usize) -> usize { file_of(sq) + (7 - rank_of(sq)) * 8 }
fn perspective_square(sq: usize, turn: u8) -> usize { if turn == 0 { sq } else { mirror_rank(sq) } }

const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

fn parse_uint(bytes: &[u8], mut i: usize) -> (u64, usize, bool) {
    let mut out = 0u64;
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        let digit = (bytes[i] - b'0') as u64;
        if out > (JS_MAX_SAFE_INTEGER - digit) / 10 { return (0, i, false); }
        out = out * 10 + digit;
        i += 1;
    }
    (out, i, i != start)
}

fn is_space(byte: u8) -> bool { matches!(byte, b' ' | b'\t' | b'\r' | b'\n') }

fn trim_spaces(bytes: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && is_space(bytes[start]) { start += 1; }
    while end > start && is_space(bytes[end - 1]) { end -= 1; }
    &bytes[start..end]
}

fn skip_spaces(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && is_space(bytes[i]) { i += 1; }
    i
}

fn ep_square_from_name(file: u8, rank: u8) -> i16 {
    if !(b'a'..=b'h').contains(&file) || !matches!(rank, b'3' | b'6') { return -1; }
    ((file - b'a') as i16) + (((rank - b'1') as i16) * 8)
}

fn parse_fen(input: &[u8]) -> Result<Board, i32> {
    let bytes = trim_spaces(input);
    let mut board = Board { squares: [EMPTY; 64], turn: 0, castling: 0, ep: -1, halfmove: 0.0, fullmove: 1 };
    let mut file = 0usize;
    let mut rank = 7usize;
    let mut i = 0usize;
    while i < bytes.len() && !is_space(bytes[i]) {
        let c = bytes[i];
        if c == b'/' {
            if file != 8 || rank == 0 { return Err(2); }
            file = 0; rank -= 1; i += 1; continue;
        }
        if (b'1'..=b'8').contains(&c) {
            file += (c - b'0') as usize;
            if file > 8 { return Err(3); }
            i += 1; continue;
        }
        if let Some(piece) = piece_code(c) {
            if file >= 8 { return Err(4); }
            board.squares[file + rank * 8] = piece;
            file += 1; i += 1; continue;
        }
        return Err(5);
    }
    if rank != 0 || file != 8 { return Err(6); }
    i = skip_spaces(bytes, i);
    if i >= bytes.len() { return Ok(board); }
    board.turn = match bytes[i] { b'w' => 0, b'b' => 1, _ => return Err(8) };
    i += 1; i = skip_spaces(bytes, i);
    if i >= bytes.len() { return Ok(board); }
    if bytes[i] == b'-' {
        i += 1;
        if i < bytes.len() && !is_space(bytes[i]) { return Err(10); }
    } else {
        while i < bytes.len() && !is_space(bytes[i]) {
            let bit = match bytes[i] { b'K' => 1, b'Q' => 2, b'k' => 4, b'q' => 8, _ => return Err(10) };
            if board.castling & bit != 0 { return Err(10); }
            board.castling |= bit;
            i += 1;
        }
    }
    i = skip_spaces(bytes, i);
    if i >= bytes.len() { return Ok(board); }
    if bytes[i] == b'-' {
        board.ep = -1;
        i += 1;
        if i < bytes.len() && !is_space(bytes[i]) { return Err(14); }
    }
    else {
        if i + 1 >= bytes.len() { return Err(12); }
        board.ep = ep_square_from_name(bytes[i], bytes[i + 1]);
        if board.ep < 0 { return Err(13); }
        i += 2;
        if i < bytes.len() && !is_space(bytes[i]) { return Err(14); }
    }
    i = skip_spaces(bytes, i);
    if i < bytes.len() {
        let (half, ni, ok) = parse_uint(bytes, i);
        if !ok { return Err(15); }
        if ni < bytes.len() && !is_space(bytes[ni]) { return Err(15); }
        board.halfmove = half as f64;
        i = skip_spaces(bytes, ni);
        if i < bytes.len() {
            let (full, ni, ok) = parse_uint(bytes, i);
            if !ok || full == 0 { return Err(16); }
            if ni < bytes.len() && !is_space(bytes[ni]) { return Err(16); }
            board.fullmove = full;
            i = skip_spaces(bytes, ni);
            if i < bytes.len() { return Err(17); }
        }
    }
    Ok(board)
}

fn is_start(board: &Board) -> bool {
    if board.turn != 0 || board.castling != 0b1111 || board.ep != -1 || board.halfmove != 0.0 || board.fullmove != 1 { return false; }
    let start = b"RNBQKBNRPPPPPPPP                                pppppppprnbqkbnr";
    for sq in 0..64 {
        let expected = match start[sq] {
            b'P' => WP, b'N' => WN, b'B' => WB, b'R' => WR, b'Q' => WQ, b'K' => WK,
            b'p' => BP, b'n' => BN, b'b' => BB, b'r' => BR, b'q' => BQ, b'k' => BK,
            _ => EMPTY,
        };
        if board.squares[sq] != expected { return false; }
    }
    true
}

fn board_before_ep_double_push(board: Board) -> Board {
    if board.ep < 0 { return board; }
    let moved = 1 - board.turn;
    let ep = board.ep as usize;
    let f = file_of(ep);
    let r = rank_of(ep);
    let current_rank = if moved == 0 { r + 1 } else { r.wrapping_sub(1) };
    let previous_rank = if moved == 0 { r.wrapping_sub(1) } else { r + 1 };
    if current_rank >= 8 || previous_rank >= 8 { return board; }
    let current = f + current_rank * 8;
    let previous = f + previous_rank * 8;
    let expected = if moved == 0 { WP } else { BP };
    if board.squares[current] != expected { return board; }
    let mut out = board;
    out.squares[current] = EMPTY;
    out.squares[previous] = expected;
    out.ep = -1;
    out
}

unsafe fn set_plane(plane: usize, sq: usize, value: f32) {
    let ptr = core::ptr::addr_of_mut!(PLANES_BUF) as *mut f32;
    let masks = core::ptr::addr_of_mut!(MASKS_BUF) as *mut u64;
    let values = core::ptr::addr_of_mut!(VALUES_BUF) as *mut f64;
    *ptr.add(plane * SQUARES + sq) = value;
    *masks.add(plane) |= 1u64 << sq;
    *values.add(plane) = value as f64;
}

fn set_all(plane: usize, value: f64) {
    unsafe {
        let ptr = core::ptr::addr_of_mut!(PLANES_BUF) as *mut f32;
        let masks = core::ptr::addr_of_mut!(MASKS_BUF) as *mut u64;
        let values = core::ptr::addr_of_mut!(VALUES_BUF) as *mut f64;
        *masks.add(plane) = u64::MAX;
        *values.add(plane) = value;
        for sq in 0..SQUARES { *ptr.add(plane * SQUARES + sq) = value as f32; }
    }
}

fn encode_board_pieces(board: &Board, history_slot: usize, perspective_turn: u8) {
    let base = history_slot * PLANES_PER_HISTORY;
    for source in 0..64 {
        let piece = board.squares[source];
        if piece == EMPTY { continue; }
        let side_offset = if color(piece) == perspective_turn { 0 } else { 6 };
        let plane = base + side_offset + role_offset(piece);
        let target = perspective_square(source, perspective_turn);
        unsafe { set_plane(plane, target, 1.0); }
    }
}

fn encode_aux(board: &Board) {
    let us = board.turn;
    let them = 1 - us;
    let our_q = if us == 0 { 2 } else { 8 };
    let our_k = if us == 0 { 1 } else { 4 };
    let their_q = if them == 0 { 2 } else { 8 };
    let their_k = if them == 0 { 1 } else { 4 };
    if board.castling & our_q != 0 { set_all(AUX_BASE, 1.0); }
    if board.castling & our_k != 0 { set_all(AUX_BASE + 1, 1.0); }
    if board.castling & their_q != 0 { set_all(AUX_BASE + 2, 1.0); }
    if board.castling & their_k != 0 { set_all(AUX_BASE + 3, 1.0); }
    if board.turn == 1 { set_all(AUX_BASE + 4, 1.0); }
    set_all(AUX_BASE + 5, board.halfmove);
    set_all(AUX_BASE + 7, 1.0);
}

fn encode_history(boards: &[Board], fen_only_fill: bool) -> Result<(), i32> {
    if boards.is_empty() { return Err(20); }
    unsafe { reset_output_buffers(); }
    let current = boards[boards.len() - 1];
    let mut history: [Board; HISTORY] = [current; HISTORY];
    let mut count = 0usize;
    for i in 0..boards.len().min(HISTORY) {
        history[i] = boards[boards.len() - 1 - i];
        count += 1;
    }
    if fen_only_fill && boards.len() == 1 && !is_start(&current) {
        let synthetic = board_before_ep_double_push(current);
        while count < HISTORY { history[count] = synthetic; count += 1; }
    }
    for i in 0..count.min(HISTORY) { encode_board_pieces(&history[i], i, current.turn); }
    encode_aux(&current);
    Ok(())
}

#[no_mangle]
pub extern "C" fn lc0_encode_fen(len: usize, history_fill: i32) -> i32 {
    if len > INPUT_CAP { return set_error(1); }
    let bytes = unsafe { &INPUT_BUF[..len] };
    match parse_fen(bytes).and_then(|board| encode_history(&[board; 1], history_fill != 0)) {
        Ok(()) => set_error(0),
        Err(code) => set_error(code),
    }
}

#[no_mangle]
pub extern "C" fn lc0_encode_fen_history(len: usize) -> i32 {
    if len > INPUT_CAP { return set_error(1); }
    let bytes = unsafe { &INPUT_BUF[..len] };
    let mut boards: [Board; HISTORY] = [Board { squares: [EMPTY;64], turn: 0, castling: 0, ep: -1, halfmove: 0.0, fullmove: 1 }; HISTORY];
    let mut count = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;
    while i <= bytes.len() {
        if i == bytes.len() || bytes[i] == b'\n' {
            if i > start {
                match parse_fen(&bytes[start..i]) {
                    Ok(board) => {
                        if count < HISTORY {
                            boards[count] = board;
                            count += 1;
                        } else {
                            let mut j = 1usize;
                            while j < HISTORY { boards[j - 1] = boards[j]; j += 1; }
                            boards[HISTORY - 1] = board;
                        }
                    }
                    Err(code) => return set_error(code),
                }
            }
            start = i + 1;
        }
        i += 1;
    }
    match encode_history(&boards[..count], false) {
        Ok(()) => set_error(0),
        Err(code) => set_error(code),
    }
}
