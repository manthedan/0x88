#![no_std]

mod policy_table;
use policy_table::{LC0_POLICY_SIZE, POLICY_FROM, POLICY_PROMO, POLICY_TO};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

const INPUT_CAP: usize = 256;
const MAX_LEGAL: usize = 256;
const EMPTY: u8 = 0;
const WP: u8 = 1; const WN: u8 = 2; const WB: u8 = 3; const WR: u8 = 4; const WQ: u8 = 5; const WK: u8 = 6;
const BP: u8 = 7; const BN: u8 = 8; const BB: u8 = 9; const BR: u8 = 10; const BQ: u8 = 11; const BK: u8 = 12;

static mut INPUT_BUF: [u8; INPUT_CAP] = [0; INPUT_CAP];
static mut LOGITS_BUF: [f32; LC0_POLICY_SIZE] = [0.0; LC0_POLICY_SIZE];
static mut OUT_INDICES: [u16; MAX_LEGAL] = [0; MAX_LEGAL];
static mut OUT_PRIORS: [f32; MAX_LEGAL] = [0.0; MAX_LEGAL];
static mut OUT_LOGITS: [f32; MAX_LEGAL] = [0.0; MAX_LEGAL];
static mut OUT_UCI: [u32; MAX_LEGAL] = [0; MAX_LEGAL];
static mut OUT_PROMO: [u8; MAX_LEGAL] = [0; MAX_LEGAL];
static mut OUT_COUNT: usize = 0;
static mut LAST_ERROR: i32 = 0;

#[derive(Clone, Copy)]
struct Board { squares: [u8; 64], turn: u8, castling: u8, ep: i16 }

#[derive(Clone, Copy)]
struct Move { from: u8, to: u8, promo: u8, en_passant: bool, castle: bool }

#[no_mangle]
pub extern "C" fn lc0_legal_input_buffer_ptr() -> *mut u8 { core::ptr::addr_of_mut!(INPUT_BUF) as *mut u8 }
#[no_mangle]
pub extern "C" fn lc0_legal_input_buffer_len() -> usize { INPUT_CAP }
#[no_mangle]
pub extern "C" fn lc0_legal_logits_buffer_ptr() -> *mut f32 { core::ptr::addr_of_mut!(LOGITS_BUF) as *mut f32 }
#[no_mangle]
pub extern "C" fn lc0_legal_logits_len() -> usize { LC0_POLICY_SIZE }
#[no_mangle]
pub extern "C" fn lc0_legal_indices_ptr() -> *const u16 { core::ptr::addr_of!(OUT_INDICES) as *const u16 }
#[no_mangle]
pub extern "C" fn lc0_legal_priors_ptr() -> *const f32 { core::ptr::addr_of!(OUT_PRIORS) as *const f32 }
#[no_mangle]
pub extern "C" fn lc0_legal_logits_out_ptr() -> *const f32 { core::ptr::addr_of!(OUT_LOGITS) as *const f32 }
#[no_mangle]
pub extern "C" fn lc0_legal_uci_ptr() -> *const u32 { core::ptr::addr_of!(OUT_UCI) as *const u32 }
#[no_mangle]
pub extern "C" fn lc0_legal_promo_ptr() -> *const u8 { core::ptr::addr_of!(OUT_PROMO) as *const u8 }
#[no_mangle]
pub extern "C" fn lc0_legal_count() -> usize { unsafe { OUT_COUNT } }
#[no_mangle]
pub extern "C" fn lc0_legal_last_error() -> i32 { unsafe { LAST_ERROR } }

fn set_error(code: i32) -> i32 { unsafe { LAST_ERROR = code; } code }
fn file(s: usize) -> i32 { (s & 7) as i32 }
fn rank(s: usize) -> i32 { (s >> 3) as i32 }
fn sq(file: i32, rank: i32) -> Option<usize> { if (0..8).contains(&file) && (0..8).contains(&rank) { Some((file + rank * 8) as usize) } else { None } }
fn color(piece: u8) -> u8 { if piece >= BP { 1 } else { 0 } }
fn is_own(piece: u8, side: u8) -> bool { piece != EMPTY && color(piece) == side }
fn is_enemy(piece: u8, side: u8) -> bool { piece != EMPTY && color(piece) != side }

fn piece_code(byte: u8) -> Option<u8> {
    Some(match byte {
        b'P' => WP, b'N' => WN, b'B' => WB, b'R' => WR, b'Q' => WQ, b'K' => WK,
        b'p' => BP, b'n' => BN, b'b' => BB, b'r' => BR, b'q' => BQ, b'k' => BK,
        _ => return None,
    })
}
fn is_space(byte: u8) -> bool { matches!(byte, b' ' | b'\t' | b'\r' | b'\n') }
fn skip_spaces(bytes: &[u8], mut i: usize) -> usize { while i < bytes.len() && is_space(bytes[i]) { i += 1; } i }
fn trim_spaces(bytes: &[u8]) -> &[u8] { let mut s=0; let mut e=bytes.len(); while s<e && is_space(bytes[s]){s+=1;} while e>s && is_space(bytes[e-1]){e-=1;} &bytes[s..e] }
fn ep_square_from_name(file: u8, rank: u8) -> i16 { if !(b'a'..=b'h').contains(&file) || !matches!(rank, b'3'|b'6') { -1 } else { ((file-b'a') as i16)+(((rank-b'1') as i16)*8) } }

fn parse_fen(input: &[u8]) -> Result<Board, i32> {
    let bytes = trim_spaces(input);
    let mut board = Board { squares: [EMPTY; 64], turn: 0, castling: 0, ep: -1 };
    let mut f = 0usize; let mut r = 7usize; let mut i = 0usize;
    while i < bytes.len() && !is_space(bytes[i]) {
        let c = bytes[i];
        if c == b'/' { if f != 8 || r == 0 { return Err(2); } f = 0; r -= 1; i += 1; continue; }
        if (b'1'..=b'8').contains(&c) { f += (c-b'0') as usize; if f > 8 { return Err(3); } i += 1; continue; }
        if let Some(p) = piece_code(c) { if f >= 8 { return Err(4); } board.squares[f + r*8] = p; f += 1; i += 1; continue; }
        return Err(5);
    }
    if r != 0 || f != 8 { return Err(6); }
    i = skip_spaces(bytes, i); if i >= bytes.len() { return Ok(board); }
    board.turn = match bytes[i] { b'w' => 0, b'b' => 1, _ => return Err(8) };
    i += 1; i = skip_spaces(bytes, i); if i >= bytes.len() { return Ok(board); }
    if bytes[i] == b'-' { i += 1; if i < bytes.len() && !is_space(bytes[i]) { return Err(10); } }
    else { while i < bytes.len() && !is_space(bytes[i]) { let bit = match bytes[i] { b'K'=>1,b'Q'=>2,b'k'=>4,b'q'=>8,_=>return Err(10) }; if board.castling & bit != 0 { return Err(10); } board.castling |= bit; i += 1; } }
    i = skip_spaces(bytes, i); if i >= bytes.len() { return Ok(board); }
    if bytes[i] == b'-' { board.ep = -1; i += 1; if i < bytes.len() && !is_space(bytes[i]) { return Err(14); } }
    else { if i+1 >= bytes.len() { return Err(12); } board.ep = ep_square_from_name(bytes[i], bytes[i+1]); if board.ep < 0 { return Err(13); } i += 2; if i < bytes.len() && !is_space(bytes[i]) { return Err(14); } }
    Ok(board)
}

fn attacked(board: &Board, target: usize, by_side: u8) -> bool {
    let tf = file(target); let tr = rank(target);
    let pawn_dir = if by_side == 0 { -1 } else { 1 };
    for df in [-1, 1] { if let Some(s)=sq(tf+df, tr+pawn_dir) { let p=board.squares[s]; if p == if by_side==0 { WP } else { BP } { return true; } } }
    for (df, dr) in [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)] { if let Some(s)=sq(tf+df,tr+dr){ let p=board.squares[s]; if p == if by_side==0 { WN } else { BN } { return true; } } }
    for (df, dr) in [(1,1),(1,-1),(-1,1),(-1,-1)] { let mut f=tf+df; let mut r=tr+dr; while let Some(s)=sq(f,r){ let p=board.squares[s]; if p!=EMPTY { if color(p)==by_side && matches!(p, WB|WQ|BB|BQ) { return true; } break; } f+=df; r+=dr; } }
    for (df, dr) in [(1,0),(-1,0),(0,1),(0,-1)] { let mut f=tf+df; let mut r=tr+dr; while let Some(s)=sq(f,r){ let p=board.squares[s]; if p!=EMPTY { if color(p)==by_side && matches!(p, WR|WQ|BR|BQ) { return true; } break; } f+=df; r+=dr; } }
    for df in -1..=1 { for dr in -1..=1 { if df==0 && dr==0 { continue; } if let Some(s)=sq(tf+df,tr+dr){ let p=board.squares[s]; if p == if by_side==0 { WK } else { BK } { return true; } } } }
    false
}
fn king_square(board: &Board, side: u8) -> Option<usize> { let k=if side==0{WK}else{BK}; for i in 0..64 { if board.squares[i]==k { return Some(i); } } None }
fn in_check(board: &Board, side: u8) -> bool { king_square(board, side).map(|k| attacked(board, k, 1-side)).unwrap_or(true) }

fn make_move(board: &Board, m: Move) -> Board {
    let mut b = *board;
    let side = board.turn; let piece = b.squares[m.from as usize];
    b.squares[m.from as usize] = EMPTY;
    if m.en_passant { let cap = if side==0 { (m.to as i32 - 8) as usize } else { (m.to as i32 + 8) as usize }; b.squares[cap]=EMPTY; }
    if m.castle {
        match (m.from, m.to) { (4,6)=>{ b.squares[7]=EMPTY; b.squares[5]=WR; }, (4,2)=>{ b.squares[0]=EMPTY; b.squares[3]=WR; }, (60,62)=>{ b.squares[63]=EMPTY; b.squares[61]=BR; }, (60,58)=>{ b.squares[56]=EMPTY; b.squares[59]=BR; }, _=>{} }
    }
    b.squares[m.to as usize] = if m.promo != 0 { match (side, m.promo) { (0,b'q')=>WQ,(0,b'r')=>WR,(0,b'b')=>WB,(0,b'n')=>WN,(1,b'q')=>BQ,(1,b'r')=>BR,(1,b'b')=>BB,(1,b'n')=>BN,_=>piece } } else { piece };
    b.turn = 1-side; b.ep = -1; b
}

fn push_if_legal(board: &Board, moves: &mut [Move; MAX_LEGAL], count: &mut usize, m: Move) {
    if *count >= MAX_LEGAL { return; }
    let next = make_move(board, m);
    if !in_check(&next, board.turn) { moves[*count] = m; *count += 1; }
}
fn push_pawn_move(board: &Board, moves: &mut [Move; MAX_LEGAL], count: &mut usize, from: usize, to: usize, capture_ep: bool) {
    let side = board.turn; let promotion_rank = if side==0 { 7 } else { 0 };
    if rank(to) == promotion_rank { for promo in [b'q', b'r', b'b', b'n'] { push_if_legal(board,moves,count,Move{from:from as u8,to:to as u8,promo,en_passant:capture_ep,castle:false}); } }
    else { push_if_legal(board,moves,count,Move{from:from as u8,to:to as u8,promo:0,en_passant:capture_ep,castle:false}); }
}

fn generate_legal(board: &Board, moves: &mut [Move; MAX_LEGAL]) -> usize {
    let mut count=0usize; let side=board.turn;
    for from in 0..64 { let piece=board.squares[from]; if !is_own(piece, side) { continue; } let f=file(from); let r=rank(from);
        match piece {
            WP|BP => { let dir=if side==0{1}else{-1}; let start_rank=if side==0{1}else{6};
                if let Some(to)=sq(f,r+dir) { if board.squares[to]==EMPTY { push_pawn_move(board,moves,&mut count,from,to,false); if r==start_rank { if let Some(to2)=sq(f,r+2*dir) { if board.squares[to2]==EMPTY { push_if_legal(board,moves,&mut count,Move{from:from as u8,to:to2 as u8,promo:0,en_passant:false,castle:false}); } } } } }
                for df in [-1,1] { if let Some(to)=sq(f+df,r+dir) { if is_enemy(board.squares[to], side) { push_pawn_move(board,moves,&mut count,from,to,false); } else if board.ep == to as i16 { push_pawn_move(board,moves,&mut count,from,to,true); } } }
            }
            WN|BN => for (df,dr) in [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)] { if let Some(to)=sq(f+df,r+dr) { if !is_own(board.squares[to], side) { push_if_legal(board,moves,&mut count,Move{from:from as u8,to:to as u8,promo:0,en_passant:false,castle:false}); } } },
            WB|BB|WR|BR|WQ|BQ => { let dirs: &[(i32,i32)] = match piece { WB|BB=>&[(1,1),(1,-1),(-1,1),(-1,-1)], WR|BR=>&[(1,0),(-1,0),(0,1),(0,-1)], _=>&[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)] }; for (df,dr) in dirs { let mut nf=f+df; let mut nr=r+dr; while let Some(to)=sq(nf,nr) { if is_own(board.squares[to], side) { break; } push_if_legal(board,moves,&mut count,Move{from:from as u8,to:to as u8,promo:0,en_passant:false,castle:false}); if is_enemy(board.squares[to], side) { break; } nf+=df; nr+=dr; } } }
            WK|BK => { for df in -1..=1 { for dr in -1..=1 { if df==0&&dr==0{continue;} if let Some(to)=sq(f+df,r+dr){ if !is_own(board.squares[to],side){ push_if_legal(board,moves,&mut count,Move{from:from as u8,to:to as u8,promo:0,en_passant:false,castle:false}); } } } }
                if !in_check(board, side) { if side==0 {
                    if board.castling&1!=0 && board.squares[4]==WK && board.squares[7]==WR && board.squares[5]==EMPTY && board.squares[6]==EMPTY && !attacked(board,5,1)&&!attacked(board,6,1){ push_if_legal(board,moves,&mut count,Move{from:4,to:6,promo:0,en_passant:false,castle:true}); }
                    if board.castling&2!=0 && board.squares[4]==WK && board.squares[0]==WR && board.squares[1]==EMPTY && board.squares[2]==EMPTY && board.squares[3]==EMPTY && !attacked(board,3,1)&&!attacked(board,2,1){ push_if_legal(board,moves,&mut count,Move{from:4,to:2,promo:0,en_passant:false,castle:true}); }
                  } else {
                    if board.castling&4!=0 && board.squares[60]==BK && board.squares[63]==BR && board.squares[61]==EMPTY && board.squares[62]==EMPTY && !attacked(board,61,0)&&!attacked(board,62,0){ push_if_legal(board,moves,&mut count,Move{from:60,to:62,promo:0,en_passant:false,castle:true}); }
                    if board.castling&8!=0 && board.squares[60]==BK && board.squares[56]==BR && board.squares[57]==EMPTY && board.squares[58]==EMPTY && board.squares[59]==EMPTY && !attacked(board,59,0)&&!attacked(board,58,0){ push_if_legal(board,moves,&mut count,Move{from:60,to:58,promo:0,en_passant:false,castle:true}); }
                  } }
            }
            _=>{}
        }
    }
    count
}

fn mirror_rank(square: u8) -> u8 { (square & 7) + (7 - (square >> 3)) * 8 }
fn policy_index_for(m: Move, side: u8) -> Option<usize> {
    let mut from=m.from; let mut to=m.to; let mut promo=m.promo;
    if m.castle { to = match (from,to) { (4,6)=>7,(4,2)=>0,(60,62)=>63,(60,58)=>56,_=>to }; }
    if promo == b'n' { promo = 0; }
    if side == 1 { from=mirror_rank(from); to=mirror_rank(to); }
    for i in 0..LC0_POLICY_SIZE { if POLICY_FROM[i]==from && POLICY_TO[i]==to && POLICY_PROMO[i]==promo { return Some(i); } }
    None
}
fn pack_uci(m: Move) -> u32 {
    let from=m.from; let to=m.to;
    let b0=b'a'+(from&7); let b1=b'1'+(from>>3); let b2=b'a'+(to&7); let b3=b'1'+(to>>3); let b4=m.promo;
    let _ = b4;
    (b0 as u32) | ((b1 as u32)<<8) | ((b2 as u32)<<16) | ((b3 as u32)<<24)
}

#[no_mangle]
pub extern "C" fn lc0_legal_priors_from_fen(len: usize, temperature: f32, top_k: usize) -> i32 {
    unsafe { OUT_COUNT = 0; }
    if len > INPUT_CAP { return set_error(1); }
    let board = match parse_fen(unsafe { &INPUT_BUF[..len] }) { Ok(b)=>b, Err(c)=>return set_error(c) };
    let mut moves=[Move{from:0,to:0,promo:0,en_passant:false,castle:false}; MAX_LEGAL];
    let count=generate_legal(&board, &mut moves);
    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    let mut idxs=[0usize; MAX_LEGAL]; let mut logits=[0.0f32; MAX_LEGAL]; let mut ucis=[0u32; MAX_LEGAL]; let mut promos=[0u8; MAX_LEGAL]; let mut n=0usize;
    for i in 0..count { if let Some(idx)=policy_index_for(moves[i], board.turn) { idxs[n]=idx; logits[n]=unsafe{LOGITS_BUF[idx]}/temp; ucis[n]=pack_uci(moves[i]); promos[n]=moves[i].promo; n+=1; } }
    if n==0 { unsafe { OUT_COUNT=0; } return set_error(0); }
    let mut max=logits[0]; for i in 1..n { if logits[i] > max { max = logits[i]; } }
    let mut sum=0.0f32; let mut priors=[0.0f32; MAX_LEGAL]; for i in 0..n { let p=libm::expf(logits[i]-max); priors[i]=p; sum+=p; }
    for i in 0..n { priors[i] /= sum; }
    for i in 1..n { let mut j=i; while j>0 && priors[j] > priors[j-1] { priors.swap(j,j-1); logits.swap(j,j-1); idxs.swap(j,j-1); ucis.swap(j,j-1); promos.swap(j,j-1); j-=1; } }
    let out_n = if top_k > 0 && top_k < n { top_k } else { n };
    unsafe { for i in 0..out_n { OUT_INDICES[i]=idxs[i] as u16; OUT_PRIORS[i]=priors[i]; OUT_LOGITS[i]=logits[i]; OUT_UCI[i]=ucis[i]; OUT_PROMO[i]=promos[i]; } OUT_COUNT=out_n; }
    set_error(0)
}
