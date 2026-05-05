use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color { White, Black }

impl Color {
    pub fn opposite(self) -> Self { match self { Color::White => Color::Black, Color::Black => Color::White } }
    pub fn sign(self) -> f32 { if self == Color::White { 1.0 } else { -1.0 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role { Pawn, Knight, Bishop, Rook, Queen, King }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece { pub color: Color, pub role: Role }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub squares: [Option<Piece>; 64],
    pub turn: Color,
    pub castling: u8, // K=1 Q=2 k=4 q=8
    pub ep_square: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move { pub from: u8, pub to: u8, pub promotion: Option<Role> }

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KNIGHT: &[(i8, i8)] = &[(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)];
const KING: &[(i8, i8)] = &[(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)];
const BISHOP: &[(i8, i8)] = &[(1,1),(1,-1),(-1,1),(-1,-1)];
const ROOK: &[(i8, i8)] = &[(1,0),(-1,0),(0,1),(0,-1)];

fn file(sq: u8) -> i8 { (sq % 8) as i8 }
fn rank(sq: u8) -> i8 { (sq / 8) as i8 }
fn on(f: i8, r: i8) -> bool { (0..8).contains(&f) && (0..8).contains(&r) }
fn idx(f: i8, r: i8) -> u8 { (f + r * 8) as u8 }

pub fn square_index(name: &str) -> Result<u8, String> {
    let b = name.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) { return Err(format!("invalid square: {name}")); }
    Ok((b[0] - b'a') + (b[1] - b'1') * 8)
}

pub fn square_name(sq: u8) -> String {
    let f = (b'a' + sq % 8) as char;
    let r = (b'1' + sq / 8) as char;
    format!("{f}{r}")
}

fn piece_from_fen(ch: char) -> Option<Piece> {
    let color = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
    let role = match ch.to_ascii_lowercase() { 'p'=>Role::Pawn, 'n'=>Role::Knight, 'b'=>Role::Bishop, 'r'=>Role::Rook, 'q'=>Role::Queen, 'k'=>Role::King, _=>return None };
    Some(Piece { color, role })
}

fn piece_to_fen(piece: Piece) -> char {
    let ch = match piece.role { Role::Pawn=>'p', Role::Knight=>'n', Role::Bishop=>'b', Role::Rook=>'r', Role::Queen=>'q', Role::King=>'k' };
    if piece.color == Color::White { ch.to_ascii_uppercase() } else { ch }
}

pub fn parse_fen(fen: &str) -> Result<Board, String> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().ok_or("missing placement")?;
    let turn = match parts.next().unwrap_or("w") { "w"=>Color::White, "b"=>Color::Black, x=>return Err(format!("invalid turn: {x}")) };
    let castling_s = parts.next().unwrap_or("-");
    let ep_s = parts.next().unwrap_or("-");
    let halfmove = parts.next().unwrap_or("0").parse().map_err(|_| "invalid halfmove")?;
    let fullmove = parts.next().unwrap_or("1").parse().map_err(|_| "invalid fullmove")?;
    let mut squares = [None; 64];
    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 { return Err("invalid rank count".into()); }
    for (fen_rank, row) in ranks.iter().enumerate() {
        let mut f = 0i8;
        for ch in row.chars() {
            if ch.is_ascii_digit() { f += ch.to_digit(10).unwrap() as i8; }
            else if let Some(piece) = piece_from_fen(ch) { if f >= 8 { return Err("file overflow".into()); } squares[idx(f, 7 - fen_rank as i8) as usize] = Some(piece); f += 1; }
            else { return Err(format!("invalid piece: {ch}")); }
        }
        if f != 8 { return Err("invalid rank width".into()); }
    }
    let mut castling = 0u8;
    if castling_s.contains('K') { castling |= 1; }
    if castling_s.contains('Q') { castling |= 2; }
    if castling_s.contains('k') { castling |= 4; }
    if castling_s.contains('q') { castling |= 8; }
    let ep_square = if ep_s == "-" { None } else { Some(square_index(ep_s)?) };
    Ok(Board { squares, turn, castling, ep_square, halfmove, fullmove })
}

pub fn board_to_fen(board: &Board) -> String {
    let mut rows = Vec::with_capacity(8);
    for r in (0..8).rev() {
        let mut row = String::new();
        let mut empty = 0;
        for f in 0..8 {
            match board.squares[(f + r * 8) as usize] {
                Some(piece) => { if empty > 0 { row.push_str(&empty.to_string()); empty = 0; } row.push(piece_to_fen(piece)); }
                None => empty += 1,
            }
        }
        if empty > 0 { row.push_str(&empty.to_string()); }
        rows.push(row);
    }
    let turn = if board.turn == Color::White { "w" } else { "b" };
    let mut castling = String::new();
    if board.castling & 1 != 0 { castling.push('K'); }
    if board.castling & 2 != 0 { castling.push('Q'); }
    if board.castling & 4 != 0 { castling.push('k'); }
    if board.castling & 8 != 0 { castling.push('q'); }
    if castling.is_empty() { castling.push('-'); }
    let ep = board.ep_square.map(square_name).unwrap_or_else(|| "-".into());
    format!("{} {turn} {castling} {ep} {} {}", rows.join("/"), board.halfmove, board.fullmove)
}

fn push_step(board: &Board, moves: &mut Vec<Move>, from: u8, df: i8, dr: i8) {
    let (f, r) = (file(from) + df, rank(from) + dr);
    if !on(f, r) { return; }
    let to = idx(f, r);
    if board.squares[to as usize].map(|p| p.color) != Some(board.turn) { moves.push(Move { from, to, promotion: None }); }
}

fn push_slides(board: &Board, moves: &mut Vec<Move>, from: u8, dirs: &[(i8, i8)]) {
    for &(df, dr) in dirs {
        let (mut f, mut r) = (file(from) + df, rank(from) + dr);
        while on(f, r) {
            let to = idx(f, r);
            match board.squares[to as usize] {
                None => moves.push(Move { from, to, promotion: None }),
                Some(piece) => { if piece.color != board.turn { moves.push(Move { from, to, promotion: None }); } break; }
            }
            f += df; r += dr;
        }
    }
}

fn can_castle(board: &Board, mask: u8) -> bool {
    if board.castling & mask == 0 { return false; }
    let white = mask == 1 || mask == 2;
    let color = if white { Color::White } else { Color::Black };
    let r = if white { 0 } else { 7 };
    let king_from = idx(4, r);
    let rook_from = if mask == 1 || mask == 4 { idx(7, r) } else { idx(0, r) };
    if board.turn != color || board.squares[king_from as usize] != Some(Piece{color, role:Role::King}) || board.squares[rook_from as usize] != Some(Piece{color, role:Role::Rook}) { return false; }
    let between: &[u8] = if mask == 1 || mask == 4 { &[idx(5,r), idx(6,r)] } else { &[idx(1,r), idx(2,r), idx(3,r)] };
    if between.iter().any(|&s| board.squares[s as usize].is_some()) { return false; }
    let pass: &[u8] = if mask == 1 || mask == 4 { &[idx(4,r), idx(5,r), idx(6,r)] } else { &[idx(4,r), idx(3,r), idx(2,r)] };
    pass.iter().all(|&s| !is_square_attacked(board, s, color.opposite()))
}

pub fn pseudo_legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    for from in 0u8..64 {
        let Some(piece) = board.squares[from as usize] else { continue; };
        if piece.color != board.turn { continue; }
        match piece.role {
            Role::Pawn => {
                let dir = if board.turn == Color::White { 1 } else { -1 };
                let start = if board.turn == Color::White { 1 } else { 6 };
                let promo = if board.turn == Color::White { 7 } else { 0 };
                let (f, r) = (file(from), rank(from));
                let one = r + dir;
                if on(f, one) && board.squares[idx(f, one) as usize].is_none() {
                    let to = idx(f, one);
                    if one == promo { for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] { moves.push(Move{from,to,promotion:Some(promotion)}); } }
                    else { moves.push(Move{from,to,promotion:None}); }
                    let two = r + 2 * dir;
                    if r == start && on(f, two) && board.squares[idx(f, two) as usize].is_none() { moves.push(Move{from,to:idx(f,two),promotion:None}); }
                }
                for df in [-1, 1] {
                    let cf = f + df;
                    if !on(cf, one) { continue; }
                    let to = idx(cf, one);
                    let capture = board.squares[to as usize].map(|p| p.color) == Some(board.turn.opposite()) || Some(to) == board.ep_square;
                    if capture {
                        if one == promo { for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] { moves.push(Move{from,to,promotion:Some(promotion)}); } }
                        else { moves.push(Move{from,to,promotion:None}); }
                    }
                }
            }
            Role::Knight => for &(df, dr) in KNIGHT { push_step(board, &mut moves, from, df, dr); },
            Role::Bishop => push_slides(board, &mut moves, from, BISHOP),
            Role::Rook => push_slides(board, &mut moves, from, ROOK),
            Role::Queen => { push_slides(board, &mut moves, from, BISHOP); push_slides(board, &mut moves, from, ROOK); }
            Role::King => {
                for &(df, dr) in KING { push_step(board, &mut moves, from, df, dr); }
                if can_castle(board, if board.turn == Color::White { 1 } else { 4 }) { moves.push(Move{from,to:idx(6, if board.turn == Color::White {0} else {7}),promotion:None}); }
                if can_castle(board, if board.turn == Color::White { 2 } else { 8 }) { moves.push(Move{from,to:idx(2, if board.turn == Color::White {0} else {7}),promotion:None}); }
            }
        }
    }
    moves
}

pub fn is_square_attacked(board: &Board, sq: u8, by: Color) -> bool {
    let (f, r) = (file(sq), rank(sq));
    let pawn_rank = r + if by == Color::White { -1 } else { 1 };
    for df in [-1, 1] { let pf = f + df; if on(pf, pawn_rank) && board.squares[idx(pf, pawn_rank) as usize] == Some(Piece{color:by, role:Role::Pawn}) { return true; } }
    for &(df, dr) in KNIGHT { let (nf, nr) = (f + df, r + dr); if on(nf,nr) && board.squares[idx(nf,nr) as usize] == Some(Piece{color:by, role:Role::Knight}) { return true; } }
    for &(df, dr) in BISHOP { if ray_attacked(board, f, r, df, dr, by, Role::Bishop) { return true; } }
    for &(df, dr) in ROOK { if ray_attacked(board, f, r, df, dr, by, Role::Rook) { return true; } }
    for &(df, dr) in KING { let (kf, kr) = (f + df, r + dr); if on(kf,kr) && board.squares[idx(kf,kr) as usize] == Some(Piece{color:by, role:Role::King}) { return true; } }
    false
}

fn ray_attacked(board: &Board, f: i8, r: i8, df: i8, dr: i8, by: Color, slider: Role) -> bool {
    let (mut sf, mut sr) = (f + df, r + dr);
    while on(sf, sr) {
        if let Some(p) = board.squares[idx(sf, sr) as usize] { return p.color == by && (p.role == slider || p.role == Role::Queen); }
        sf += df; sr += dr;
    }
    false
}

pub fn king_square(board: &Board, color: Color) -> Option<u8> {
    board.squares.iter().position(|&p| p == Some(Piece{color, role:Role::King})).map(|i| i as u8)
}

pub fn in_check(board: &Board, color: Color) -> bool {
    king_square(board, color).map(|sq| is_square_attacked(board, sq, color.opposite())).unwrap_or(true)
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    pseudo_legal_moves(board).into_iter().filter(|&m| !in_check(&make_move(board, m), board.turn)).collect()
}

fn remove_castling(c: &mut u8, mask: u8) { *c &= !mask; }

pub fn make_move(board: &Board, m: Move) -> Board {
    let mut next = board.clone();
    let piece = next.squares[m.from as usize].expect("No piece on source square");
    let captured = next.squares[m.to as usize];
    let is_pawn = piece.role == Role::Pawn;
    let is_castle = piece.role == Role::King && (file(m.to) - file(m.from)).abs() == 2;
    let is_ep = is_pawn && Some(m.to) == board.ep_square && captured.is_none() && file(m.from) != file(m.to);
    next.squares[m.from as usize] = None;
    if is_ep { next.squares[idx(file(m.to), rank(m.from)) as usize] = None; }
    next.squares[m.to as usize] = Some(Piece { color: board.turn, role: m.promotion.unwrap_or(piece.role) });
    if is_castle {
        let r = if board.turn == Color::White { 0 } else { 7 };
        if file(m.to) == 6 { next.squares[idx(5,r) as usize] = next.squares[idx(7,r) as usize]; next.squares[idx(7,r) as usize] = None; }
        else { next.squares[idx(3,r) as usize] = next.squares[idx(0,r) as usize]; next.squares[idx(0,r) as usize] = None; }
    }
    match piece { Piece{color:Color::White, role:Role::King} => remove_castling(&mut next.castling, 1|2), Piece{color:Color::Black, role:Role::King} => remove_castling(&mut next.castling, 4|8), _=>{} }
    for (sq, mask) in [(0,2), (7,1), (56,8), (63,4)] { if m.from == sq || m.to == sq { remove_castling(&mut next.castling, mask); } }
    next.turn = board.turn.opposite();
    next.ep_square = if is_pawn && (rank(m.to) - rank(m.from)).abs() == 2 { Some(idx(file(m.from), (rank(m.from) + rank(m.to)) / 2)) } else { None };
    next.halfmove = if is_pawn || captured.is_some() || is_ep { 0 } else { board.halfmove + 1 };
    next.fullmove = board.fullmove + if board.turn == Color::Black { 1 } else { 0 };
    next
}

pub fn fen_features(fen: &str) -> Result<[f32; 15], String> {
    let board = parse_fen(fen)?;
    let mut counts = [0f32; 12];
    for piece in board.squares.iter().flatten() { counts[piece_index(*piece)] += 1.0; }
    let vals = [1f32,3.0,3.0,5.0,9.0,0.0];
    let white_mat: f32 = (0..6).map(|i| counts[i] * vals[i]).sum();
    let black_mat: f32 = (0..6).map(|i| counts[i + 6] * vals[i]).sum();
    let mut feats = [0f32; 15];
    feats[0] = 1.0;
    feats[1] = board.turn.sign();
    for i in 0..12 { feats[2 + i] = (counts[i] - 2.0) / 8.0; }
    feats[14] = (white_mat - black_mat) / 39.0;
    Ok(feats)
}

fn piece_index(piece: Piece) -> usize {
    let base = match piece.role { Role::Pawn=>0, Role::Knight=>1, Role::Bishop=>2, Role::Rook=>3, Role::Queen=>4, Role::King=>5 };
    base + if piece.color == Color::White { 0 } else { 6 }
}

pub fn move_to_uci(m: Move) -> String {
    let promo = match m.promotion { Some(Role::Knight)=>"n", Some(Role::Bishop)=>"b", Some(Role::Rook)=>"r", Some(Role::Queen)=>"q", _=>"" };
    format!("{}{}{}", square_name(m.from), square_name(m.to), promo)
}

pub fn move_to_action_id(m: Move) -> u32 {
    let promo = match m.promotion { Some(Role::Knight)=>1, Some(Role::Bishop)=>2, Some(Role::Rook)=>3, Some(Role::Queen)=>4, _=>0 };
    ((m.from as u32 * 64 + m.to as u32) * 5) + promo
}

#[derive(Clone, Debug, Deserialize)]
pub struct StudentArtifact {
    pub kind: String,
    pub moves: Vec<String>,
    pub policy_weights: Vec<Vec<f32>>,
    pub wdl_weights: Vec<Vec<f32>>,
    pub policy_feature_dim: usize,
    pub wdl_feature_dim: usize,
    pub weight_average_count: Option<u32>,
    pub conv_channels: Option<usize>,
    pub conv_layers: Option<usize>,
}

pub struct Evaluation { pub policy: Vec<(u32, f32)>, pub wdl: [f32; 3] }
pub trait PositionEvaluator { fn evaluate(&self, board: &Board) -> Evaluation; }

pub struct StudentEvaluator { artifact: StudentArtifact, move_index: HashMap<String, usize> }

impl StudentEvaluator {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let artifact: StudentArtifact = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let move_index = artifact.moves.iter().enumerate().map(|(i, m)| (m.clone(), i)).collect();
        Ok(Self { artifact, move_index })
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (*x - m).exp()).collect();
    let total: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.into_iter().map(|x| x / total).collect()
}

fn dot(weights: &[f32], values: &[f32]) -> f32 { weights.iter().zip(values.iter()).map(|(a, b)| a * b).sum() }

fn wdl_features_from_fen(fen: &str) -> Result<Vec<f32>, String> {
    let base = fen_features(fen)?;
    let side = base[1];
    let mut out = base.to_vec();
    out.extend(base[2..].iter().map(|v| side * *v));
    Ok(out)
}

fn stable_weight(values: &[u64]) -> f32 {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &v in values { seed ^= v.wrapping_add(0x9E37_79B9).wrapping_add(seed << 6).wrapping_add(seed >> 2); }
    (((seed % 2001) as f32 / 1000.0) - 1.0) / ((values.len() + 1) as f32).sqrt()
}

fn conv_student_features(fen: &str, channels: usize, layers: usize) -> Vec<f32> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().unwrap_or("8/8/8/8/8/8/8/8");
    let side = parts.next().unwrap_or("w");
    let mut maps = vec![[[0f32; 8]; 8]; 13];
    let (mut rank_i, mut file_i) = (0usize, 0usize);
    for ch in placement.chars() {
        if ch == '/' { rank_i += 1; file_i = 0; }
        else if ch.is_ascii_digit() { file_i += ch.to_digit(10).unwrap() as usize; }
        else if let Some(pi) = "PNBRQKpnbrqk".find(ch) { maps[pi][rank_i][file_i] = 1.0; file_i += 1; }
    }
    let side_value = if side == "w" { 1.0 } else { -1.0 };
    for r in 0..8 { for f in 0..8 { maps[12][r][f] = side_value; } }
    let mut prev = maps;
    for layer in 0..layers {
        let prev_channels = prev.len();
        let mut out = vec![[[0f32; 8]; 8]; channels];
        for c in 0..channels { for r in 0..8usize { for f in 0..8usize {
            let mut acc = stable_weight(&[layer as u64, c as u64, 99]);
            for (pc, prev_map) in prev.iter().enumerate() { for dri in 0..3usize {
                let rr = r as isize + dri as isize - 1; if !(0..8).contains(&rr) { continue; }
                for dfi in 0..3usize { let ff = f as isize + dfi as isize - 1; if (0..8).contains(&ff) {
                    let k = stable_weight(&[layer as u64, c as u64, pc as u64, dri as u64, dfi as u64]);
                    acc += prev_map[rr as usize][ff as usize] * k;
                } }
            } }
            out[c][r][f] = (acc / ((prev_channels * 4) as f32).sqrt()).tanh();
        } } }
        prev = out;
    }
    let mut feats = Vec::with_capacity(2 + channels * 3);
    feats.push(1.0); feats.push(side_value);
    for channel in prev.iter() {
        let (mut sum, mut mx, mut mn) = (0f32, f32::NEG_INFINITY, f32::INFINITY);
        for row in channel { for &v in row { sum += v; mx = mx.max(v); mn = mn.min(v); } }
        feats.push(sum / 64.0); feats.push(mx); feats.push(mn);
    }
    feats
}

impl PositionEvaluator for StudentEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation {
        let fen = board_to_fen(board);
        let policy_features = if self.artifact.kind == "frozen_conv_fen_student" { conv_student_features(&fen, self.artifact.conv_channels.unwrap_or(0), self.artifact.conv_layers.unwrap_or(0)) } else { fen_features(&fen).unwrap().to_vec() };
        let value_features = if self.artifact.kind == "frozen_conv_fen_student" { policy_features.clone() } else { wdl_features_from_fen(&fen).unwrap() };
        let logits: Vec<f32> = self.artifact.policy_weights.iter().map(|w| dot(w, &policy_features)).collect();
        let probs = softmax(&logits);
        let legal = legal_moves(board);
        let mut raw = Vec::with_capacity(legal.len());
        let mut legal_mass = 0f32;
        for m in legal.iter() { let p = self.move_index.get(&move_to_uci(*m)).and_then(|&i| probs.get(i)).copied().unwrap_or(0.0).max(0.0); raw.push(p); legal_mass += p; }
        let policy = if legal.is_empty() { Vec::new() } else if legal_mass <= 0.0 { legal.iter().map(|&m| (move_to_action_id(m), 1.0 / legal.len() as f32)).collect() } else { legal.iter().zip(raw.iter()).map(|(&m, &p)| (move_to_action_id(m), p / legal_mass)).collect() };
        let wdl_logits: Vec<f32> = self.artifact.wdl_weights.iter().map(|w| dot(w, &value_features)).collect();
        let w = softmax(&wdl_logits);
        Evaluation { policy, wdl: [w.first().copied().unwrap_or(0.0), w.get(1).copied().unwrap_or(0.0), w.get(2).copied().unwrap_or(0.0)] }
    }
}

pub struct UniformEvaluator;
impl PositionEvaluator for UniformEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation {
        let moves = legal_moves(board); let p = if moves.is_empty() { 0.0 } else { 1.0 / moves.len() as f32 };
        Evaluation { policy: moves.into_iter().map(|m| (move_to_action_id(m), p)).collect(), wdl: [0.25, 0.5, 0.25] }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchOptions { pub visits: u32, pub cpuct: f32, pub temperature: f32 }
impl Default for SearchOptions { fn default() -> Self { Self { visits: 8, cpuct: 1.5, temperature: 1.0 } } }
#[derive(Clone, Debug)]
pub struct SearchPolicyEntry { pub mv: Move, pub visits: u32, pub prior: f32, pub q: f32, pub probability: f32 }
#[derive(Clone, Debug)]
pub struct SearchResult { pub mv: Option<Move>, pub visits: u32, pub value: f32, pub policy: Vec<SearchPolicyEntry> }
struct Edge { mv: Move, prior: f32, child: Option<Box<Node>>, visits: u32, value_sum: f32 }
struct Node { board: Board, expanded: bool, terminal_value: Option<f32>, edges: Vec<Edge> }
fn value_from_wdl(wdl: [f32; 3]) -> f32 { wdl[0] - wdl[2] }
fn edge_q_for_parent(edge: &Edge) -> f32 { if edge.visits > 0 { -edge.value_sum / edge.visits as f32 } else { 0.0 } }
fn expand_node(node: &mut Node, evaluator: &dyn PositionEvaluator) -> f32 {
    let moves = legal_moves(&node.board);
    if moves.is_empty() { node.expanded = true; node.terminal_value = Some(if in_check(&node.board, node.board.turn) { -1.0 } else { 0.0 }); node.edges.clear(); return node.terminal_value.unwrap(); }
    let evaln = evaluator.evaluate(&node.board);
    let policy_map: HashMap<u32, f32> = evaln.policy.into_iter().collect();
    let raw: Vec<f32> = moves.iter().map(|&m| policy_map.get(&move_to_action_id(m)).copied().unwrap_or(0.0).max(0.0)).collect();
    let total: f32 = raw.iter().sum(); let fallback = 1.0 / moves.len() as f32;
    node.edges = moves.into_iter().enumerate().map(|(i, mv)| Edge { mv, prior: if total > 0.0 { raw[i] / total } else { fallback }, child: None, visits: 0, value_sum: 0.0 }).collect();
    node.expanded = true; node.terminal_value = None; value_from_wdl(evaln.wdl)
}
fn simulate(node: &mut Node, evaluator: &dyn PositionEvaluator, cpuct: f32) -> f32 {
    if !node.expanded { return expand_node(node, evaluator); }
    if let Some(v) = node.terminal_value { return v; }
    let parent_visits: u32 = node.edges.iter().map(|e| e.visits).sum(); let sqrt_parent = ((parent_visits + 1) as f32).sqrt();
    let mut best_i = 0usize; let mut best_score = f32::NEG_INFINITY;
    for (i, edge) in node.edges.iter().enumerate() { let score = edge_q_for_parent(edge) + cpuct * edge.prior * sqrt_parent / (1.0 + edge.visits as f32); if score > best_score { best_i = i; best_score = score; } }
    let edge = &mut node.edges[best_i];
    if edge.child.is_none() { edge.child = Some(Box::new(Node { board: make_move(&node.board, edge.mv), expanded: false, terminal_value: None, edges: Vec::new() })); }
    let child_value = simulate(edge.child.as_mut().unwrap(), evaluator, cpuct);
    edge.visits += 1; edge.value_sum += child_value; -child_value
}
fn visit_policy(edges: &[Edge], temperature: f32) -> Vec<SearchPolicyEntry> {
    if edges.is_empty() { return Vec::new(); }
    let tau = temperature.max(0.0);
    if tau == 0.0 { let best_i = edges.iter().enumerate().max_by_key(|(_, e)| e.visits).map(|(i, _)| i).unwrap_or(0); return edges.iter().enumerate().map(|(i, e)| SearchPolicyEntry { mv: e.mv, visits: e.visits, prior: e.prior, q: edge_q_for_parent(e), probability: if i == best_i { 1.0 } else { 0.0 } }).collect(); }
    let weights: Vec<f32> = edges.iter().map(|e| (e.visits as f32).max(1e-9).powf(1.0 / tau)).collect(); let total: f32 = weights.iter().sum::<f32>().max(1e-30);
    edges.iter().zip(weights.iter()).map(|(e, &w)| SearchPolicyEntry { mv: e.mv, visits: e.visits, prior: e.prior, q: edge_q_for_parent(e), probability: w / total }).collect()
}
pub fn search_root(board: &Board, evaluator: &dyn PositionEvaluator, options: SearchOptions) -> SearchResult {
    let visits = options.visits.max(1); let mut root = Node { board: board.clone(), expanded: false, terminal_value: None, edges: Vec::new() };
    let root_value = expand_node(&mut root, evaluator); if root.edges.is_empty() { return SearchResult { mv: None, visits: 0, value: root_value, policy: Vec::new() }; }
    for _ in 0..visits { simulate(&mut root, evaluator, options.cpuct); }
    let policy = visit_policy(&root.edges, options.temperature); let best = policy.iter().max_by(|a, b| a.probability.total_cmp(&b.probability));
    SearchResult { mv: best.map(|e| e.mv), visits: root.edges.iter().map(|e| e.visits).sum(), value: best.map(|e| e.q).unwrap_or(root_value), policy }
}

#[no_mangle]
pub extern "C" fn tiny_leela_startpos_uniform_search_best_action(visits: u32) -> u32 {
    let board = parse_fen(START_FEN).expect("valid start fen");
    search_root(&board, &UniformEvaluator, SearchOptions { visits: visits.max(1), cpuct: 1.5, temperature: 0.0 }).mv.map(move_to_action_id).unwrap_or(u32::MAX)
}

#[no_mangle]
pub extern "C" fn tiny_leela_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn tiny_leela_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() { drop(Vec::from_raw_parts(ptr, 0, len)); }
}

unsafe fn wasm_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, ()> {
    if ptr.is_null() { return Err(()); }
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).map_err(|_| ())
}

#[no_mangle]
pub unsafe extern "C" fn tiny_leela_student_wdl(
    artifact_ptr: *const u8,
    artifact_len: usize,
    fen_ptr: *const u8,
    fen_len: usize,
    out_wdl_ptr: *mut f32,
) -> i32 {
    let Ok(artifact_json) = wasm_str(artifact_ptr, artifact_len) else { return -1; };
    let Ok(fen) = wasm_str(fen_ptr, fen_len) else { return -2; };
    let Ok(evaluator) = StudentEvaluator::from_json(artifact_json) else { return -3; };
    let Ok(board) = parse_fen(fen) else { return -4; };
    if out_wdl_ptr.is_null() { return -5; }
    let evaln = evaluator.evaluate(&board);
    for i in 0..3 { *out_wdl_ptr.add(i) = evaln.wdl[i]; }
    0
}

#[no_mangle]
pub unsafe extern "C" fn tiny_leela_student_search_best_action(
    artifact_ptr: *const u8,
    artifact_len: usize,
    fen_ptr: *const u8,
    fen_len: usize,
    visits: u32,
) -> u32 {
    let Ok(artifact_json) = wasm_str(artifact_ptr, artifact_len) else { return u32::MAX; };
    let Ok(fen) = wasm_str(fen_ptr, fen_len) else { return u32::MAX; };
    let Ok(evaluator) = StudentEvaluator::from_json(artifact_json) else { return u32::MAX; };
    let Ok(board) = parse_fen(fen) else { return u32::MAX; };
    search_root(&board, &evaluator, SearchOptions { visits: visits.max(1), cpuct: 1.5, temperature: 0.0 }).mv.map(move_to_action_id).unwrap_or(u32::MAX)
}

#[no_mangle]
pub extern "C" fn tiny_leela_startpos_legal_count() -> u32 {
    legal_moves(&parse_fen(START_FEN).expect("valid start fen")).len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn start_position_has_20_legal_moves() { assert_eq!(legal_moves(&parse_fen(START_FEN).unwrap()).len(), 20); }
    #[test]
    fn fen_roundtrip_start() { assert_eq!(board_to_fen(&parse_fen(START_FEN).unwrap()), START_FEN); }
    #[test]
    fn rejects_king_into_check() {
        let b = parse_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves: Vec<String> = legal_moves(&b).into_iter().map(move_to_uci).collect();
        assert!(!moves.contains(&"e1e2".to_string()));
    }
    #[test]
    fn castling_and_ep() {
        let b = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let moves: Vec<String> = legal_moves(&b).into_iter().map(move_to_uci).collect();
        assert!(moves.contains(&"e1g1".to_string()));
        assert!(moves.contains(&"e1c1".to_string()));
        let ep = make_move(&parse_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1").unwrap(), Move{from:36,to:43,promotion:None});
        assert!(ep.squares[35].is_none());
        assert_eq!(ep.squares[43], Some(Piece{color:Color::White, role:Role::Pawn}));
    }
}
