use serde::Deserialize;
use std::collections::HashMap;

mod board;
pub use board::{
    board_to_fen, parse_fen, square_index, square_name, Board, Color, Move, Piece, Role, START_FEN,
};
use board::{file, idx, on, rank, BISHOP, KING, KNIGHT, ROOK};

fn push_step(board: &Board, moves: &mut Vec<Move>, from: u8, df: i8, dr: i8) {
    let (f, r) = (file(from) + df, rank(from) + dr);
    if !on(f, r) {
        return;
    }
    let to = idx(f, r);
    if board.squares[to as usize].map(|p| p.color) != Some(board.turn) {
        moves.push(Move {
            from,
            to,
            promotion: None,
        });
    }
}

fn push_slides(board: &Board, moves: &mut Vec<Move>, from: u8, dirs: &[(i8, i8)]) {
    for &(df, dr) in dirs {
        let (mut f, mut r) = (file(from) + df, rank(from) + dr);
        while on(f, r) {
            let to = idx(f, r);
            match board.squares[to as usize] {
                None => moves.push(Move {
                    from,
                    to,
                    promotion: None,
                }),
                Some(piece) => {
                    if piece.color != board.turn {
                        moves.push(Move {
                            from,
                            to,
                            promotion: None,
                        });
                    }
                    break;
                }
            }
            f += df;
            r += dr;
        }
    }
}

fn can_castle(board: &Board, mask: u8) -> bool {
    if board.castling & mask == 0 {
        return false;
    }
    let white = mask == 1 || mask == 2;
    let color = if white { Color::White } else { Color::Black };
    let r = if white { 0 } else { 7 };
    let king_from = idx(4, r);
    let rook_from = if mask == 1 || mask == 4 {
        idx(7, r)
    } else {
        idx(0, r)
    };
    if board.turn != color
        || board.squares[king_from as usize]
            != Some(Piece {
                color,
                role: Role::King,
            })
        || board.squares[rook_from as usize]
            != Some(Piece {
                color,
                role: Role::Rook,
            })
    {
        return false;
    }
    let between: &[u8] = if mask == 1 || mask == 4 {
        &[idx(5, r), idx(6, r)]
    } else {
        &[idx(1, r), idx(2, r), idx(3, r)]
    };
    if between.iter().any(|&s| board.squares[s as usize].is_some()) {
        return false;
    }
    let pass: &[u8] = if mask == 1 || mask == 4 {
        &[idx(4, r), idx(5, r), idx(6, r)]
    } else {
        &[idx(4, r), idx(3, r), idx(2, r)]
    };
    pass.iter()
        .all(|&s| !is_square_attacked(board, s, color.opposite()))
}

pub fn pseudo_legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    for from in 0u8..64 {
        let Some(piece) = board.squares[from as usize] else {
            continue;
        };
        if piece.color != board.turn {
            continue;
        }
        match piece.role {
            Role::Pawn => {
                let dir = if board.turn == Color::White { 1 } else { -1 };
                let start = if board.turn == Color::White { 1 } else { 6 };
                let promo = if board.turn == Color::White { 7 } else { 0 };
                let (f, r) = (file(from), rank(from));
                let one = r + dir;
                if on(f, one) && board.squares[idx(f, one) as usize].is_none() {
                    let to = idx(f, one);
                    if one == promo {
                        for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
                            moves.push(Move {
                                from,
                                to,
                                promotion: Some(promotion),
                            });
                        }
                    } else {
                        moves.push(Move {
                            from,
                            to,
                            promotion: None,
                        });
                    }
                    let two = r + 2 * dir;
                    if r == start && on(f, two) && board.squares[idx(f, two) as usize].is_none() {
                        moves.push(Move {
                            from,
                            to: idx(f, two),
                            promotion: None,
                        });
                    }
                }
                for df in [-1, 1] {
                    let cf = f + df;
                    if !on(cf, one) {
                        continue;
                    }
                    let to = idx(cf, one);
                    let capture = board.squares[to as usize].map(|p| p.color)
                        == Some(board.turn.opposite())
                        || Some(to) == board.ep_square;
                    if capture {
                        if one == promo {
                            for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
                                moves.push(Move {
                                    from,
                                    to,
                                    promotion: Some(promotion),
                                });
                            }
                        } else {
                            moves.push(Move {
                                from,
                                to,
                                promotion: None,
                            });
                        }
                    }
                }
            }
            Role::Knight => {
                for &(df, dr) in KNIGHT {
                    push_step(board, &mut moves, from, df, dr);
                }
            }
            Role::Bishop => push_slides(board, &mut moves, from, BISHOP),
            Role::Rook => push_slides(board, &mut moves, from, ROOK),
            Role::Queen => {
                push_slides(board, &mut moves, from, BISHOP);
                push_slides(board, &mut moves, from, ROOK);
            }
            Role::King => {
                for &(df, dr) in KING {
                    push_step(board, &mut moves, from, df, dr);
                }
                if can_castle(board, if board.turn == Color::White { 1 } else { 4 }) {
                    moves.push(Move {
                        from,
                        to: idx(6, if board.turn == Color::White { 0 } else { 7 }),
                        promotion: None,
                    });
                }
                if can_castle(board, if board.turn == Color::White { 2 } else { 8 }) {
                    moves.push(Move {
                        from,
                        to: idx(2, if board.turn == Color::White { 0 } else { 7 }),
                        promotion: None,
                    });
                }
            }
        }
    }
    moves
}

pub fn is_square_attacked(board: &Board, sq: u8, by: Color) -> bool {
    let (f, r) = (file(sq), rank(sq));
    let pawn_rank = r + if by == Color::White { -1 } else { 1 };
    for df in [-1, 1] {
        let pf = f + df;
        if on(pf, pawn_rank)
            && board.squares[idx(pf, pawn_rank) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::Pawn,
                })
        {
            return true;
        }
    }
    for &(df, dr) in KNIGHT {
        let (nf, nr) = (f + df, r + dr);
        if on(nf, nr)
            && board.squares[idx(nf, nr) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::Knight,
                })
        {
            return true;
        }
    }
    for &(df, dr) in BISHOP {
        if ray_attacked(board, f, r, df, dr, by, Role::Bishop) {
            return true;
        }
    }
    for &(df, dr) in ROOK {
        if ray_attacked(board, f, r, df, dr, by, Role::Rook) {
            return true;
        }
    }
    for &(df, dr) in KING {
        let (kf, kr) = (f + df, r + dr);
        if on(kf, kr)
            && board.squares[idx(kf, kr) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::King,
                })
        {
            return true;
        }
    }
    false
}

fn ray_attacked(board: &Board, f: i8, r: i8, df: i8, dr: i8, by: Color, slider: Role) -> bool {
    let (mut sf, mut sr) = (f + df, r + dr);
    while on(sf, sr) {
        if let Some(p) = board.squares[idx(sf, sr) as usize] {
            return p.color == by && (p.role == slider || p.role == Role::Queen);
        }
        sf += df;
        sr += dr;
    }
    false
}

pub fn king_square(board: &Board, color: Color) -> Option<u8> {
    board
        .squares
        .iter()
        .position(|&p| {
            p == Some(Piece {
                color,
                role: Role::King,
            })
        })
        .map(|i| i as u8)
}

pub fn in_check(board: &Board, color: Color) -> bool {
    king_square(board, color)
        .map(|sq| is_square_attacked(board, sq, color.opposite()))
        .unwrap_or(true)
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    pseudo_legal_moves(board)
        .into_iter()
        .filter(|&m| !in_check(&make_move(board, m), board.turn))
        .collect()
}

fn remove_castling(c: &mut u8, mask: u8) {
    *c &= !mask;
}

pub fn make_move(board: &Board, m: Move) -> Board {
    let mut next = board.clone();
    let piece = next.squares[m.from as usize].expect("No piece on source square");
    let captured = next.squares[m.to as usize];
    let is_pawn = piece.role == Role::Pawn;
    let is_castle = piece.role == Role::King && (file(m.to) - file(m.from)).abs() == 2;
    let is_ep = is_pawn
        && Some(m.to) == board.ep_square
        && captured.is_none()
        && file(m.from) != file(m.to);
    next.squares[m.from as usize] = None;
    if is_ep {
        next.squares[idx(file(m.to), rank(m.from)) as usize] = None;
    }
    next.squares[m.to as usize] = Some(Piece {
        color: board.turn,
        role: m.promotion.unwrap_or(piece.role),
    });
    if is_castle {
        let r = if board.turn == Color::White { 0 } else { 7 };
        if file(m.to) == 6 {
            next.squares[idx(5, r) as usize] = next.squares[idx(7, r) as usize];
            next.squares[idx(7, r) as usize] = None;
        } else {
            next.squares[idx(3, r) as usize] = next.squares[idx(0, r) as usize];
            next.squares[idx(0, r) as usize] = None;
        }
    }
    match piece {
        Piece {
            color: Color::White,
            role: Role::King,
        } => remove_castling(&mut next.castling, 1 | 2),
        Piece {
            color: Color::Black,
            role: Role::King,
        } => remove_castling(&mut next.castling, 4 | 8),
        _ => {}
    }
    for (sq, mask) in [(0, 2), (7, 1), (56, 8), (63, 4)] {
        if m.from == sq || m.to == sq {
            remove_castling(&mut next.castling, mask);
        }
    }
    next.turn = board.turn.opposite();
    next.ep_square = if is_pawn && (rank(m.to) - rank(m.from)).abs() == 2 {
        Some(idx(file(m.from), (rank(m.from) + rank(m.to)) / 2))
    } else {
        None
    };
    next.halfmove = if is_pawn || captured.is_some() || is_ep {
        0
    } else {
        board.halfmove + 1
    };
    next.fullmove = board.fullmove + if board.turn == Color::Black { 1 } else { 0 };
    next
}

pub fn fen_features(fen: &str) -> Result<[f32; 15], String> {
    let board = parse_fen(fen)?;
    let mut counts = [0f32; 12];
    for piece in board.squares.iter().flatten() {
        counts[piece_index(*piece)] += 1.0;
    }
    let vals = [1f32, 3.0, 3.0, 5.0, 9.0, 0.0];
    let white_mat: f32 = (0..6).map(|i| counts[i] * vals[i]).sum();
    let black_mat: f32 = (0..6).map(|i| counts[i + 6] * vals[i]).sum();
    let mut feats = [0f32; 15];
    feats[0] = 1.0;
    feats[1] = board.turn.sign();
    for i in 0..12 {
        feats[2 + i] = (counts[i] - 2.0) / 8.0;
    }
    feats[14] = (white_mat - black_mat) / 39.0;
    Ok(feats)
}

fn piece_index(piece: Piece) -> usize {
    let base = match piece.role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    };
    base + if piece.color == Color::White { 0 } else { 6 }
}

pub fn move_to_uci(m: Move) -> String {
    let promo = match m.promotion {
        Some(Role::Knight) => "n",
        Some(Role::Bishop) => "b",
        Some(Role::Rook) => "r",
        Some(Role::Queen) => "q",
        _ => "",
    };
    format!("{}{}{}", square_name(m.from), square_name(m.to), promo)
}

pub fn move_to_action_id(m: Move) -> u32 {
    let promo = match m.promotion {
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        _ => 0,
    };
    ((m.from as u32 * 64 + m.to as u32) * 5) + promo
}

#[derive(Clone, Debug, Deserialize)]
pub struct StudentArtifact {
    pub kind: String,
    pub moves: Vec<String>,
    #[serde(default)]
    pub policy_weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub wdl_weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub policy_feature_dim: usize,
    #[serde(default)]
    pub wdl_feature_dim: usize,
    pub weight_average_count: Option<u32>,
    pub conv_channels: Option<usize>,
    pub conv_layers: Option<usize>,
    pub feature_dim: Option<usize>,
    pub hidden: Option<usize>,
    pub w1: Option<Vec<Vec<f32>>>,
    pub b1: Option<Vec<f32>>,
    pub policy_w: Option<Vec<Vec<f32>>>,
    pub policy_b: Option<Vec<f32>>,
    pub wdl_w: Option<Vec<Vec<f32>>>,
    pub wdl_b: Option<Vec<f32>>,
    pub channels: Option<usize>,
    pub c1_weight: Option<Vec<Vec<Vec<Vec<f32>>>>>,
    pub c1_bias: Option<Vec<f32>>,
    pub c2_weight: Option<Vec<Vec<Vec<Vec<f32>>>>>,
    pub c2_bias: Option<Vec<f32>>,
    pub c3_weight: Option<Vec<Vec<Vec<Vec<f32>>>>>,
    pub c3_bias: Option<Vec<f32>>,
    pub policy_weight: Option<Vec<Vec<f32>>>,
    pub policy_bias: Option<Vec<f32>>,
    pub wdl_weight: Option<Vec<Vec<f32>>>,
    pub wdl_bias: Option<Vec<f32>>,
    pub policy_head: Option<String>,
    pub policy_map: Option<String>,
    pub history_plies: Option<usize>,
    pub input_planes: Option<usize>,
    pub architecture: Option<String>,
    pub blocks: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub policy: Vec<(u32, f32)>,
    pub wdl: [f32; 3],
    pub action_values: Option<Vec<(u32, f32)>>,
    pub rank_scores: Option<Vec<(u32, f32)>>,
    pub regrets: Option<Vec<(u32, f32)>>,
    pub risks: Option<Vec<(u32, f32)>>,
    pub uncertainties: Option<Vec<(u32, f32)>>,
}

impl Evaluation {
    pub fn new(policy: Vec<(u32, f32)>, wdl: [f32; 3]) -> Self {
        Self {
            policy,
            wdl,
            action_values: None,
            rank_scores: None,
            regrets: None,
            risks: None,
            uncertainties: None,
        }
    }
}
pub trait PositionEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation;
    fn evaluate_with_history(&self, board: &Board, _history_fens: &[String]) -> Evaluation {
        self.evaluate(board)
    }
}

struct ConvLayerParams {
    prev_channels: usize,
    biases: Vec<f32>,
    kernels: Vec<f32>, // (((out_c * prev_channels + in_c) * 3 + dr) * 3 + df)
}

struct ConvParams {
    layers: Vec<ConvLayerParams>,
}

pub struct StudentEvaluator {
    artifact: StudentArtifact,
    move_index: HashMap<String, usize>,
    conv_params: Option<ConvParams>,
}

impl StudentEvaluator {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let artifact: StudentArtifact = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let move_index = artifact
            .moves
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i))
            .collect();
        let conv_params = if artifact.kind == "frozen_conv_fen_student"
            || artifact.kind == "frozen_conv_feature_mlp_student"
        {
            Some(precompute_conv_params(
                artifact.conv_channels.unwrap_or(0),
                artifact.conv_layers.unwrap_or(0),
            ))
        } else {
            None
        };
        Ok(Self {
            artifact,
            move_index,
            conv_params,
        })
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (*x - m).exp()).collect();
    let total: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.into_iter().map(|x| x / total).collect()
}

fn dot(weights: &[f32], values: &[f32]) -> f32 {
    weights.iter().zip(values.iter()).map(|(a, b)| a * b).sum()
}

fn fixed_policy_moves() -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    let files = b"abcdefgh";
    let sq = |f: i32, r: i32| format!("{}{}", files[f as usize] as char, r + 1);
    let dirs = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let knights = [
        (1, 2),
        (2, 1),
        (-1, 2),
        (-2, 1),
        (1, -2),
        (2, -1),
        (-1, -2),
        (-2, -1),
    ];
    for r in 0..8 {
        for f in 0..8 {
            let from = sq(f, r);
            for (df, dr) in dirs {
                for n in 1..8 {
                    let tf = f + df * n;
                    let tr = r + dr * n;
                    if !(0..8).contains(&tf) || !(0..8).contains(&tr) {
                        break;
                    }
                    out.insert(format!("{}{}", from, sq(tf, tr)));
                }
            }
            for (df, dr) in knights {
                let tf = f + df;
                let tr = r + dr;
                if (0..8).contains(&tf) && (0..8).contains(&tr) {
                    out.insert(format!("{}{}", from, sq(tf, tr)));
                }
            }
        }
    }
    for r in [1, 6] {
        let tr = if r == 6 { 7 } else { 0 };
        for f in 0..8 {
            for df in [-1, 0, 1] {
                let tf = f + df;
                if (0..8).contains(&tf) {
                    for p in ['q', 'r', 'b', 'n'] {
                        out.insert(format!("{}{}{}", sq(f, r), sq(tf, tr), p));
                    }
                }
            }
        }
    }
    out.into_iter().collect()
}

fn move_to_fixed_policy_index(m: Move) -> Option<usize> {
    let uci = move_to_uci(m);
    fixed_policy_moves().binary_search(&uci).ok()
}

fn wdl_features_from_fen(fen: &str) -> Result<Vec<f32>, String> {
    let base = fen_features(fen)?;
    let side = base[1];
    let mut out = base.to_vec();
    out.extend(base[2..].iter().map(|v| side * *v));
    Ok(out)
}

fn stable_weight(values: &[u64]) -> f32 {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &v in values {
        seed ^= v
            .wrapping_add(0x9E37_79B9)
            .wrapping_add(seed << 6)
            .wrapping_add(seed >> 2);
    }
    (((seed % 2001) as f32 / 1000.0) - 1.0) / ((values.len() + 1) as f32).sqrt()
}

fn kernel_index(out_c: usize, prev_channels: usize, in_c: usize, dr: usize, df: usize) -> usize {
    (((out_c * prev_channels + in_c) * 3 + dr) * 3) + df
}

fn precompute_conv_params(channels: usize, layers: usize) -> ConvParams {
    let mut out = Vec::with_capacity(layers);
    for layer in 0..layers {
        let prev_channels = if layer == 0 { 13 } else { channels };
        let mut biases = vec![0.0; channels];
        let mut kernels = vec![0.0; channels * prev_channels * 3 * 3];
        for c in 0..channels {
            biases[c] = stable_weight(&[layer as u64, c as u64, 99]);
            for pc in 0..prev_channels {
                for dri in 0..3usize {
                    for dfi in 0..3usize {
                        kernels[kernel_index(c, prev_channels, pc, dri, dfi)] = stable_weight(&[
                            layer as u64,
                            c as u64,
                            pc as u64,
                            dri as u64,
                            dfi as u64,
                        ]);
                    }
                }
            }
        }
        out.push(ConvLayerParams {
            prev_channels,
            biases,
            kernels,
        });
    }
    ConvParams { layers: out }
}

fn board_planes(fen: &str, input_planes: usize, history_plies: usize) -> Vec<[[f32; 8]; 8]> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().unwrap_or("8/8/8/8/8/8/8/8");
    let side = parts.next().unwrap_or("w");
    let castling = parts.next().unwrap_or("-");
    let ep = parts.next().unwrap_or("-");
    let mut maps = vec![[[0f32; 8]; 8]; input_planes];
    let (mut rank_i, mut file_i) = (0usize, 0usize);
    for ch in placement.chars() {
        if ch == '/' {
            rank_i += 1;
            file_i = 0;
        } else if ch.is_ascii_digit() {
            file_i += ch.to_digit(10).unwrap() as usize;
        } else if let Some(pi) = "PNBRQKpnbrqk".find(ch) {
            maps[pi][rank_i][file_i] = 1.0;
            file_i += 1;
        }
    }
    let state0 = 12 * (history_plies + 1);
    let side_value = if side == "w" { 1.0 } else { -1.0 };
    for r in 0..8 {
        for f in 0..8 {
            maps[state0][r][f] = side_value;
        }
    }
    if input_planes.saturating_sub(state0) >= 10 {
        for (i, flag) in ['K', 'Q', 'k', 'q'].iter().enumerate() {
            if castling.contains(*flag) {
                for r in 0..8 {
                    for f in 0..8 {
                        maps[state0 + 1 + i][r][f] = 1.0;
                    }
                }
            }
        }
        if ep != "-" {
            let b = ep.as_bytes();
            if b.len() >= 2 {
                let ef = b[0].saturating_sub(b'a') as usize;
                let er = 8usize.saturating_sub((b[1].saturating_sub(b'0')) as usize);
                if er < 8 && ef < 8 {
                    maps[state0 + 5][er][ef] = 1.0;
                }
            }
        }
        let (mut stm_check, mut opp_check) = (0.0, 0.0);
        if input_planes.saturating_sub(state0) >= 10 {
            if let Ok(board) = parse_fen(fen) {
                stm_check = if in_check(&board, board.turn) {
                    1.0
                } else {
                    0.0
                };
                opp_check = if in_check(&board, board.turn.opposite()) {
                    1.0
                } else {
                    0.0
                };
            }
        }
        for r in 0..8 {
            for f in 0..8 {
                maps[state0 + 6][r][f] = 1.0;
                maps[state0 + 7][r][f] = if side == "w" { 1.0 } else { 0.0 };
                maps[state0 + 8][r][f] = stm_check;
                maps[state0 + 9][r][f] = opp_check;
            }
        }
    } else {
        for r in 0..8 {
            for f in 0..8 {
                maps[state0 + 1][r][f] = 1.0;
            }
        }
    }
    maps
}

fn board_cnn_forward(fen: &str, a: &StudentArtifact) -> (Vec<f32>, Vec<f32>) {
    fn conv_relu_res(
        input: &[[[f32; 8]; 8]],
        weight: &[Vec<Vec<Vec<f32>>>],
        bias: &[f32],
        residual: bool,
    ) -> Vec<[[f32; 8]; 8]> {
        let out_c = bias.len();
        let in_c = input.len();
        let mut out = vec![[[0f32; 8]; 8]; out_c];
        for oc in 0..out_c {
            for r in 0..8usize {
                for f in 0..8usize {
                    let mut acc = bias[oc];
                    for ic in 0..in_c {
                        for kr in 0..3usize {
                            for kf in 0..3usize {
                                let rr = r as isize + kr as isize - 1;
                                let ff = f as isize + kf as isize - 1;
                                if (0..8).contains(&rr) && (0..8).contains(&ff) {
                                    acc += input[ic][rr as usize][ff as usize]
                                        * weight[oc][ic][kr][kf];
                                }
                            }
                        }
                    }
                    let v = acc.max(0.0);
                    out[oc][r][f] = if residual && oc < input.len() {
                        v + input[oc][r][f]
                    } else {
                        v
                    };
                }
            }
        }
        out
    }
    let input_planes = a.input_planes.unwrap_or_else(|| {
        a.c1_weight
            .as_ref()
            .and_then(|w| w.first().map(|oc| oc.len()))
            .unwrap_or(14)
    });
    let x0 = board_planes(fen, input_planes, a.history_plies.unwrap_or(0));
    let h1 = conv_relu_res(
        &x0,
        a.c1_weight.as_ref().unwrap(),
        a.c1_bias.as_ref().unwrap(),
        false,
    );
    let h2 = conv_relu_res(
        &h1,
        a.c2_weight.as_ref().unwrap(),
        a.c2_bias.as_ref().unwrap(),
        true,
    );
    let h3 = conv_relu_res(
        &h2,
        a.c3_weight.as_ref().unwrap(),
        a.c3_bias.as_ref().unwrap(),
        true,
    );
    let mut pooled = vec![0f32; h3.len()];
    let mut spatial = Vec::with_capacity(h3.len() * 64);
    for c in 0..h3.len() {
        for r in 0..8 {
            for f in 0..8 {
                pooled[c] += h3[c][r][f] / 64.0;
                spatial.push(h3[c][r][f]);
            }
        }
    }
    let policy_features = if a.policy_head.as_deref() == Some("spatial") {
        &spatial
    } else {
        &pooled
    };
    let pw = a.policy_weight.as_ref().unwrap();
    let pb = a.policy_bias.as_ref().unwrap();
    let vw = a.wdl_weight.as_ref().unwrap();
    let vb = a.wdl_bias.as_ref().unwrap();
    let policy = (0..pb.len())
        .map(|m| {
            pb[m]
                + policy_features
                    .iter()
                    .enumerate()
                    .map(|(c, &x)| x * pw[m][c])
                    .sum::<f32>()
        })
        .collect();
    let wdl = (0..vb.len())
        .map(|k| {
            vb[k]
                + pooled
                    .iter()
                    .enumerate()
                    .map(|(c, &x)| x * vw[k][c])
                    .sum::<f32>()
        })
        .collect();
    (policy, wdl)
}

fn conv_student_features(
    fen: &str,
    channels: usize,
    layers: usize,
    params: Option<&ConvParams>,
) -> Vec<f32> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().unwrap_or("8/8/8/8/8/8/8/8");
    let side = parts.next().unwrap_or("w");
    let mut maps = vec![[[0f32; 8]; 8]; 13];
    let (mut rank_i, mut file_i) = (0usize, 0usize);
    for ch in placement.chars() {
        if ch == '/' {
            rank_i += 1;
            file_i = 0;
        } else if ch.is_ascii_digit() {
            file_i += ch.to_digit(10).unwrap() as usize;
        } else if let Some(pi) = "PNBRQKpnbrqk".find(ch) {
            maps[pi][rank_i][file_i] = 1.0;
            file_i += 1;
        }
    }
    let side_value = if side == "w" { 1.0 } else { -1.0 };
    for r in 0..8 {
        for f in 0..8 {
            maps[12][r][f] = side_value;
        }
    }
    let owned_params;
    let params = match params {
        Some(params) => params,
        None => {
            owned_params = precompute_conv_params(channels, layers);
            &owned_params
        }
    };
    let mut prev = maps;
    for (layer, layer_params) in params.layers.iter().enumerate().take(layers) {
        let prev_channels = prev.len();
        debug_assert_eq!(
            layer_params.prev_channels, prev_channels,
            "cached conv layer {layer} channel mismatch"
        );
        let mut out = vec![[[0f32; 8]; 8]; channels];
        for c in 0..channels {
            for r in 0..8usize {
                for f in 0..8usize {
                    let mut acc = layer_params.biases[c];
                    for (pc, prev_map) in prev.iter().enumerate() {
                        for dri in 0..3usize {
                            let rr = r as isize + dri as isize - 1;
                            if !(0..8).contains(&rr) {
                                continue;
                            }
                            for dfi in 0..3usize {
                                let ff = f as isize + dfi as isize - 1;
                                if (0..8).contains(&ff) {
                                    let k = layer_params.kernels
                                        [kernel_index(c, prev_channels, pc, dri, dfi)];
                                    acc += prev_map[rr as usize][ff as usize] * k;
                                }
                            }
                        }
                    }
                    out[c][r][f] = (acc / ((prev_channels * 4) as f32).sqrt()).tanh();
                }
            }
        }
        prev = out;
    }
    let mut feats = Vec::with_capacity(2 + channels * 3);
    feats.push(1.0);
    feats.push(side_value);
    for channel in prev.iter() {
        let (mut sum, mut mx, mut mn) = (0f32, f32::NEG_INFINITY, f32::INFINITY);
        for row in channel {
            for &v in row {
                sum += v;
                mx = mx.max(v);
                mn = mn.min(v);
            }
        }
        feats.push(sum / 64.0);
        feats.push(mx);
        feats.push(mn);
    }
    feats
}

pub fn frozen_conv_student_features(fen: &str, channels: usize, layers: usize) -> Vec<f32> {
    conv_student_features(fen, channels, layers, None)
}

impl PositionEvaluator for StudentEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation {
        let fen = board_to_fen(board);
        let policy_features = if self.artifact.kind == "frozen_conv_fen_student"
            || self.artifact.kind == "frozen_conv_feature_mlp_student"
        {
            conv_student_features(
                &fen,
                self.artifact.conv_channels.unwrap_or(64),
                self.artifact.conv_layers.unwrap_or(6),
                self.conv_params.as_ref(),
            )
        } else {
            fen_features(&fen).unwrap().to_vec()
        };
        let value_features = if self.artifact.kind == "frozen_conv_fen_student"
            || self.artifact.kind == "frozen_conv_feature_mlp_student"
        {
            policy_features.clone()
        } else {
            wdl_features_from_fen(&fen).unwrap()
        };
        let (logits, wdl_logits): (Vec<f32>, Vec<f32>) = if self.artifact.kind
            == "tiny_board_residual_student"
            || self.artifact.architecture.as_deref() == Some("residual_tower")
        {
            panic!("Residual tower artifacts require the upcoming ONNX/runtime path");
        } else if self.artifact.kind == "tiny_board_cnn_student" {
            board_cnn_forward(&fen, &self.artifact)
        } else if self.artifact.kind == "frozen_conv_feature_mlp_student" {
            let w1 = self.artifact.w1.as_ref().unwrap();
            let b1 = self.artifact.b1.as_ref().unwrap();
            let mut hidden = vec![0.0; b1.len()];
            for h in 0..b1.len() {
                let mut acc = b1[h];
                for (f, &x) in policy_features.iter().enumerate() {
                    acc += x * w1.get(f).and_then(|row| row.get(h)).copied().unwrap_or(0.0);
                }
                hidden[h] = acc.max(0.0);
            }
            let pw = self.artifact.policy_w.as_ref().unwrap();
            let pb = self.artifact.policy_b.as_ref().unwrap();
            let ww = self.artifact.wdl_w.as_ref().unwrap();
            let wb = self.artifact.wdl_b.as_ref().unwrap();
            let pl = (0..pb.len())
                .map(|m| {
                    pb[m]
                        + hidden
                            .iter()
                            .enumerate()
                            .map(|(h, &x)| {
                                x * pw.get(h).and_then(|row| row.get(m)).copied().unwrap_or(0.0)
                            })
                            .sum::<f32>()
                })
                .collect();
            let wl = (0..wb.len())
                .map(|k| {
                    wb[k]
                        + hidden
                            .iter()
                            .enumerate()
                            .map(|(h, &x)| {
                                x * ww.get(h).and_then(|row| row.get(k)).copied().unwrap_or(0.0)
                            })
                            .sum::<f32>()
                })
                .collect();
            (pl, wl)
        } else {
            (
                self.artifact
                    .policy_weights
                    .iter()
                    .map(|w| dot(w, &policy_features))
                    .collect(),
                self.artifact
                    .wdl_weights
                    .iter()
                    .map(|w| dot(w, &value_features))
                    .collect(),
            )
        };
        let probs = softmax(&logits);
        let legal = legal_moves(board);
        let mut raw = Vec::with_capacity(legal.len());
        let mut legal_mass = 0f32;
        for m in legal.iter() {
            let idx = if self.artifact.policy_map.as_deref() == Some("uci_queen_knight_promo_v1") {
                move_to_fixed_policy_index(*m)
            } else {
                self.move_index.get(&move_to_uci(*m)).copied()
            };
            let p = idx
                .and_then(|i| probs.get(i))
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            raw.push(p);
            legal_mass += p;
        }
        let policy = if legal.is_empty() {
            Vec::new()
        } else if legal_mass <= 0.0 {
            legal
                .iter()
                .map(|&m| (move_to_action_id(m), 1.0 / legal.len() as f32))
                .collect()
        } else {
            legal
                .iter()
                .zip(raw.iter())
                .map(|(&m, &p)| (move_to_action_id(m), p / legal_mass))
                .collect()
        };
        let w = softmax(&wdl_logits);
        Evaluation::new(
            policy,
            [
                w.first().copied().unwrap_or(0.0),
                w.get(1).copied().unwrap_or(0.0),
                w.get(2).copied().unwrap_or(0.0),
            ],
        )
    }
}

pub struct UniformEvaluator;
impl PositionEvaluator for UniformEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation {
        let moves = legal_moves(board);
        let p = if moves.is_empty() {
            0.0
        } else {
            1.0 / moves.len() as f32
        };
        Evaluation::new(
            moves
                .into_iter()
                .map(|m| (move_to_action_id(m), p))
                .collect(),
            [0.25, 0.5, 0.25],
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPolicyMode {
    Classic,
    ActionValue,
    Aux,
}

impl SearchPolicyMode {
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "av" | "action_value" | "action-value" => Self::ActionValue,
            "aux" | "aux_puct" | "aux-puct" => Self::Aux,
            _ => Self::Classic,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchOptions {
    pub visits: u32,
    pub cpuct: f32,
    pub fpu: f32,
    pub temperature: f32,
    pub policy_mode: SearchPolicyMode,
    pub av_weight: f32,
    pub rank_weight: f32,
    pub regret_weight: f32,
    pub risk_weight: f32,
    pub uncertainty_weight: f32,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            visits: 8,
            cpuct: 1.5,
            fpu: 0.0,
            temperature: 1.0,
            policy_mode: SearchPolicyMode::Classic,
            av_weight: 0.25,
            rank_weight: 0.0,
            regret_weight: 0.0,
            risk_weight: 0.0,
            uncertainty_weight: 0.0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SearchPolicyEntry {
    pub mv: Move,
    pub visits: u32,
    pub prior: f32,
    pub q: f32,
    pub probability: f32,
}
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub mv: Option<Move>,
    pub visits: u32,
    pub value: f32,
    pub policy: Vec<SearchPolicyEntry>,
}
struct Edge {
    mv: Move,
    prior: f32,
    child: Option<Box<Node>>,
    visits: u32,
    value_sum: f32,
    action_value_prior: Option<f32>,
    rank_score: Option<f32>,
    regret: Option<f32>,
    risk: Option<f32>,
    uncertainty: Option<f32>,
}
struct Node {
    board: Board,
    history_fens: Vec<String>,
    expanded: bool,
    terminal_value: Option<f32>,
    edges: Vec<Edge>,
}
fn value_from_wdl(wdl: [f32; 3]) -> f32 {
    wdl[0] - wdl[2]
}
fn edge_q_for_parent(edge: &Edge, fpu: f32) -> f32 {
    if edge.visits > 0 {
        -edge.value_sum / edge.visits as f32
    } else {
        fpu
    }
}
fn edge_is_better_for_root(candidate: &Edge, incumbent: &Edge, fpu: f32) -> bool {
    if candidate.visits != incumbent.visits {
        return candidate.visits > incumbent.visits;
    }
    let cq = edge_q_for_parent(candidate, fpu);
    let iq = edge_q_for_parent(incumbent, fpu);
    match cq.total_cmp(&iq) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => candidate.prior > incumbent.prior,
    }
}
fn expand_node(node: &mut Node, evaluator: &dyn PositionEvaluator) -> f32 {
    let moves = legal_moves(&node.board);
    if moves.is_empty() {
        node.expanded = true;
        node.terminal_value = Some(if in_check(&node.board, node.board.turn) {
            -1.0
        } else {
            0.0
        });
        node.edges.clear();
        return node.terminal_value.unwrap();
    }
    let evaln = evaluator.evaluate_with_history(&node.board, &node.history_fens);
    let policy_map: HashMap<u32, f32> = evaln.policy.iter().copied().collect();
    let action_values: HashMap<u32, f32> = evaln
        .action_values
        .as_ref()
        .map(|xs| xs.iter().copied().collect())
        .unwrap_or_default();
    let rank_scores: HashMap<u32, f32> = evaln
        .rank_scores
        .as_ref()
        .map(|xs| xs.iter().copied().collect())
        .unwrap_or_default();
    let regrets: HashMap<u32, f32> = evaln
        .regrets
        .as_ref()
        .map(|xs| xs.iter().copied().collect())
        .unwrap_or_default();
    let risks: HashMap<u32, f32> = evaln
        .risks
        .as_ref()
        .map(|xs| xs.iter().copied().collect())
        .unwrap_or_default();
    let uncertainties: HashMap<u32, f32> = evaln
        .uncertainties
        .as_ref()
        .map(|xs| xs.iter().copied().collect())
        .unwrap_or_default();
    let raw: Vec<f32> = moves
        .iter()
        .map(|&m| {
            policy_map
                .get(&move_to_action_id(m))
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    let total: f32 = raw.iter().sum();
    let fallback = 1.0 / moves.len() as f32;
    node.edges = moves
        .into_iter()
        .enumerate()
        .map(|(i, mv)| {
            let action_id = move_to_action_id(mv);
            Edge {
                mv,
                prior: if total > 0.0 {
                    raw[i] / total
                } else {
                    fallback
                },
                child: None,
                visits: 0,
                value_sum: 0.0,
                action_value_prior: action_values.get(&action_id).copied(),
                rank_score: rank_scores.get(&action_id).copied(),
                regret: regrets.get(&action_id).copied(),
                risk: risks.get(&action_id).copied(),
                uncertainty: uncertainties.get(&action_id).copied(),
            }
        })
        .collect();
    node.expanded = true;
    node.terminal_value = None;
    value_from_wdl(evaln.wdl)
}
fn aux_edge_bonus(edge: &Edge, options: SearchOptions) -> f32 {
    let sv = 1.0 + edge.visits as f32;
    match options.policy_mode {
        SearchPolicyMode::Classic => 0.0,
        SearchPolicyMode::ActionValue => {
            options.av_weight * edge.action_value_prior.unwrap_or(0.0) / sv
        }
        SearchPolicyMode::Aux => {
            (options.av_weight * edge.action_value_prior.unwrap_or(0.0)
                + options.rank_weight * edge.rank_score.unwrap_or(0.0)
                - options.regret_weight * edge.regret.unwrap_or(0.0)
                - options.risk_weight * edge.risk.unwrap_or(0.0)
                + options.uncertainty_weight * edge.uncertainty.unwrap_or(0.0))
                / sv
        }
    }
}

fn simulate(node: &mut Node, evaluator: &dyn PositionEvaluator, options: SearchOptions) -> f32 {
    if !node.expanded {
        return expand_node(node, evaluator);
    }
    if let Some(v) = node.terminal_value {
        return v;
    }
    let parent_visits: u32 = node.edges.iter().map(|e| e.visits).sum();
    let sqrt_parent = ((parent_visits + 1) as f32).sqrt();
    let mut best_i = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, edge) in node.edges.iter().enumerate() {
        let score = edge_q_for_parent(edge, options.fpu)
            + options.cpuct * edge.prior * sqrt_parent / (1.0 + edge.visits as f32)
            + aux_edge_bonus(edge, options);
        if score > best_score {
            best_i = i;
            best_score = score;
        }
    }
    let parent_fen = board_to_fen(&node.board);
    let parent_history = node.history_fens.clone();
    let edge = &mut node.edges[best_i];
    if edge.child.is_none() {
        let mut history_fens = Vec::with_capacity(parent_history.len() + 1);
        history_fens.push(parent_fen);
        history_fens.extend(parent_history.into_iter().take(7));
        edge.child = Some(Box::new(Node {
            board: make_move(&node.board, edge.mv),
            history_fens,
            expanded: false,
            terminal_value: None,
            edges: Vec::new(),
        }));
    }
    let child_value = simulate(edge.child.as_mut().unwrap(), evaluator, options);
    edge.visits += 1;
    edge.value_sum += child_value;
    -child_value
}
fn visit_policy(edges: &[Edge], temperature: f32, fpu: f32) -> Vec<SearchPolicyEntry> {
    if edges.is_empty() {
        return Vec::new();
    }
    let tau = temperature.max(0.0);
    if tau == 0.0 {
        let mut best_i = 0usize;
        for i in 1..edges.len() {
            if edge_is_better_for_root(&edges[i], &edges[best_i], fpu) {
                best_i = i;
            }
        }
        return edges
            .iter()
            .enumerate()
            .map(|(i, e)| SearchPolicyEntry {
                mv: e.mv,
                visits: e.visits,
                prior: e.prior,
                q: edge_q_for_parent(e, fpu),
                probability: if i == best_i { 1.0 } else { 0.0 },
            })
            .collect();
    }
    let weights: Vec<f32> = edges
        .iter()
        .map(|e| (e.visits as f32).max(1e-9).powf(1.0 / tau))
        .collect();
    let total: f32 = weights.iter().sum::<f32>().max(1e-30);
    edges
        .iter()
        .zip(weights.iter())
        .map(|(e, &w)| SearchPolicyEntry {
            mv: e.mv,
            visits: e.visits,
            prior: e.prior,
            q: edge_q_for_parent(e, fpu),
            probability: w / total,
        })
        .collect()
}
pub fn search_root_with_history(
    board: &Board,
    evaluator: &dyn PositionEvaluator,
    options: SearchOptions,
    history_fens: &[String],
) -> SearchResult {
    let visits = options.visits.max(1);
    let mut root = Node {
        board: board.clone(),
        history_fens: history_fens.to_vec(),
        expanded: false,
        terminal_value: None,
        edges: Vec::new(),
    };
    let root_value = expand_node(&mut root, evaluator);
    if root.edges.is_empty() {
        return SearchResult {
            mv: None,
            visits: 0,
            value: root_value,
            policy: Vec::new(),
        };
    }
    for _ in 0..visits {
        simulate(&mut root, evaluator, options);
    }
    let policy = visit_policy(&root.edges, options.temperature, options.fpu);
    let mut best = policy.first();
    for entry in policy.iter().skip(1) {
        if best
            .map(|b| entry.probability > b.probability)
            .unwrap_or(true)
        {
            best = Some(entry);
        }
    }
    SearchResult {
        mv: best.map(|e| e.mv),
        visits: root.edges.iter().map(|e| e.visits).sum(),
        value: best.map(|e| e.q).unwrap_or(root_value),
        policy,
    }
}

pub fn search_root(
    board: &Board,
    evaluator: &dyn PositionEvaluator,
    options: SearchOptions,
) -> SearchResult {
    search_root_with_history(board, evaluator, options, &[])
}

#[no_mangle]
pub extern "C" fn tiny_leela_startpos_uniform_search_best_action(visits: u32) -> u32 {
    let board = parse_fen(START_FEN).expect("valid start fen");
    search_root(
        &board,
        &UniformEvaluator,
        SearchOptions {
            visits: visits.max(1),
            cpuct: 1.5,
            fpu: 0.0,
            temperature: 0.0,
            ..SearchOptions::default()
        },
    )
    .mv
    .map(move_to_action_id)
    .unwrap_or(u32::MAX)
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
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

unsafe fn wasm_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, ()> {
    if ptr.is_null() {
        return Err(());
    }
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
    let Ok(artifact_json) = wasm_str(artifact_ptr, artifact_len) else {
        return -1;
    };
    let Ok(fen) = wasm_str(fen_ptr, fen_len) else {
        return -2;
    };
    let Ok(evaluator) = StudentEvaluator::from_json(artifact_json) else {
        return -3;
    };
    let Ok(board) = parse_fen(fen) else {
        return -4;
    };
    if out_wdl_ptr.is_null() {
        return -5;
    }
    let evaln = evaluator.evaluate(&board);
    for i in 0..3 {
        *out_wdl_ptr.add(i) = evaln.wdl[i];
    }
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
    let Ok(artifact_json) = wasm_str(artifact_ptr, artifact_len) else {
        return u32::MAX;
    };
    let Ok(fen) = wasm_str(fen_ptr, fen_len) else {
        return u32::MAX;
    };
    let Ok(evaluator) = StudentEvaluator::from_json(artifact_json) else {
        return u32::MAX;
    };
    let Ok(board) = parse_fen(fen) else {
        return u32::MAX;
    };
    search_root(
        &board,
        &evaluator,
        SearchOptions {
            visits: visits.max(1),
            cpuct: 1.5,
            fpu: 0.0,
            temperature: 0.0,
            ..SearchOptions::default()
        },
    )
    .mv
    .map(move_to_action_id)
    .unwrap_or(u32::MAX)
}

#[no_mangle]
pub extern "C" fn tiny_leela_startpos_legal_count() -> u32 {
    legal_moves(&parse_fen(START_FEN).expect("valid start fen")).len() as u32
}

#[cfg(feature = "native-ort")]
#[derive(Clone, Debug, Deserialize)]
pub struct OnnxEvaluatorMeta {
    pub kind: String,
    pub architecture: String,
    pub policy_map: String,
    #[serde(default)]
    pub moves: Vec<String>,
    pub input_planes: usize,
    #[serde(default)]
    pub history_plies: usize,
    #[serde(default)]
    pub av_head_exported: bool,
    #[serde(default)]
    pub max_legal_moves: Option<usize>,
    #[serde(default)]
    pub onnx_fixed_legal_moves: Option<usize>,
    #[serde(default)]
    pub num_move_features: Option<usize>,
    #[serde(default)]
    pub allow_legal_overflow_zero_prior: bool,
}

#[cfg(feature = "native-ort")]
#[cfg(feature = "native-ort")]
#[derive(Clone, Debug, Deserialize)]
pub struct SquareFormerEvaluatorMeta {
    pub kind: String,
    #[serde(default)]
    pub input_dim: Option<usize>,
    #[serde(default)]
    pub token_features: Option<usize>,
    #[serde(default)]
    pub input_mode: Option<String>,
    #[serde(default)]
    pub input_format: Option<String>,
    pub policy_size: usize,
    #[serde(default)]
    pub history_plies: usize,
    #[serde(default)]
    pub av_head_exported: bool,
    #[serde(default)]
    pub max_legal_moves: Option<usize>,
    #[serde(default)]
    pub onnx_fixed_legal_moves: Option<usize>,
}

#[cfg(feature = "native-ort")]
pub struct OnnxEvaluator {
    session: std::sync::Mutex<ort::session::Session>,
    board_meta: Option<OnnxEvaluatorMeta>,
    square_meta: Option<SquareFormerEvaluatorMeta>,
}

#[cfg(feature = "native-ort")]
fn plane_index_for_sq(sq: u8) -> usize {
    let f = (sq % 8) as usize;
    let r = (sq / 8) as usize;
    (7 - r) * 8 + f
}

#[cfg(feature = "native-ort")]
fn add_board_piece_planes(data: &mut [f32], board: &Board, offset: usize, input_planes: usize) {
    for (sq, piece) in board.squares.iter().enumerate() {
        if let Some(piece) = piece {
            let p = offset + piece_index(*piece);
            if p < input_planes {
                data[p * 64 + plane_index_for_sq(sq as u8)] = 1.0;
            }
        }
    }
}

#[cfg(feature = "native-ort")]
fn onnx_input_planes(board: &Board, meta: &OnnxEvaluatorMeta, history_fens: &[String]) -> Vec<f32> {
    let input_planes = meta.input_planes;
    let mut data = vec![0.0f32; input_planes * 64];
    add_board_piece_planes(&mut data, board, 0, input_planes);
    for h in 0..meta.history_plies.min(history_fens.len()) {
        if let Ok(hist) = parse_fen(&history_fens[h]) {
            add_board_piece_planes(&mut data, &hist, 12 * (h + 1), input_planes);
        }
    }
    let state0 = 12 * (meta.history_plies + 1);
    let fill_plane = |data: &mut [f32], p: usize, v: f32| {
        if p < input_planes {
            data[p * 64..(p + 1) * 64].fill(v);
        }
    };
    fill_plane(
        &mut data,
        state0,
        if board.turn == Color::White {
            1.0
        } else {
            -1.0
        },
    );
    if input_planes.saturating_sub(state0) >= 10 {
        for (flag, p) in [
            (1u8, state0 + 1),
            (2, state0 + 2),
            (4, state0 + 3),
            (8, state0 + 4),
        ] {
            if board.castling & flag != 0 {
                fill_plane(&mut data, p, 1.0);
            }
        }
        if let Some(ep) = board.ep_square {
            data[(state0 + 5) * 64 + plane_index_for_sq(ep)] = 1.0;
        }
        fill_plane(&mut data, state0 + 6, 1.0);
        fill_plane(
            &mut data,
            state0 + 7,
            if board.turn == Color::White { 1.0 } else { 0.0 },
        );
    } else {
        fill_plane(&mut data, state0 + 1, 1.0);
    }
    data
}

#[cfg(feature = "native-ort")]
pub fn encode_onnx_input_planes(
    board: &Board,
    meta: &OnnxEvaluatorMeta,
    history_fens: &[String],
) -> Vec<f32> {
    onnx_input_planes(board, meta, history_fens)
}

#[cfg(feature = "native-ort")]
fn move_to_chessbench_av_class(m: Move) -> i64 {
    let ft = m.from as i64 * 64 + m.to as i64;
    match m.promotion {
        None => ft,
        Some(Role::Knight) => 4096 + ft * 4,
        Some(Role::Bishop) => 4096 + ft * 4 + 1,
        Some(Role::Rook) => 4096 + ft * 4 + 2,
        Some(Role::Queen) => 4096 + ft * 4 + 3,
        Some(_) => ft,
    }
}

#[cfg(feature = "native-ort")]
fn role_index(role: Role) -> f32 {
    match role {
        Role::Pawn => 1.0,
        Role::Knight => 2.0,
        Role::Bishop => 3.0,
        Role::Rook => 4.0,
        Role::Queen => 5.0,
        Role::King => 6.0,
    }
}

#[cfg(feature = "native-ort")]
fn piece_value(role: Role) -> f32 {
    match role {
        Role::Pawn => 1.0,
        Role::Knight => 3.0,
        Role::Bishop => 3.0,
        Role::Rook => 5.0,
        Role::Queen => 9.0,
        Role::King => 0.0,
    }
}

#[cfg(feature = "native-ort")]
fn promo_index(role: Role) -> f32 {
    match role {
        Role::Knight => 1.0,
        Role::Bishop => 2.0,
        Role::Rook => 3.0,
        Role::Queen => 4.0,
        _ => 0.0,
    }
}

#[cfg(feature = "native-ort")]
fn chebyshev(a: u8, b: u8) -> f32 {
    ((file(a) - file(b)).abs()).max((rank(a) - rank(b)).abs()) as f32
}

#[cfg(feature = "native-ort")]
fn moveformer_legal_inputs(
    board: &Board,
    width: usize,
    feature_count: usize,
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    let moves = legal_moves(board);
    let mut action_ids = vec![20480i64; width];
    let mut features = vec![0.0f32; width * feature_count];
    let mut mask = vec![0.0f32; width];
    let own_king =
        king_square(board, board.turn).unwrap_or(if board.turn == Color::White { 4 } else { 60 });
    let enemy_king = king_square(board, board.turn.opposite()).unwrap_or(
        if board.turn.opposite() == Color::White {
            4
        } else {
            60
        },
    );
    for (j, mv) in moves.iter().take(width).enumerate() {
        action_ids[j] = move_to_action_id(*mv) as i64;
        mask[j] = 1.0;
        let base = j * feature_count;
        let moving = board.squares[mv.from as usize];
        let captured = board.squares[mv.to as usize];
        let moving_role = moving.map(|p| p.role);
        let captured_role = captured.map(|p| p.role);
        let promo = mv.promotion.map(promo_index).unwrap_or(0.0);
        if feature_count > 0 {
            features[base] = moving_role.map(role_index).unwrap_or(0.0);
        }
        if feature_count > 1 {
            features[base + 1] = captured_role.map(role_index).unwrap_or(0.0);
        }
        if feature_count > 2 {
            features[base + 2] = promo;
        }
        if feature_count > 3 {
            features[base + 3] = if captured.is_some() { 1.0 } else { 0.0 };
        }
        if feature_count > 6 {
            features[base + 6] = if promo != 0.0 { 1.0 } else { 0.0 };
        }
        if feature_count > 14 {
            features[base + 14] = moving_role.map(piece_value).unwrap_or(0.0);
        }
        let captured_value = captured_role.map(piece_value).unwrap_or(0.0);
        let promo_value = mv.promotion.map(|r| piece_value(r) - 1.0).unwrap_or(0.0);
        if feature_count > 15 {
            features[base + 15] = captured_value;
        }
        if feature_count > 16 {
            features[base + 16] = captured_value + promo_value;
        }
        if feature_count > 18 {
            features[base + 18] = chebyshev(mv.to, enemy_king);
        }
        if feature_count > 19 {
            features[base + 19] = chebyshev(mv.to, own_king);
        }
    }
    (moves, action_ids, features, mask)
}

#[cfg(feature = "native-ort")]
pub fn encode_moveformer_legal_inputs(
    board: &Board,
    width: usize,
    feature_count: usize,
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    moveformer_legal_inputs(board, width, feature_count)
}

#[cfg(feature = "native-ort")]
fn squareformer_piece_id(piece: Option<Piece>) -> i64 {
    let Some(piece) = piece else { return 0 };
    let base = match piece.role {
        Role::Pawn => 1,
        Role::Knight => 2,
        Role::Bishop => 3,
        Role::Rook => 4,
        Role::Queen => 5,
        Role::King => 6,
    };
    base + if piece.color == Color::White { 0 } else { 6 }
}

#[cfg(feature = "native-ort")]
fn squareformer_castle_mask(castling: u8) -> i64 {
    castling as i64
}

#[cfg(feature = "native-ort")]
pub fn is_squareformer_compact_meta(meta: &SquareFormerEvaluatorMeta) -> bool {
    meta.input_mode.as_deref() == Some("embedding")
        || meta.input_format.as_deref() == Some("compact_uint8_embeddings")
        || meta.input_format.as_deref() == Some("compact_uint8_tokens")
}

#[cfg(feature = "native-ort")]
pub fn encode_squareformer_compact_input(
    board: &Board,
    meta: &SquareFormerEvaluatorMeta,
    history_fens: &[String],
) -> Vec<i64> {
    let history = meta.history_plies;
    let stride = meta.token_features.unwrap_or(history + 9);
    let mut data = vec![0i64; 64 * stride];
    for sq in 0..64usize {
        data[sq * stride] = squareformer_piece_id(board.squares[sq]);
    }
    for h in 0..history.min(history_fens.len()) {
        if let Ok(hist) = parse_fen(&history_fens[h]) {
            for sq in 0..64usize {
                data[sq * stride + h + 1] = squareformer_piece_id(hist.squares[sq]);
            }
        }
    }
    let base = history + 1;
    let stm = if board.turn == Color::White { 1 } else { 2 };
    let flags = squareformer_castle_mask(board.castling);
    let half = board.halfmove.min(255) as i64;
    for sq in 0..64usize {
        let rank = sq / 8;
        let file = sq % 8;
        let row = sq * stride;
        if base < stride {
            data[row + base] = stm;
        }
        if base + 1 < stride {
            data[row + base + 1] = flags;
        }
        if base + 2 < stride {
            data[row + base + 2] = if board.ep_square == Some(sq as u8) {
                1
            } else {
                0
            };
        }
        if base + 3 < stride {
            data[row + base + 3] = half;
        }
        if base + 4 < stride {
            data[row + base + 4] = rank as i64;
        }
        if base + 5 < stride {
            data[row + base + 5] = file as i64;
        }
        if base + 6 < stride {
            data[row + base + 6] = ((rank + file) & 1) as i64;
        }
        if base + 7 < stride {
            data[row + base + 7] = sq as i64;
        }
    }
    data
}

#[cfg(feature = "native-ort")]
pub fn encode_squareformer_float_input(
    board: &Board,
    meta: &SquareFormerEvaluatorMeta,
    history_fens: &[String],
) -> Vec<f32> {
    let history = meta.history_plies;
    let planes_per_board = 13usize;
    let input_dim = meta
        .input_dim
        .unwrap_or((history + 1) * planes_per_board + 8);
    let mut data = vec![0.0f32; 64 * input_dim];
    for sq in 0..64usize {
        let pid = squareformer_piece_id(board.squares[sq]) as usize;
        data[sq * input_dim + pid] = 1.0;
    }
    for h in 0..history.min(history_fens.len()) {
        if let Ok(hist) = parse_fen(&history_fens[h]) {
            let offset = (h + 1) * planes_per_board;
            for sq in 0..64usize {
                let pid = squareformer_piece_id(hist.squares[sq]) as usize;
                if offset + pid < input_dim {
                    data[sq * input_dim + offset + pid] = 1.0;
                }
            }
        }
    }
    let base = (history + 1) * planes_per_board;
    for sq in 0..64usize {
        let row = sq * input_dim;
        if base < input_dim {
            data[row + base] = if board.turn == Color::White { 1.0 } else { 0.0 };
        }
        if base + 1 < input_dim {
            data[row + base + 1] = if board.turn == Color::Black { 1.0 } else { 0.0 };
        }
        for (i, flag) in [1u8, 2, 4, 8].into_iter().enumerate() {
            if base + 2 + i < input_dim && board.castling & flag != 0 {
                data[row + base + 2 + i] = 1.0;
            }
        }
        if base + 7 < input_dim {
            data[row + base + 7] = board.halfmove.min(255) as f32 / 100.0;
        }
    }
    if let Some(ep) = board.ep_square {
        let row = ep as usize * input_dim;
        if base + 6 < input_dim {
            data[row + base + 6] = 1.0;
        }
    }
    data
}

#[cfg(feature = "native-ort")]
pub fn encode_squareformer_legal_ids(board: &Board, width: usize) -> (Vec<Move>, Vec<i64>) {
    let moves = legal_moves(board);
    let mut ids = vec![0i64; width];
    for (j, mv) in moves.iter().take(width).enumerate() {
        ids[j] = move_to_chessbench_av_class(*mv);
    }
    (moves, ids)
}

#[cfg(feature = "native-ort")]
fn optional_output_vec(
    outputs: &ort::session::SessionOutputs,
    names: &[&str],
) -> Result<Option<Vec<f32>>, String> {
    for name in names {
        if let Some(value) = outputs.get(*name) {
            return Ok(Some(
                value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?
                    .1
                    .to_vec(),
            ));
        }
    }
    Ok(None)
}

#[cfg(feature = "native-ort")]
fn aux_pairs_for_moves(
    moves: &[Move],
    raw: Option<&[f32]>,
    width: usize,
) -> Option<Vec<(u32, f32)>> {
    let raw = raw?;
    let n = moves.len().min(width);
    let mut out = Vec::with_capacity(n);
    for (j, &mv) in moves.iter().take(n).enumerate() {
        out.push((move_to_action_id(mv), raw.get(j).copied().unwrap_or(0.0)));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(feature = "native-ort")]
impl OnnxEvaluator {
    pub fn from_files(model_path: &str, meta_path: &str) -> Result<Self, String> {
        let meta_json = std::fs::read_to_string(meta_path).map_err(|e| e.to_string())?;
        let meta_value: serde_json::Value =
            serde_json::from_str(&meta_json).map_err(|e| e.to_string())?;
        let kind = meta_value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let (board_meta, square_meta) = if kind == "squareformer" || kind == "squareformer_v2" {
            let meta: SquareFormerEvaluatorMeta =
                serde_json::from_value(meta_value).map_err(|e| e.to_string())?;
            (None, Some(meta))
        } else {
            let meta: OnnxEvaluatorMeta =
                serde_json::from_value(meta_value).map_err(|e| e.to_string())?;
            if meta.policy_map != "uci_queen_knight_promo_v1" {
                return Err(format!("unsupported policy_map {}", meta.policy_map));
            }
            (Some(meta), None)
        };
        let mut builder = ort::session::Session::builder().map_err(|e| e.to_string())?;
        if let Ok(t) =
            std::env::var("ORT_INTRA_OP_NUM_THREADS").or_else(|_| std::env::var("ORT_NUM_THREADS"))
        {
            if let Ok(n) = t.parse::<usize>() {
                if n > 0 {
                    builder = builder.with_intra_threads(n).map_err(|e| e.to_string())?;
                }
            }
        }
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            session: std::sync::Mutex::new(session),
            board_meta,
            square_meta,
        })
    }

    fn eval_squareformer_onnx(
        &self,
        board: &Board,
        meta: &SquareFormerEvaluatorMeta,
        history_fens: &[String],
    ) -> Result<Evaluation, String> {
        let compact = is_squareformer_compact_meta(meta);
        let stride = if compact {
            meta.token_features.unwrap_or(meta.history_plies + 9)
        } else {
            meta.input_dim.unwrap_or((meta.history_plies + 1) * 13 + 8)
        };
        let mut session = self
            .session
            .lock()
            .map_err(|_| "onnx session mutex poisoned".to_string())?;
        let outputs = if compact {
            let tokens = encode_squareformer_compact_input(board, meta, history_fens);
            let tokens_t =
                ort::value::Tensor::<i64>::from_array(([1usize, 64usize, stride], tokens))
                    .map_err(|e| e.to_string())?;
            if meta.av_head_exported {
                let width = meta
                    .onnx_fixed_legal_moves
                    .or(meta.max_legal_moves)
                    .unwrap_or(128)
                    .max(1);
                let (_moves, ids) = encode_squareformer_legal_ids(board, width);
                let ids_t = ort::value::Tensor::<i64>::from_array(([1usize, width], ids))
                    .map_err(|e| e.to_string())?;
                session
                    .run(ort::inputs!["tokens" => tokens_t, "legal_action_ids" => ids_t])
                    .map_err(|e| e.to_string())?
            } else {
                session
                    .run(ort::inputs!["tokens" => tokens_t])
                    .map_err(|e| e.to_string())?
            }
        } else {
            let tokens = encode_squareformer_float_input(board, meta, history_fens);
            let tokens_t =
                ort::value::Tensor::<f32>::from_array(([1usize, 64usize, stride], tokens))
                    .map_err(|e| e.to_string())?;
            if meta.av_head_exported {
                let width = meta
                    .onnx_fixed_legal_moves
                    .or(meta.max_legal_moves)
                    .unwrap_or(128)
                    .max(1);
                let (_moves, ids) = encode_squareformer_legal_ids(board, width);
                let ids_t = ort::value::Tensor::<i64>::from_array(([1usize, width], ids))
                    .map_err(|e| e.to_string())?;
                session
                    .run(ort::inputs!["tokens" => tokens_t, "legal_action_ids" => ids_t])
                    .map_err(|e| e.to_string())?
            } else {
                session
                    .run(ort::inputs!["tokens" => tokens_t])
                    .map_err(|e| e.to_string())?
            }
        };
        let wdl_raw = outputs
            .get("wdl")
            .ok_or("missing wdl")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?
            .1
            .to_vec();
        let w = softmax(&wdl_raw[..wdl_raw.len().min(3)]);
        let logits = outputs
            .get("policy")
            .ok_or("missing policy")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?
            .1
            .to_vec();
        let legal = legal_moves(board);
        let legal_logits: Vec<f32> = legal
            .iter()
            .map(|&m| {
                logits
                    .get(move_to_chessbench_av_class(m) as usize)
                    .copied()
                    .unwrap_or(-100.0)
            })
            .collect();
        let probs = softmax(&legal_logits);
        let policy = legal
            .iter()
            .zip(probs.iter())
            .map(|(&m, &p)| (move_to_action_id(m), p))
            .collect();
        let av_width = meta
            .onnx_fixed_legal_moves
            .or(meta.max_legal_moves)
            .unwrap_or(legal.len())
            .max(1);
        let action_values = aux_pairs_for_moves(
            &legal,
            optional_output_vec(&outputs, &["action_values"])?.as_deref(),
            av_width,
        );
        let mut eval = Evaluation::new(
            policy,
            [
                w.first().copied().unwrap_or(0.0),
                w.get(1).copied().unwrap_or(0.0),
                w.get(2).copied().unwrap_or(0.0),
            ],
        );
        eval.action_values = action_values;
        Ok(eval)
    }

    fn eval_onnx(&self, board: &Board, history_fens: &[String]) -> Result<Evaluation, String> {
        if let Some(meta) = &self.square_meta {
            return self.eval_squareformer_onnx(board, meta, history_fens);
        }
        let meta = self
            .board_meta
            .as_ref()
            .ok_or("missing board ONNX metadata")?;
        let planes = onnx_input_planes(board, meta, history_fens);
        let planes_t =
            ort::value::Tensor::<f32>::from_array(([1usize, meta.input_planes, 8, 8], planes))
                .map_err(|e| e.to_string())?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| "onnx session mutex poisoned".to_string())?;
        let outputs = if meta.architecture == "cnn_move_token_transformer"
            || meta.architecture == "cnn_square_move_transformer"
        {
            let width = meta
                .onnx_fixed_legal_moves
                .or(meta.max_legal_moves)
                .unwrap_or(128)
                .max(1);
            let feature_count = meta.num_move_features.unwrap_or(20).max(1);
            let (_moves, action_ids, features, mask) =
                moveformer_legal_inputs(board, width, feature_count);
            let action_t = ort::value::Tensor::<i64>::from_array(([1usize, width], action_ids))
                .map_err(|e| e.to_string())?;
            let feat_t =
                ort::value::Tensor::<f32>::from_array(([1usize, width, feature_count], features))
                    .map_err(|e| e.to_string())?;
            let mask_t = ort::value::Tensor::<f32>::from_array(([1usize, width], mask))
                .map_err(|e| e.to_string())?;
            session.run(ort::inputs!["planes" => planes_t, "legal_action_ids" => action_t, "legal_features" => feat_t, "legal_mask" => mask_t]).map_err(|e| e.to_string())?
        } else if meta.av_head_exported {
            let moves = legal_moves(board);
            let width = moves.len().max(1);
            let classes: Vec<i64> = moves
                .iter()
                .map(|&m| move_to_chessbench_av_class(m))
                .collect();
            let cand_t = ort::value::Tensor::<i64>::from_array(([1usize, width], classes))
                .map_err(|e| e.to_string())?;
            session
                .run(ort::inputs!["planes" => planes_t, "candidate_moves" => cand_t])
                .map_err(|e| e.to_string())?
        } else {
            session
                .run(ort::inputs!["planes" => planes_t])
                .map_err(|e| e.to_string())?
        };
        let wdl_raw = outputs
            .get("wdl_logits")
            .ok_or("missing wdl_logits")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?
            .1
            .to_vec();
        let w = softmax(&wdl_raw[..wdl_raw.len().min(3)]);
        let legal = legal_moves(board);
        if meta.architecture == "cnn_move_token_transformer"
            || meta.architecture == "cnn_square_move_transformer"
        {
            let width = meta
                .onnx_fixed_legal_moves
                .or(meta.max_legal_moves)
                .unwrap_or(128)
                .max(1);
            if legal.len() > width && !meta.allow_legal_overflow_zero_prior {
                return Err(format!(
                    "legal move overflow: model width={width} legal={}",
                    legal.len()
                ));
            }
            let logits = outputs
                .get("policy_logits_legal")
                .ok_or("missing policy_logits_legal")?
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?
                .1
                .to_vec();
            let n = legal.len().min(width);
            let probs = softmax(&logits[..n]);
            let mut policy: Vec<(u32, f32)> = legal
                .iter()
                .take(n)
                .zip(probs.iter())
                .map(|(&m, &p)| (move_to_action_id(m), p))
                .collect();
            for &m in legal.iter().skip(n) {
                policy.push((move_to_action_id(m), 0.0));
            }
            let action_values = aux_pairs_for_moves(
                &legal,
                optional_output_vec(
                    &outputs,
                    &["action_values", "action_value", "action_value_logits"],
                )?
                .as_deref(),
                width,
            );
            let rank_scores = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["rank_scores"])?.as_deref(),
                width,
            );
            let regrets = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["regrets"])?.as_deref(),
                width,
            );
            let risks = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["risks"])?.as_deref(),
                width,
            );
            let uncertainties = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["uncertainties"])?.as_deref(),
                width,
            );
            let mut eval = Evaluation::new(
                policy,
                [
                    w.first().copied().unwrap_or(0.0),
                    w.get(1).copied().unwrap_or(0.0),
                    w.get(2).copied().unwrap_or(0.0),
                ],
            );
            eval.action_values = action_values;
            eval.rank_scores = rank_scores;
            eval.regrets = regrets;
            eval.risks = risks;
            eval.uncertainties = uncertainties;
            Ok(eval)
        } else {
            let logits = outputs
                .get("policy_logits")
                .ok_or("missing policy_logits")?
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?
                .1
                .to_vec();
            let probs = softmax(&logits);
            let raw: Vec<f32> = legal
                .iter()
                .map(|&m| {
                    move_to_fixed_policy_index(m)
                        .and_then(|i| probs.get(i))
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0)
                })
                .collect();
            let total: f32 = raw.iter().sum();
            let fallback = if legal.is_empty() {
                0.0
            } else {
                1.0 / legal.len() as f32
            };
            let policy = legal
                .iter()
                .zip(raw.iter())
                .map(|(&m, &p)| {
                    (
                        move_to_action_id(m),
                        if total > 0.0 { p / total } else { fallback },
                    )
                })
                .collect();
            let width = legal.len().max(1);
            let action_values = aux_pairs_for_moves(
                &legal,
                optional_output_vec(
                    &outputs,
                    &["action_values", "action_value", "action_value_logits"],
                )?
                .as_deref(),
                width,
            );
            let rank_scores = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["rank_scores"])?.as_deref(),
                width,
            );
            let regrets = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["regrets"])?.as_deref(),
                width,
            );
            let risks = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["risks"])?.as_deref(),
                width,
            );
            let uncertainties = aux_pairs_for_moves(
                &legal,
                optional_output_vec(&outputs, &["uncertainties"])?.as_deref(),
                width,
            );
            let mut eval = Evaluation::new(
                policy,
                [
                    w.first().copied().unwrap_or(0.0),
                    w.get(1).copied().unwrap_or(0.0),
                    w.get(2).copied().unwrap_or(0.0),
                ],
            );
            eval.action_values = action_values;
            eval.rank_scores = rank_scores;
            eval.regrets = regrets;
            eval.risks = risks;
            eval.uncertainties = uncertainties;
            Ok(eval)
        }
    }
}

#[cfg(feature = "native-ort")]
impl PositionEvaluator for OnnxEvaluator {
    fn evaluate(&self, board: &Board) -> Evaluation {
        self.eval_onnx(board, &[]).expect("ONNX evaluation failed")
    }
    fn evaluate_with_history(&self, board: &Board, history_fens: &[String]) -> Evaluation {
        self.eval_onnx(board, history_fens)
            .expect("ONNX evaluation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn start_position_has_20_legal_moves() {
        assert_eq!(legal_moves(&parse_fen(START_FEN).unwrap()).len(), 20);
    }
    #[test]
    fn fen_roundtrip_start() {
        assert_eq!(board_to_fen(&parse_fen(START_FEN).unwrap()), START_FEN);
    }
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
        let ep = make_move(
            &parse_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1").unwrap(),
            Move {
                from: 36,
                to: 43,
                promotion: None,
            },
        );
        assert!(ep.squares[35].is_none());
        assert_eq!(
            ep.squares[43],
            Some(Piece {
                color: Color::White,
                role: Role::Pawn
            })
        );
    }
}
