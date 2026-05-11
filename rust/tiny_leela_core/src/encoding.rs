use serde::Deserialize;

use crate::board::{file, idx, on, rank, KING, KNIGHT};
use crate::eval::piece_index;
use crate::{
    in_check, king_square, legal_moves, make_move, move_to_action_id, parse_fen, Board, Color,
    Move, Piece, Role,
};

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

fn plane_index_for_sq(sq: u8) -> usize {
    let f = (sq % 8) as usize;
    let r = (sq / 8) as usize;
    (7 - r) * 8 + f
}

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

pub(crate) fn onnx_input_planes(
    board: &Board,
    meta: &OnnxEvaluatorMeta,
    history_fens: &[String],
) -> Vec<f32> {
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

pub fn encode_onnx_input_planes(
    board: &Board,
    meta: &OnnxEvaluatorMeta,
    history_fens: &[String],
) -> Vec<f32> {
    onnx_input_planes(board, meta, history_fens)
}

pub(crate) fn move_to_chessbench_av_class(m: Move) -> i64 {
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

fn promo_index(role: Role) -> f32 {
    match role {
        Role::Knight => 1.0,
        Role::Bishop => 2.0,
        Role::Rook => 3.0,
        Role::Queen => 4.0,
        _ => 0.0,
    }
}

fn chebyshev(a: u8, b: u8) -> f32 {
    ((file(a) - file(b)).abs()).max((rank(a) - rank(b)).abs()) as f32
}

fn captured_piece_for_move(board: &Board, mv: Move) -> Option<Piece> {
    let moving = board.squares[mv.from as usize]?;
    let direct = board.squares[mv.to as usize];
    if moving.role == Role::Pawn
        && Some(mv.to) == board.ep_square
        && direct.is_none()
        && file(mv.from) != file(mv.to)
    {
        board.squares[idx(file(mv.to), rank(mv.from)) as usize]
    } else {
        direct
    }
}

fn ray_clear(board: &Board, from: u8, to: u8, df: i8, dr: i8) -> bool {
    let (mut f, mut r) = (file(from) + df, rank(from) + dr);
    while on(f, r) {
        let sq = idx(f, r);
        if sq == to {
            return true;
        }
        if board.squares[sq as usize].is_some() {
            return false;
        }
        f += df;
        r += dr;
    }
    false
}

fn piece_attacks_square(board: &Board, from: u8, piece: Piece, to: u8) -> bool {
    let df = file(to) - file(from);
    let dr = rank(to) - rank(from);
    match piece.role {
        Role::Pawn => dr == if piece.color == Color::White { 1 } else { -1 } && df.abs() == 1,
        Role::Knight => KNIGHT.iter().any(|&(f, r)| f == df && r == dr),
        Role::King => KING.iter().any(|&(f, r)| f == df && r == dr),
        Role::Bishop => {
            df.abs() == dr.abs() && df != 0 && ray_clear(board, from, to, df.signum(), dr.signum())
        }
        Role::Rook => {
            (df == 0) ^ (dr == 0)
                && (df != 0 || dr != 0)
                && ray_clear(board, from, to, df.signum(), dr.signum())
        }
        Role::Queen => {
            let bishop_like = df.abs() == dr.abs() && df != 0;
            let rook_like = (df == 0) ^ (dr == 0);
            (bishop_like || rook_like) && ray_clear(board, from, to, df.signum(), dr.signum())
        }
    }
}

fn attacker_count(board: &Board, color: Color, sq: u8) -> usize {
    board
        .squares
        .iter()
        .enumerate()
        .filter(|(from, piece)| {
            piece
                .map(|p| p.color == color && piece_attacks_square(board, *from as u8, p, sq))
                .unwrap_or(false)
        })
        .count()
}

fn attacker_counts_by_role(board: &Board, color: Color, sq: u8) -> [usize; 6] {
    let mut counts = [0usize; 6];
    for (from, piece) in board.squares.iter().enumerate() {
        let Some(piece) = *piece else {
            continue;
        };
        if piece.color == color && piece_attacks_square(board, from as u8, piece, sq) {
            counts[piece.role as usize] += 1;
        }
    }
    counts
}

fn least_attacker_value(board: &Board, color: Color, sq: u8) -> f32 {
    board
        .squares
        .iter()
        .enumerate()
        .filter_map(|(from, piece)| {
            let piece = (*piece)?;
            if piece.color == color && piece_attacks_square(board, from as u8, piece, sq) {
                Some(piece_value(piece.role))
            } else {
                None
            }
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

fn is_pinned_to_king(board: &Board, color: Color, sq: u8) -> bool {
    let Some(piece) = board.squares[sq as usize] else {
        return false;
    };
    if piece.color != color || piece.role == Role::King {
        return false;
    }
    let Some(king) = king_square(board, color) else {
        return false;
    };
    let mut without = board.clone();
    without.squares[sq as usize] = None;
    crate::is_square_attacked(&without, king, color.opposite())
}

pub const MOVEFORMER_BASE_FEATURE_NAMES: [&str; 20] = [
    "moving_piece_type",
    "captured_piece_type",
    "promotion_type",
    "is_capture",
    "is_check",
    "is_castle",
    "is_promotion",
    "is_en_passant",
    "from_attacked_by_enemy_pre",
    "from_defended_by_own_pre",
    "to_attacked_by_enemy_after",
    "to_defended_by_own_after",
    "to_enemy_attackers_after_capped8",
    "to_own_defenders_after_capped8",
    "moving_piece_value",
    "captured_piece_value",
    "material_delta",
    "from_piece_pinned_pre",
    "king_distance_to_enemy_after",
    "king_distance_to_own_after",
];

pub const MOVEFORMER_ATTACK_FEATURE_NAMES: [&str; 19] = [
    "from_enemy_attackers_count_pre_capped8",
    "from_own_defenders_count_pre_capped8",
    "to_enemy_attackers_after_pawn",
    "to_enemy_attackers_after_knight",
    "to_enemy_attackers_after_bishop",
    "to_enemy_attackers_after_rook",
    "to_enemy_attackers_after_queen",
    "to_enemy_attackers_after_king",
    "to_own_defenders_after_pawn",
    "to_own_defenders_after_knight",
    "to_own_defenders_after_bishop",
    "to_own_defenders_after_rook",
    "to_own_defenders_after_queen",
    "to_own_defenders_after_king",
    "to_lva_enemy_after",
    "to_lvd_own_after",
    "to_see_lite_after",
    "moved_piece_hanging_after",
    "moved_piece_defended_after",
];

pub const MOVEFORMER_DELTA_FEATURE_NAMES: [&str; 12] = [
    "queen_lost_immediately_after",
    "own_queen_en_prise_after",
    "enemy_can_capture_own_queen_after",
    "enemy_can_capture_moved_piece_after",
    "capture_value_minus_moved_value",
    "promotion_gain",
    "material_delta_after_move_signed",
    "own_legal_moves_pre_capped64",
    "enemy_replies_after_capped64",
    "is_quiet_hanging_move",
    "is_capture_to_undefended_square",
    "is_sacrifice_like",
];

pub const MOVEFORMER_RAY_FEATURE_NAMES: [&str; 12] = [
    "own_slider_attackers_enemy_king_pre",
    "own_slider_attackers_enemy_king_after",
    "own_slider_attackers_enemy_king_delta",
    "own_slider_attackers_enemy_queen_pre",
    "own_slider_attackers_enemy_queen_after",
    "own_slider_attackers_enemy_queen_delta",
    "enemy_slider_attackers_own_king_after",
    "enemy_slider_attackers_own_queen_after",
    "moved_piece_slider_pressure_enemy_king_after",
    "moved_piece_slider_pressure_enemy_queen_after",
    "move_opens_line_from_own_rook_bishop_queen",
    "queen_exposed_to_slider_after",
];

pub const MOVEFORMER_KINGZONE_FEATURE_NAMES: [&str; 12] = [
    "move_to_enemy_king_zone",
    "move_from_own_king_zone",
    "capture_in_enemy_king_zone",
    "check_or_adjacent_enemy_king",
    "own_attacks_enemy_king_zone_pre_capped16",
    "own_attacks_enemy_king_zone_after_capped16",
    "own_attacks_enemy_king_zone_delta",
    "enemy_attacks_own_king_zone_pre_capped16",
    "enemy_attacks_own_king_zone_after_capped16",
    "enemy_attacks_own_king_zone_delta",
    "enemy_king_escape_squares_after_capped8",
    "own_king_escape_squares_after_capped8",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveFormerFeatureGroup {
    Base,
    Attack,
    Delta,
    Ray,
    KingZone,
}

fn base_moveformer_feature(board: &Board, mv: Move) -> [f32; 20] {
    let moving = board.squares[mv.from as usize];
    let moved_color = moving.map(|p| p.color).unwrap_or(board.turn);
    let enemy = moved_color.opposite();
    let moving_role = moving.map(|p| p.role);
    let captured = captured_piece_for_move(board, mv);
    let captured_role = captured.map(|p| p.role);
    let promo = mv.promotion.map(promo_index).unwrap_or(0.0);
    let is_ep = moving_role == Some(Role::Pawn)
        && Some(mv.to) == board.ep_square
        && captured.is_some()
        && board.squares[mv.to as usize].is_none()
        && file(mv.from) != file(mv.to);
    let from_enemy = attacker_count(board, enemy, mv.from);
    let from_own = attacker_count(board, moved_color, mv.from);
    let is_castle = moving_role == Some(Role::King) && (file(mv.to) - file(mv.from)).abs() == 2;
    let after = make_move(board, mv);
    let gives_check = in_check(&after, enemy);
    let to_enemy = attacker_count(&after, enemy, mv.to);
    let to_own = attacker_count(&after, moved_color, mv.to);
    let own_king = king_square(&after, moved_color).unwrap_or(if moved_color == Color::White {
        4
    } else {
        60
    });
    let enemy_king =
        king_square(&after, enemy).unwrap_or(if enemy == Color::White { 4 } else { 60 });
    let captured_value = captured_role.map(piece_value).unwrap_or(0.0);
    let promo_gain = mv.promotion.map(|r| piece_value(r) - 1.0).unwrap_or(0.0);
    [
        moving_role.map(role_index).unwrap_or(0.0),
        captured_role.map(role_index).unwrap_or(0.0),
        promo,
        if captured.is_some() || is_ep {
            1.0
        } else {
            0.0
        },
        if gives_check { 1.0 } else { 0.0 },
        if is_castle { 1.0 } else { 0.0 },
        if promo != 0.0 { 1.0 } else { 0.0 },
        if is_ep { 1.0 } else { 0.0 },
        if from_enemy > 0 { 1.0 } else { 0.0 },
        if from_own > 0 { 1.0 } else { 0.0 },
        if to_enemy > 0 { 1.0 } else { 0.0 },
        if to_own > 0 { 1.0 } else { 0.0 },
        to_enemy.min(8) as f32,
        to_own.min(8) as f32,
        moving_role.map(piece_value).unwrap_or(0.0),
        captured_value,
        captured_value + promo_gain,
        if is_pinned_to_king(board, moved_color, mv.from) {
            1.0
        } else {
            0.0
        },
        chebyshev(mv.to, enemy_king),
        chebyshev(mv.to, own_king),
    ]
}

fn first_piece_square(board: &Board, color: Color, role: Role) -> Option<u8> {
    board
        .squares
        .iter()
        .position(|&p| p == Some(Piece { color, role }))
        .map(|i| i as u8)
}

fn slider_attackers_to(board: &Board, color: Color, target: Option<u8>) -> usize {
    let Some(target) = target else {
        return 0;
    };
    board
        .squares
        .iter()
        .enumerate()
        .filter(|(from, piece)| {
            let Some(piece) = **piece else {
                return false;
            };
            piece.color == color
                && matches!(piece.role, Role::Bishop | Role::Rook | Role::Queen)
                && piece_attacks_square(board, *from as u8, piece, target)
        })
        .count()
}

fn king_zone(center: Option<u8>) -> Vec<u8> {
    let Some(center) = center else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(9);
    for df in -1..=1 {
        for dr in -1..=1 {
            let f = file(center) + df;
            let r = rank(center) + dr;
            if on(f, r) {
                out.push(idx(f, r));
            }
        }
    }
    out
}

fn zone_attack_count(board: &Board, color: Color, zone: &[u8]) -> usize {
    zone.iter()
        .map(|&sq| attacker_count(board, color, sq))
        .sum()
}

fn escape_squares(board: &Board, color: Color) -> usize {
    let Some(king) = king_square(board, color) else {
        return 0;
    };
    let enemy = color.opposite();
    king_zone(Some(king))
        .into_iter()
        .filter(|&sq| sq != king)
        .filter(|&sq| board.squares[sq as usize].map(|p| p.color) != Some(color))
        .filter(|&sq| !crate::is_square_attacked(board, sq, enemy))
        .count()
}

fn moved_slider_pressure(board: &Board, mv: Move, color: Color, target: Option<u8>) -> bool {
    let Some(target) = target else {
        return false;
    };
    let Some(piece) = board.squares[mv.to as usize] else {
        return false;
    };
    piece.color == color
        && matches!(piece.role, Role::Bishop | Role::Rook | Role::Queen)
        && piece_attacks_square(board, mv.to, piece, target)
}

fn append_attack_features(
    board: &Board,
    after: &Board,
    mv: Move,
    color: Color,
    out: &mut Vec<f32>,
) {
    let enemy = color.opposite();
    let moved_value = board.squares[mv.from as usize]
        .map(|p| piece_value(p.role))
        .unwrap_or(0.0);
    let enemy_counts = attacker_counts_by_role(after, enemy, mv.to);
    let own_counts = attacker_counts_by_role(after, color, mv.to);
    let lva_enemy = least_attacker_value(after, enemy, mv.to);
    let lvd_own = least_attacker_value(after, color, mv.to);
    out.push(attacker_count(board, enemy, mv.from).min(8) as f32);
    out.push(attacker_count(board, color, mv.from).min(8) as f32);
    out.extend(enemy_counts.into_iter().map(|x| x.min(8) as f32));
    out.extend(own_counts.into_iter().map(|x| x.min(8) as f32));
    out.push(lva_enemy);
    out.push(lvd_own);
    out.push(if lva_enemy > 0.0 {
        moved_value - lva_enemy
    } else {
        moved_value
    });
    out.push(
        if lva_enemy > 0.0 && (lvd_own == 0.0 || lva_enemy < moved_value) {
            1.0
        } else {
            0.0
        },
    );
    out.push(if lvd_own > 0.0 { 1.0 } else { 0.0 });
}

fn append_delta_features(board: &Board, after: &Board, mv: Move, color: Color, out: &mut Vec<f32>) {
    let enemy = color.opposite();
    let moved_value = board.squares[mv.from as usize]
        .map(|p| piece_value(p.role))
        .unwrap_or(0.0);
    let cap_value = captured_piece_for_move(board, mv)
        .map(|p| piece_value(p.role))
        .unwrap_or(0.0);
    let promo_gain = mv.promotion.map(|r| piece_value(r) - 1.0).unwrap_or(0.0);
    let own_queens_before = board
        .squares
        .iter()
        .filter(|&&p| {
            p == Some(Piece {
                color,
                role: Role::Queen,
            })
        })
        .count();
    let own_queen_after = first_piece_square(after, color, Role::Queen);
    let own_queens_after = after
        .squares
        .iter()
        .filter(|&&p| {
            p == Some(Piece {
                color,
                role: Role::Queen,
            })
        })
        .count();
    let queen_lost = own_queens_after < own_queens_before;
    let queen_en_prise = own_queen_after
        .map(|q| crate::is_square_attacked(after, q, enemy))
        .unwrap_or(false);
    let replies = legal_moves(after);
    let mut queen_can_be_captured = false;
    let mut moved_piece_can_be_captured = false;
    for reply in &replies {
        let target = after.squares[reply.to as usize];
        if target
            == Some(Piece {
                color,
                role: Role::Queen,
            })
        {
            queen_can_be_captured = true;
        }
        if reply.to == mv.to && target.map(|p| p.color) == Some(color) {
            moved_piece_can_be_captured = true;
        }
    }
    let own_legal = legal_moves(board).len();
    let hanging = moved_piece_can_be_captured && !crate::is_square_attacked(after, mv.to, color);
    out.extend([
        if queen_lost { 1.0 } else { 0.0 },
        if queen_en_prise { 1.0 } else { 0.0 },
        if queen_can_be_captured { 1.0 } else { 0.0 },
        if moved_piece_can_be_captured {
            1.0
        } else {
            0.0
        },
        cap_value - moved_value,
        promo_gain,
        cap_value + promo_gain
            - if moved_piece_can_be_captured && cap_value < moved_value {
                moved_value
            } else {
                0.0
            },
        own_legal.min(64) as f32,
        replies.len().min(64) as f32,
        if hanging && cap_value == 0.0 {
            1.0
        } else {
            0.0
        },
        if cap_value > 0.0 && !crate::is_square_attacked(after, mv.to, color) {
            1.0
        } else {
            0.0
        },
        if moved_piece_can_be_captured && cap_value + promo_gain < moved_value {
            1.0
        } else {
            0.0
        },
    ]);
}

fn append_ray_features(board: &Board, after: &Board, mv: Move, color: Color, out: &mut Vec<f32>) {
    let enemy = color.opposite();
    let enemy_king_pre = king_square(board, enemy);
    let own_king_after = king_square(after, color);
    let enemy_king_after = king_square(after, enemy);
    let own_queen_after = first_piece_square(after, color, Role::Queen);
    let enemy_queen_pre = first_piece_square(board, enemy, Role::Queen);
    let enemy_queen_after = first_piece_square(after, enemy, Role::Queen);
    let kpre = slider_attackers_to(board, color, enemy_king_pre);
    let kafter = slider_attackers_to(after, color, enemy_king_after);
    let qpre = slider_attackers_to(board, color, enemy_queen_pre);
    let qafter = slider_attackers_to(after, color, enemy_queen_after);
    let enemy_ok = slider_attackers_to(after, enemy, own_king_after);
    let enemy_oq = slider_attackers_to(after, enemy, own_queen_after);
    let moved_k = moved_slider_pressure(after, mv, color, enemy_king_after);
    let moved_q = moved_slider_pressure(after, mv, color, enemy_queen_after);
    out.extend([
        kpre as f32,
        kafter as f32,
        kafter as f32 - kpre as f32,
        qpre as f32,
        qafter as f32,
        qafter as f32 - qpre as f32,
        enemy_ok as f32,
        enemy_oq as f32,
        if moved_k { 1.0 } else { 0.0 },
        if moved_q { 1.0 } else { 0.0 },
        if kafter > kpre || qafter > qpre {
            1.0
        } else {
            0.0
        },
        if enemy_oq > 0 { 1.0 } else { 0.0 },
    ]);
}

fn append_kingzone_features(
    board: &Board,
    after: &Board,
    mv: Move,
    color: Color,
    out: &mut Vec<f32>,
) {
    let enemy = color.opposite();
    let own_king_pre = king_square(board, color);
    let enemy_king_pre = king_square(board, enemy);
    let own_king_after = king_square(after, color);
    let enemy_king_after = king_square(after, enemy);
    let enemy_zone_pre = king_zone(enemy_king_pre);
    let own_zone_pre = king_zone(own_king_pre);
    let enemy_zone_after = king_zone(enemy_king_after);
    let own_zone_after = king_zone(own_king_after);
    let own_pre = zone_attack_count(board, color, &enemy_zone_pre);
    let own_after = zone_attack_count(after, color, &enemy_zone_after);
    let enemy_pre = zone_attack_count(board, enemy, &own_zone_pre);
    let enemy_after = zone_attack_count(after, enemy, &own_zone_after);
    let near_enemy = enemy_king_after
        .map(|k| chebyshev(mv.to, k) <= 1.0)
        .unwrap_or(false);
    let from_own = own_king_pre
        .map(|k| chebyshev(mv.from, k) <= 1.0)
        .unwrap_or(false);
    let capture_enemy_zone = captured_piece_for_move(board, mv).is_some() && near_enemy;
    out.extend([
        if near_enemy { 1.0 } else { 0.0 },
        if from_own { 1.0 } else { 0.0 },
        if capture_enemy_zone { 1.0 } else { 0.0 },
        if in_check(after, enemy) || near_enemy {
            1.0
        } else {
            0.0
        },
        own_pre.min(16) as f32,
        own_after.min(16) as f32,
        (own_after as i32 - own_pre as i32).clamp(-16, 16) as f32,
        enemy_pre.min(16) as f32,
        enemy_after.min(16) as f32,
        (enemy_after as i32 - enemy_pre as i32).clamp(-16, 16) as f32,
        escape_squares(after, enemy).min(8) as f32,
        escape_squares(after, color).min(8) as f32,
    ]);
}

pub fn moveformer_feature_groups_from_spec(
    spec: &str,
) -> Result<Vec<MoveFormerFeatureGroup>, String> {
    let raw: Vec<String> = spec
        .replace('+', ",")
        .split(',')
        .filter_map(|s| {
            let s = s.trim().to_ascii_lowercase();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        })
        .collect();
    let raw = if raw.is_empty() {
        vec!["base".to_string()]
    } else {
        raw
    };
    let expanded = if raw.iter().any(|s| s == "all" || s == "all_tactical") {
        vec!["base", "attack", "delta", "ray", "kingzone"]
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        raw
    };
    let mut groups = Vec::new();
    if !expanded.iter().any(|s| s == "base") {
        groups.push(MoveFormerFeatureGroup::Base);
    }
    for item in expanded {
        let group = match item.as_str() {
            "base" => MoveFormerFeatureGroup::Base,
            "attack" | "attackmap" => MoveFormerFeatureGroup::Attack,
            "delta" | "afterstate" => MoveFormerFeatureGroup::Delta,
            "ray" | "ray_summary" => MoveFormerFeatureGroup::Ray,
            "king" | "kingzone" => MoveFormerFeatureGroup::KingZone,
            _ => return Err(format!("unknown MoveFormer feature group: {item}")),
        };
        if !groups.contains(&group) {
            groups.push(group);
        }
    }
    Ok(groups)
}

pub fn moveformer_feature_names_for_groups(groups: &[MoveFormerFeatureGroup]) -> Vec<&'static str> {
    let mut names = Vec::new();
    for group in groups {
        match group {
            MoveFormerFeatureGroup::Base => names.extend(MOVEFORMER_BASE_FEATURE_NAMES),
            MoveFormerFeatureGroup::Attack => names.extend(MOVEFORMER_ATTACK_FEATURE_NAMES),
            MoveFormerFeatureGroup::Delta => names.extend(MOVEFORMER_DELTA_FEATURE_NAMES),
            MoveFormerFeatureGroup::Ray => names.extend(MOVEFORMER_RAY_FEATURE_NAMES),
            MoveFormerFeatureGroup::KingZone => names.extend(MOVEFORMER_KINGZONE_FEATURE_NAMES),
        }
    }
    names
}

pub(crate) fn tactical_moveformer_legal_inputs(
    board: &Board,
    width: usize,
    groups: &[MoveFormerFeatureGroup],
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    let moves = legal_moves(board);
    let feature_count = moveformer_feature_names_for_groups(groups).len();
    let mut action_ids = vec![20480i64; width];
    let mut features = vec![0.0f32; width * feature_count];
    let mut mask = vec![0.0f32; width];
    for (j, mv) in moves.iter().take(width).enumerate() {
        action_ids[j] = move_to_action_id(*mv) as i64;
        mask[j] = 1.0;
        let color = board.squares[mv.from as usize]
            .map(|p| p.color)
            .unwrap_or(board.turn);
        let after = make_move(board, *mv);
        let mut row = Vec::with_capacity(feature_count);
        for group in groups {
            match group {
                MoveFormerFeatureGroup::Base => row.extend(base_moveformer_feature(board, *mv)),
                MoveFormerFeatureGroup::Attack => {
                    append_attack_features(board, &after, *mv, color, &mut row)
                }
                MoveFormerFeatureGroup::Delta => {
                    append_delta_features(board, &after, *mv, color, &mut row)
                }
                MoveFormerFeatureGroup::Ray => {
                    append_ray_features(board, &after, *mv, color, &mut row)
                }
                MoveFormerFeatureGroup::KingZone => {
                    append_kingzone_features(board, &after, *mv, color, &mut row)
                }
            }
        }
        let base = j * feature_count;
        features[base..base + feature_count].copy_from_slice(&row[..feature_count]);
    }
    (moves, action_ids, features, mask)
}

pub fn encode_tactical_moveformer_legal_inputs(
    board: &Board,
    width: usize,
    groups: &[MoveFormerFeatureGroup],
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    tactical_moveformer_legal_inputs(board, width, groups)
}

pub(crate) fn moveformer_legal_inputs(
    board: &Board,
    width: usize,
    feature_count: usize,
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    let moves = legal_moves(board);
    let mut action_ids = vec![20480i64; width];
    let mut features = vec![0.0f32; width * feature_count];
    let mut mask = vec![0.0f32; width];
    for (j, mv) in moves.iter().take(width).enumerate() {
        action_ids[j] = move_to_action_id(*mv) as i64;
        mask[j] = 1.0;
        let base = j * feature_count;
        let base_features = base_moveformer_feature(board, *mv);
        for (k, value) in base_features.into_iter().take(feature_count).enumerate() {
            features[base + k] = value;
        }
    }
    (moves, action_ids, features, mask)
}

pub fn encode_moveformer_legal_inputs(
    board: &Board,
    width: usize,
    feature_count: usize,
) -> (Vec<Move>, Vec<i64>, Vec<f32>, Vec<f32>) {
    moveformer_legal_inputs(board, width, feature_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::Path;

    fn fixture_positions() -> Vec<Value> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("tests/fixtures/contracts/positions.edge_cases.jsonl");
        std::fs::read_to_string(path)
            .expect("read position fixtures")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("parse position fixture"))
            .collect()
    }

    #[test]
    fn onnx_planes_cover_contract_position_fixtures() {
        let meta = OnnxEvaluatorMeta {
            kind: "onnx".to_string(),
            architecture: "residual".to_string(),
            policy_map: "move_action_id_v1".to_string(),
            moves: Vec::new(),
            input_planes: 22,
            history_plies: 0,
            av_head_exported: false,
            max_legal_moves: None,
            onnx_fixed_legal_moves: None,
            num_move_features: None,
            allow_legal_overflow_zero_prior: false,
        };
        for row in fixture_positions() {
            let fen = row["fen"].as_str().unwrap();
            let expected_side = row["expected"]["side_to_move"].as_str().unwrap();
            let board = parse_fen(fen).unwrap();
            let planes = encode_onnx_input_planes(&board, &meta, &[]);
            assert_eq!(planes.len(), meta.input_planes * 64, "{}", row["id"]);
            let piece_count = board.squares.iter().filter(|p| p.is_some()).count() as f32;
            let piece_plane_sum: f32 = planes[..12 * 64].iter().sum();
            assert_eq!(piece_plane_sum, piece_count, "{}", row["id"]);
            let side_plane = &planes[12 * 64..13 * 64];
            let expected = if expected_side == "w" { 1.0 } else { -1.0 };
            assert!(side_plane.iter().all(|&v| v == expected), "{}", row["id"]);
        }
    }

    #[test]
    fn moveformer_legal_inputs_are_aligned() {
        let board = parse_fen(crate::START_FEN).unwrap();
        let (moves, ids, features, mask) = encode_moveformer_legal_inputs(&board, 32, 20);
        assert_eq!(moves.len(), 20);
        assert_eq!(ids.len(), 32);
        assert_eq!(features.len(), 32 * 20);
        assert_eq!(mask.iter().filter(|&&v| v == 1.0).count(), 20);
        for (mv, id) in moves.iter().zip(ids.iter()) {
            assert_eq!(*id, move_to_action_id(*mv) as i64);
        }
        let e2e4 = moves
            .iter()
            .position(|m| crate::move_to_uci(*m) == "e2e4")
            .unwrap();
        let row = &features[e2e4 * 20..e2e4 * 20 + 20];
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 0.0);
        assert_eq!(row[3], 0.0);
        assert_eq!(row[14], 1.0);
    }

    #[test]
    fn tactical_moveformer_feature_groups_shape_outputs() {
        let board = parse_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1").unwrap();
        let groups = moveformer_feature_groups_from_spec("base,attack,delta,ray,kingzone").unwrap();
        let names = moveformer_feature_names_for_groups(&groups);
        assert_eq!(names.len(), 75);
        let (moves, ids, features, mask) =
            encode_tactical_moveformer_legal_inputs(&board, 96, &groups);
        assert_eq!(ids.len(), 96);
        assert_eq!(features.len(), 96 * names.len());
        assert_eq!(mask.iter().filter(|&&v| v == 1.0).count(), moves.len());
        assert!(moves.iter().any(|m| crate::move_to_uci(*m) == "e5d6"));
    }
}
