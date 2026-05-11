use serde::Deserialize;

use crate::board::{file, rank};
use crate::eval::piece_index;
use crate::{king_square, legal_moves, move_to_action_id, parse_fen, Board, Color, Move, Role};

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

pub(crate) fn moveformer_legal_inputs(
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
        let (moves, ids, features, mask) = encode_moveformer_legal_inputs(&board, 32, 12);
        assert_eq!(moves.len(), 20);
        assert_eq!(ids.len(), 32);
        assert_eq!(features.len(), 32 * 12);
        assert_eq!(mask.iter().filter(|&&v| v == 1.0).count(), 20);
        for (mv, id) in moves.iter().zip(ids.iter()) {
            assert_eq!(*id, move_to_action_id(*mv) as i64);
        }
    }
}
