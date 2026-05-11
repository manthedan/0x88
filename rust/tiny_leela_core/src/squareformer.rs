use serde::Deserialize;

use crate::encoding::move_to_chessbench_av_class;
use crate::{legal_moves, parse_fen, Board, Color, Move, Piece, Role};

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

fn squareformer_castle_mask(castling: u8) -> i64 {
    castling as i64
}

pub fn is_squareformer_compact_meta(meta: &SquareFormerEvaluatorMeta) -> bool {
    meta.input_mode.as_deref() == Some("embedding")
        || meta.input_format.as_deref() == Some("compact_uint8_embeddings")
        || meta.input_format.as_deref() == Some("compact_uint8_tokens")
}

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

pub fn encode_squareformer_legal_ids(board: &Board, width: usize) -> (Vec<Move>, Vec<i64>) {
    let moves = legal_moves(board);
    let mut ids = vec![0i64; width];
    for (j, mv) in moves.iter().take(width).enumerate() {
        ids[j] = move_to_chessbench_av_class(*mv);
    }
    (moves, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_squareformer_tokens_encode_state() {
        let board = parse_fen("8/8/8/3pP3/8/8/8/4K2k w Kq d6 17 1").unwrap();
        let meta = SquareFormerEvaluatorMeta {
            kind: "squareformer".to_string(),
            input_dim: None,
            token_features: Some(11),
            input_mode: Some("compact".to_string()),
            input_format: None,
            policy_size: 4096 + 4096 * 4,
            history_plies: 2,
            av_head_exported: false,
            max_legal_moves: None,
            onnx_fixed_legal_moves: None,
        };
        let tokens = encode_squareformer_compact_input(&board, &meta, &[]);
        assert_eq!(tokens.len(), 64 * 11);
        assert_eq!(tokens[36 * 11], 1); // white pawn on e5
        assert_eq!(tokens[35 * 11], 7); // black pawn on d5
        assert_eq!(tokens[43 * 11 + 5], 1); // en-passant square d6
        assert_eq!(tokens[11 + 4], 9); // Kq castling mask broadcast
        assert_eq!(tokens[11 + 6], 17); // halfmove clock broadcast
        assert_eq!(tokens[11 + 3], 1); // white to move broadcast
    }

    #[test]
    fn squareformer_legal_ids_match_move_classes() {
        let board = parse_fen(crate::START_FEN).unwrap();
        let (moves, ids) = encode_squareformer_legal_ids(&board, 32);
        assert_eq!(moves.len(), 20);
        for (mv, id) in moves.iter().zip(ids.iter()) {
            assert_eq!(*id, move_to_chessbench_av_class(*mv));
        }
        assert!(ids[20..].iter().all(|&v| v == 0));
    }
}
