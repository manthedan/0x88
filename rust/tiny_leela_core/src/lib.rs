mod board;
mod eval;
mod fen;
mod move_codec;
mod movegen;
#[cfg(feature = "native-ort")]
mod onnx;
mod search;
pub use board::{square_index, square_name, Board, Color, Move, Piece, Role, START_FEN};
pub use eval::{
    fen_features, frozen_conv_student_features, Evaluation, PositionEvaluator, StudentArtifact,
    StudentEvaluator, UniformEvaluator,
};
pub use fen::{board_to_fen, parse_fen};
pub use move_codec::{move_to_action_id, move_to_uci};
pub use movegen::{
    in_check, is_square_attacked, king_square, legal_moves, make_move, pseudo_legal_moves,
};
#[cfg(feature = "native-ort")]
pub use onnx::{
    encode_moveformer_legal_inputs, encode_onnx_input_planes, encode_squareformer_compact_input,
    encode_squareformer_float_input, encode_squareformer_legal_ids, is_squareformer_compact_meta,
    OnnxEvaluator, OnnxEvaluatorMeta, SquareFormerEvaluatorMeta,
};
pub use search::{
    search_root, search_root_with_history, SearchOptions, SearchPolicyEntry, SearchPolicyMode,
    SearchResult,
};

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
