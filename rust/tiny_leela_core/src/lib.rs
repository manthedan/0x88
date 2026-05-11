mod board;
mod encoding;
mod eval;
mod fen;
mod matchplay;
mod move_codec;
mod movegen;
#[cfg(feature = "native-ort")]
mod onnx;
mod search;
mod squareformer;
pub use board::{square_index, square_name, Board, Color, Move, Piece, Role, START_FEN};
pub use encoding::{encode_moveformer_legal_inputs, encode_onnx_input_planes, OnnxEvaluatorMeta};
pub use eval::{
    fen_features, frozen_conv_student_features, Evaluation, PositionEvaluator, StudentArtifact,
    StudentEvaluator, UniformEvaluator,
};
pub use fen::{board_to_fen, parse_fen};
pub use matchplay::{
    plan_round_robin_jobs, round_robin_total_games, score_for_color, shard_jobs, RoundRobinJob,
};
pub use move_codec::{move_to_action_id, move_to_uci};
pub use movegen::{
    in_check, is_square_attacked, king_square, legal_moves, make_move, pseudo_legal_moves,
};
#[cfg(feature = "native-ort")]
pub use onnx::OnnxEvaluator;
pub use search::{
    search_root, search_root_with_history, SearchOptions, SearchPolicyEntry, SearchPolicyMode,
    SearchResult,
};
pub use squareformer::{
    encode_squareformer_compact_input, encode_squareformer_float_input,
    encode_squareformer_legal_ids, is_squareformer_compact_meta, SquareFormerEvaluatorMeta,
};
