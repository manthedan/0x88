#[cfg(feature = "native-ort")]
use serde::Deserialize;

mod board;
mod eval;
mod fen;
mod move_codec;
mod movegen;
mod search;
#[cfg(feature = "native-ort")]
use board::{file, rank};
pub use board::{square_index, square_name, Board, Color, Move, Piece, Role, START_FEN};
pub use eval::{
    fen_features, frozen_conv_student_features, Evaluation, PositionEvaluator, StudentArtifact,
    StudentEvaluator, UniformEvaluator,
};
#[cfg(feature = "native-ort")]
use eval::{move_to_fixed_policy_index, piece_index, softmax};
pub use fen::{board_to_fen, parse_fen};
pub use move_codec::{move_to_action_id, move_to_uci};
pub use movegen::{
    in_check, is_square_attacked, king_square, legal_moves, make_move, pseudo_legal_moves,
};
pub use search::{
    search_root, search_root_with_history, SearchOptions, SearchPolicyEntry, SearchPolicyMode,
    SearchResult,
};

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
