use crate::encoding::{move_to_chessbench_av_class, moveformer_legal_inputs, onnx_input_planes};
use crate::eval::{move_to_fixed_policy_index, softmax};
use crate::{
    encode_squareformer_compact_input, encode_squareformer_float_input,
    encode_squareformer_legal_ids, is_squareformer_compact_meta, legal_moves, move_to_action_id,
    Board, Evaluation, Move, OnnxEvaluatorMeta, PositionEvaluator, SquareFormerEvaluatorMeta,
};

#[cfg(feature = "native-ort")]
pub struct OnnxEvaluator {
    session: std::sync::Mutex<ort::session::Session>,
    board_meta: Option<OnnxEvaluatorMeta>,
    square_meta: Option<SquareFormerEvaluatorMeta>,
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
