use serde::Deserialize;
use std::collections::HashMap;

use crate::{
    board_to_fen, in_check, legal_moves, move_to_action_id, move_to_uci, parse_fen, Board, Color,
    Move, Piece, Role,
};

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

pub(crate) fn piece_index(piece: Piece) -> usize {
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

pub(crate) fn softmax(xs: &[f32]) -> Vec<f32> {
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

pub(crate) fn move_to_fixed_policy_index(m: Move) -> Option<usize> {
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
