use std::collections::HashMap;

use crate::{
    board_to_fen, in_check, legal_moves, make_move, move_to_action_id, parse_fen, Board, Move,
    PositionEvaluator, StudentEvaluator, UniformEvaluator, START_FEN,
};

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
