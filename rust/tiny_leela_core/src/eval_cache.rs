use std::collections::{HashMap, VecDeque};

use crate::{board_to_fen, Board, Evaluation, PositionEvaluator};

#[derive(Clone, Debug)]
pub struct EvalCacheOptions {
    pub include_history: bool,
    pub max_entries: Option<usize>,
}

impl Default for EvalCacheOptions {
    fn default() -> Self {
        Self {
            include_history: true,
            max_entries: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvalCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
}

impl EvalCacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvalCacheKey {
    fen: String,
    history_fens: Vec<String>,
}

impl EvalCacheKey {
    pub fn new(board: &Board, history_fens: &[String], include_history: bool) -> Self {
        Self {
            fen: board_to_fen(board),
            history_fens: if include_history {
                history_fens.to_vec()
            } else {
                Vec::new()
            },
        }
    }
}

pub struct CachedEvaluator<E> {
    evaluator: E,
    options: EvalCacheOptions,
    cache: HashMap<EvalCacheKey, Evaluation>,
    order: VecDeque<EvalCacheKey>,
    metrics: EvalCacheMetrics,
}

impl<E: PositionEvaluator> CachedEvaluator<E> {
    pub fn new(evaluator: E, options: EvalCacheOptions) -> Self {
        Self {
            evaluator,
            options,
            cache: HashMap::new(),
            order: VecDeque::new(),
            metrics: EvalCacheMetrics::default(),
        }
    }

    pub fn per_search(evaluator: E) -> Self {
        Self::new(evaluator, EvalCacheOptions::default())
    }

    pub fn cross_game(evaluator: E, max_entries: usize) -> Self {
        Self::new(
            evaluator,
            EvalCacheOptions {
                include_history: true,
                max_entries: Some(max_entries.max(1)),
            },
        )
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    pub fn metrics(&self) -> &EvalCacheMetrics {
        &self.metrics
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    fn insert(&mut self, key: EvalCacheKey, eval: Evaluation) {
        if !self.cache.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.cache.insert(key, eval);
        self.metrics.inserts += 1;
        if let Some(max_entries) = self.options.max_entries {
            while self.cache.len() > max_entries {
                if let Some(old) = self.order.pop_front() {
                    if self.cache.remove(&old).is_some() {
                        self.metrics.evictions += 1;
                    }
                } else {
                    break;
                }
            }
        }
    }
}

impl<E: PositionEvaluator> CachedEvaluator<E> {
    pub fn evaluate_cached(&mut self, board: &Board) -> Evaluation {
        self.evaluate_with_history_cached(board, &[])
    }

    pub fn evaluate_with_history_cached(
        &mut self,
        board: &Board,
        history_fens: &[String],
    ) -> Evaluation {
        let key = EvalCacheKey::new(board, history_fens, self.options.include_history);
        if let Some(eval) = self.cache.get(&key) {
            self.metrics.hits += 1;
            return eval.clone();
        }
        self.metrics.misses += 1;
        let eval = self.evaluator.evaluate_with_history(board, history_fens);
        self.insert(key, eval.clone());
        eval
    }

    pub fn into_inner(self) -> E {
        self.evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_move, parse_fen, Move, Role, UniformEvaluator, START_FEN};

    #[test]
    fn cached_evaluator_tracks_hits_and_history_keys() {
        let board = parse_fen(START_FEN).unwrap();
        let mut cached = CachedEvaluator::per_search(UniformEvaluator);
        let _ = cached.evaluate_with_history_cached(&board, &["a".to_string()]);
        let _ = cached.evaluate_with_history_cached(&board, &["a".to_string()]);
        let _ = cached.evaluate_with_history_cached(&board, &["b".to_string()]);
        assert_eq!(cached.metrics().misses, 2);
        assert_eq!(cached.metrics().hits, 1);
        assert_eq!(cached.len(), 2);
    }

    #[test]
    fn cached_evaluator_evicts_old_entries() {
        let board = parse_fen(START_FEN).unwrap();
        let board2 = make_move(
            &board,
            Move {
                from: 12,
                to: 28,
                promotion: None::<Role>,
            },
        );
        let mut cached = CachedEvaluator::cross_game(UniformEvaluator, 1);
        let _ = cached.evaluate_cached(&board);
        let _ = cached.evaluate_cached(&board2);
        assert_eq!(cached.len(), 1);
        assert_eq!(cached.metrics().evictions, 1);
    }
}
