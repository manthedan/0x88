use std::time::Instant;

use crate::{Board, Evaluation, PositionEvaluator};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationTicket(pub u64);

#[derive(Clone, Debug)]
pub struct BatchedEvaluatorOptions {
    pub max_batch_size: usize,
    pub max_wait_micros: u64,
}

impl Default for BatchedEvaluatorOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_micros: 1_000,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BatchedEvaluatorMetrics {
    pub enqueued_positions: u64,
    pub evaluated_positions: u64,
    pub batches: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_eval_micros: u128,
    pub last_batch_size: usize,
}

impl BatchedEvaluatorMetrics {
    pub fn average_batch_size(&self) -> f64 {
        if self.batches == 0 {
            0.0
        } else {
            self.evaluated_positions as f64 / self.batches as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    pub fn throughput_positions_per_second(&self) -> f64 {
        if self.total_eval_micros == 0 {
            0.0
        } else {
            self.evaluated_positions as f64 / (self.total_eval_micros as f64 / 1_000_000.0)
        }
    }
}

#[derive(Clone, Debug)]
struct QueuedEvaluation {
    ticket: EvaluationTicket,
    board: Board,
    history_fens: Vec<String>,
}

/// Synchronous batching adapter for deterministic search/eval orchestration.
///
/// The adapter preserves ticket/result ordering across every flush.  It is intentionally
/// executor-free: callers can queue leaves from one or more games/searches and decide when to
/// flush based on `should_flush`, `max_batch_size`, or their own wall-clock policy.
pub struct BatchedEvaluator<E> {
    evaluator: E,
    options: BatchedEvaluatorOptions,
    queue: Vec<QueuedEvaluation>,
    next_ticket: u64,
    metrics: BatchedEvaluatorMetrics,
}

impl<E: PositionEvaluator> BatchedEvaluator<E> {
    pub fn new(evaluator: E, options: BatchedEvaluatorOptions) -> Self {
        let max_batch_size = options.max_batch_size.max(1);
        Self {
            evaluator,
            options: BatchedEvaluatorOptions {
                max_batch_size,
                ..options
            },
            queue: Vec::with_capacity(max_batch_size),
            next_ticket: 0,
            metrics: BatchedEvaluatorMetrics::default(),
        }
    }

    pub fn enqueue(&mut self, board: Board, history_fens: Vec<String>) -> EvaluationTicket {
        let ticket = EvaluationTicket(self.next_ticket);
        self.next_ticket += 1;
        self.queue.push(QueuedEvaluation {
            ticket,
            board,
            history_fens,
        });
        self.metrics.enqueued_positions += 1;
        ticket
    }

    pub fn should_flush(&self) -> bool {
        self.queue.len() >= self.options.max_batch_size
    }

    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    pub fn options(&self) -> &BatchedEvaluatorOptions {
        &self.options
    }

    pub fn metrics(&self) -> &BatchedEvaluatorMetrics {
        &self.metrics
    }

    pub fn flush(&mut self) -> Vec<(EvaluationTicket, Evaluation)> {
        if self.queue.is_empty() {
            return Vec::new();
        }
        let queued = std::mem::take(&mut self.queue);
        let started = Instant::now();
        let mut out = Vec::with_capacity(queued.len());
        for item in queued {
            let eval = self
                .evaluator
                .evaluate_with_history(&item.board, &item.history_fens);
            out.push((item.ticket, eval));
        }
        let elapsed = started.elapsed().as_micros();
        self.metrics.batches += 1;
        self.metrics.evaluated_positions += out.len() as u64;
        self.metrics.cache_misses += out.len() as u64;
        self.metrics.total_eval_micros += elapsed;
        self.metrics.last_batch_size = out.len();
        out
    }

    pub fn into_inner(self) -> E {
        self.evaluator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_fen, UniformEvaluator, START_FEN};

    #[test]
    fn batcher_preserves_ticket_order_and_metrics() {
        let evaluator = UniformEvaluator;
        let mut batcher = BatchedEvaluator::new(
            evaluator,
            BatchedEvaluatorOptions {
                max_batch_size: 2,
                max_wait_micros: 500,
            },
        );
        let board = parse_fen(START_FEN).unwrap();
        let t0 = batcher.enqueue(board.clone(), Vec::new());
        assert!(!batcher.should_flush());
        let t1 = batcher.enqueue(board, Vec::new());
        assert!(batcher.should_flush());
        let out = batcher.flush();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, t0);
        assert_eq!(out[1].0, t1);
        assert_eq!(batcher.metrics().batches, 1);
        assert_eq!(batcher.metrics().evaluated_positions, 2);
        assert_eq!(batcher.metrics().last_batch_size, 2);
        assert_eq!(batcher.metrics().cache_hit_rate(), 0.0);
        assert!(batcher.metrics().throughput_positions_per_second() >= 0.0);
    }
}
