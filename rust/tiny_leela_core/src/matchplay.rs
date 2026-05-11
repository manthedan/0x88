use crate::Color;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundRobinJob {
    pub game: usize,
    pub a: usize,
    pub b: usize,
    pub local_game: usize,
    pub a_color: Color,
    pub opening_index: usize,
}

pub fn score_for_color(white_score: f32, color: Color) -> f32 {
    if (white_score - 0.5).abs() < 1e-6 {
        0.5
    } else if (white_score == 1.0 && color == Color::White)
        || (white_score == 0.0 && color == Color::Black)
    {
        1.0
    } else {
        0.0
    }
}

pub fn round_robin_total_games(player_count: usize, games_per_pair: usize) -> usize {
    player_count.saturating_mul(player_count.saturating_sub(1)) / 2 * games_per_pair
}

pub fn plan_round_robin_jobs(
    player_count: usize,
    games_per_pair: usize,
    opening_count: usize,
) -> Vec<RoundRobinJob> {
    assert!(player_count >= 2, "round robin needs at least two players");
    assert!(opening_count > 0, "round robin needs at least one opening");
    let mut jobs = Vec::with_capacity(round_robin_total_games(player_count, games_per_pair));
    let mut game = 0usize;
    for a in 0..player_count {
        for b in (a + 1)..player_count {
            for local_game in 0..games_per_pair {
                jobs.push(RoundRobinJob {
                    game,
                    a,
                    b,
                    local_game,
                    a_color: if local_game % 2 == 0 {
                        Color::White
                    } else {
                        Color::Black
                    },
                    opening_index: (local_game / 2) % opening_count,
                });
                game += 1;
            }
        }
    }
    jobs
}

pub fn shard_jobs(
    jobs: &[RoundRobinJob],
    shard_count: usize,
    shard_index: usize,
) -> Vec<RoundRobinJob> {
    assert!(shard_count > 0, "shard_count must be positive");
    assert!(
        shard_index < shard_count,
        "shard_index must be less than shard_count"
    );
    jobs.iter()
        .filter(|job| job.game % shard_count == shard_index)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_game_ids_are_stable() {
        let jobs = plan_round_robin_jobs(3, 2, 4);
        let ids: Vec<usize> = jobs.iter().map(|j| j.game).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            (jobs[0].a, jobs[0].b, jobs[0].a_color),
            (0, 1, Color::White)
        );
        assert_eq!(
            (jobs[1].a, jobs[1].b, jobs[1].a_color),
            (0, 1, Color::Black)
        );
        assert_eq!(
            (jobs[2].a, jobs[2].b, jobs[2].a_color),
            (0, 2, Color::White)
        );
    }

    #[test]
    fn shards_partition_round_robin_games() {
        let jobs = plan_round_robin_jobs(8, 2, 24);
        assert_eq!(jobs.len(), 56);
        let mut ids = Vec::new();
        let sizes: Vec<usize> = (0..6)
            .map(|i| {
                let shard = shard_jobs(&jobs, 6, i);
                ids.extend(shard.iter().map(|j| j.game));
                shard.len()
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..56).collect::<Vec<_>>());
        assert_eq!(sizes, vec![10, 10, 9, 9, 9, 9]);
    }

    #[test]
    fn score_for_color_maps_white_result() {
        assert_eq!(score_for_color(1.0, Color::White), 1.0);
        assert_eq!(score_for_color(0.0, Color::White), 0.0);
        assert_eq!(score_for_color(1.0, Color::Black), 0.0);
        assert_eq!(score_for_color(0.0, Color::Black), 1.0);
        assert_eq!(score_for_color(0.5, Color::Black), 0.5);
    }
}
