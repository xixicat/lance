// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::process::ExitCode;

use lance_index::vector::bq::residual_levels::{
    ResidualEncoder, encode_rotated, lower_bound_sq, search_partition,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

struct Dataset {
    rows: Vec<Vec<f32>>,
}

fn gaussian(dim: usize, rows: usize, rng: &mut StdRng) -> Dataset {
    let normal = StandardNormal;
    Dataset {
        rows: (0..rows)
            .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
            .collect(),
    }
}

fn clustered(dim: usize, rows: usize, rng: &mut StdRng) -> Dataset {
    let normal = StandardNormal;
    let centers = 20usize;
    let centers: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
        .collect();
    Dataset {
        rows: (0..rows)
            .map(|idx| {
                let center = &centers[idx % centers.len()];
                center
                    .iter()
                    .map(|value| {
                        let noise: f32 = normal.sample(rng);
                        value + 0.15f32 * noise
                    })
                    .collect()
            })
            .collect(),
    }
}

fn recall_for(
    encoder: &ResidualEncoder,
    rows: &[Vec<f32>],
    queries: &[Vec<f32>],
    levels: usize,
    k: usize,
    prune: bool,
) -> (f32, f32) {
    let encoded: Vec<_> = rows
        .iter()
        .map(|row| encoder.encode(row, levels, true).unwrap())
        .collect();
    let mut recall = 0.0f32;
    let mut pruned = 0.0f32;
    for query in queries {
        let rotated_query = encoder.rotate(query).unwrap();
        let mut exact: Vec<(f32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(id, row)| {
                let rotated = encoder.rotate(row).unwrap();
                let dist = rotated
                    .iter()
                    .zip(rotated_query.iter())
                    .map(|(left, right)| {
                        let diff = left - right;
                        diff * diff
                    })
                    .sum();
                (dist, id)
            })
            .collect();
        exact.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let truth: HashSet<usize> = exact.into_iter().take(k).map(|(_, id)| id).collect();
        let stats = search_partition(&rotated_query, &encoded, k, prune);
        let hits = stats
            .hits
            .iter()
            .filter(|hit| truth.contains(&hit.id))
            .count();
        recall += hits as f32 / k as f32;
        pruned += stats.pruned as f32 / rows.len() as f32;
    }
    (recall / queries.len() as f32, pruned / queries.len() as f32)
}

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() -> ExitCode {
    let mut rng = StdRng::seed_from_u64(42);
    let mut failed = false;
    println!("dataset\tdim\trows\tqueries\tm\tprune\trecall_at_10\tprune_ratio");

    let rotated = vec![0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 0.3, -0.7];
    let probe = encode_rotated(&rotated, 4, true).unwrap();
    let self_distance = lance_index::vector::bq::residual_levels::estimated_l2_sq(&rotated, &probe);
    if self_distance.abs() > 1.0e-2 {
        eprintln!("self-query distance failed distance={self_distance}");
        failed = true;
    }
    for done in 1..=probe.levels.len() {
        let bound = lower_bound_sq(&rotated, &probe, done);
        if bound > 1.0e-2 {
            eprintln!("self-query bound failed done={done} bound={bound}");
            failed = true;
        }
    }

    for (dim, rows, queries_n) in [(128usize, 400usize, 40usize), (768, 200, 20)] {
        let data = gaussian(dim, rows, &mut rng);
        let queries: Vec<Vec<f32>> = (0..queries_n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand_distr::StandardNormal.sample(&mut rng))
                    .collect()
            })
            .collect();
        let encoder = ResidualEncoder::new(dim);
        let mut recall_m1 = 0.0f32;
        let mut recall_m8 = 0.0f32;
        for levels in [1usize, 2, 4, 8] {
            let (recall, _) = recall_for(&encoder, &data.rows, &queries, levels, 10, false);
            println!("gaussian\t{dim}\t{rows}\t{queries_n}\t{levels}\toff\t{recall:.4}\t0");
            if levels == 1 {
                recall_m1 = recall;
            }
            if levels == 8 {
                recall_m8 = recall;
            }
        }
        let (recall_pruned, prune_ratio) = recall_for(&encoder, &data.rows, &queries, 8, 10, true);
        println!(
            "gaussian\t{dim}\t{rows}\t{queries_n}\t8\ton\t{recall_pruned:.4}\t{prune_ratio:.4}"
        );
        if dim == 128 && recall_m8 < 0.5 {
            eprintln!("gate failed: dim 128 m=8 recall {recall_m8} < 0.5");
            failed = true;
        }
        if dim == 128 && recall_m8 + 0.02 < recall_m1 {
            eprintln!("gate failed: m=8 recall {recall_m8} regressed past m=1 recall {recall_m1}");
            failed = true;
        }
        if dim == 128 && recall_pruned + 1.0e-6 < recall_m8 {
            eprintln!(
                "gate failed: pruned recall {recall_pruned} dropped below full recall {recall_m8}"
            );
            failed = true;
        }

        let clustered = clustered(dim, rows, &mut rng);
        let (recall, _) = recall_for(&encoder, &clustered.rows, &queries, 8, 10, false);
        println!("clustered\t{dim}\t{rows}\t{queries_n}\t8\toff\t{recall:.4}\t0");
    }

    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
