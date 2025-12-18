//! MCTS benchmarks for performance profiling.
//!
//! Run with: `cargo bench -p mcts`
//!
//! These benchmarks measure:
//! - Full MCTS search with varying simulation counts
//! - Tree operations (selection, backpropagation, policy extraction)
//! - Search from different game states (opening, midgame, near-terminal)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use engine_core::EngineContext;
use mcts::{MctsConfig, MctsSearch, MctsTree, UniformEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Register TicTacToe game once for all benchmarks.
fn setup_tictactoe() -> EngineContext {
    games_tictactoe::register_tictactoe();
    EngineContext::new("tictactoe").unwrap()
}

/// Helper to create a game state after playing a sequence of moves.
fn play_moves(ctx: &mut EngineContext, seed: u64, moves: &[u32]) -> (Vec<u8>, Vec<u8>, u64) {
    let reset = ctx.reset(seed, &[]).unwrap();
    let mut state = reset.state;
    let mut obs = reset.obs;
    let mut info = 0b111111111u64; // All positions legal initially

    for &m in moves {
        let action = m.to_le_bytes().to_vec();
        let step = ctx.step(&state, &action).unwrap();
        state = step.state;
        obs = step.obs;
        info = step.info;
    }

    let legal_mask = info & 0x1FF;
    (state, obs, legal_mask)
}

// =============================================================================
// Full MCTS Search Benchmarks
// =============================================================================

fn bench_mcts_search_simulations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_search_simulations");

    // Test different simulation counts
    for sims in [50, 100, 200, 400, 800] {
        group.throughput(Throughput::Elements(sims as u64));
        group.bench_with_input(BenchmarkId::new("uniform", sims), &sims, |b, &sims| {
            let mut ctx = setup_tictactoe();
            let evaluator = UniformEvaluator::new();
            let config = MctsConfig::for_testing().with_simulations(sims);

            b.iter(|| {
                let reset = ctx.reset(42, &[]).unwrap();
                let legal_mask = 0b111111111u64;
                let mut rng = ChaCha20Rng::seed_from_u64(42);

                let mut search = MctsSearch::new(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    reset.state,
                    reset.obs,
                    legal_mask,
                )
                .unwrap();

                black_box(search.run(&mut rng).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_mcts_game_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_game_phases");
    let sims = 200u32;

    // Opening position (all 9 moves available)
    group.bench_function("opening", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Midgame position (5 moves available)
    // Board: X at 4, O at 0, X at 2, O at 6
    group.bench_function("midgame", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let (state, obs, legal_mask) = play_moves(&mut ctx, 42, &[4, 0, 2, 6]);
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search =
                MctsSearch::new(&mut ctx, &evaluator, config.clone(), state, obs, legal_mask)
                    .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Near-terminal position (winning move available)
    // Board: X at 0, O at 3, X at 1, O at 4 -> X can win at 2
    group.bench_function("near_terminal", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let (state, obs, legal_mask) = play_moves(&mut ctx, 42, &[0, 3, 1, 4]);
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search =
                MctsSearch::new(&mut ctx, &evaluator, config.clone(), state, obs, legal_mask)
                    .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    group.finish();
}

// =============================================================================
// Tree Operation Benchmarks
// =============================================================================

fn bench_tree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_tree_ops");

    // Benchmark node allocation
    group.bench_function("allocate_node", |b| {
        b.iter(|| {
            let mut tree = MctsTree::new(vec![0u8; 16], vec![0u8; 32], 0b111111111);

            // Allocate 100 child nodes
            for i in 0..100u8 {
                tree.add_child(
                    tree.root(),
                    i % 9,
                    0.11,
                    vec![i; 16],
                    vec![i; 32],
                    0b111111111,
                    false,
                    0.0,
                );
            }

            black_box(tree.len())
        });
    });

    // Benchmark child selection (UCB calculation)
    group.bench_function("select_child", |b| {
        // Pre-build a tree with children
        let mut tree = MctsTree::new(vec![0u8; 16], vec![0u8; 32], 0b111111111);

        // Add 9 children with varying priors and visit counts
        for i in 0..9u8 {
            let child_id = tree.add_child(
                tree.root(),
                i,
                (i as f32 + 1.0) / 45.0, // Varying priors
                vec![i; 16],
                vec![i; 32],
                0b111111111,
                false,
                0.0,
            );
            // Simulate some visits
            let child = tree.get_mut(child_id);
            child.visit_count = (i as u32 + 1) * 10;
            child.value_sum = (i as f32 - 4.0) * 0.1 * child.visit_count as f32;
        }

        // Update root visit count
        tree.get_mut(tree.root()).visit_count = 450;

        b.iter(|| black_box(tree.select_child(tree.root(), 1.25)));
    });

    // Benchmark backpropagation
    group.bench_function("backpropagate_depth_5", |b| {
        b.iter_batched(
            || {
                // Setup: create a tree with depth 5
                let mut tree = MctsTree::new(vec![0u8; 16], vec![0u8; 32], 0b111111111);
                let mut parent = tree.root();

                for i in 0..5 {
                    let child = tree.add_child(
                        parent,
                        i,
                        0.5,
                        vec![i; 16],
                        vec![i; 32],
                        0b111111111,
                        i == 4, // Last one is terminal
                        if i == 4 { 1.0 } else { 0.0 },
                    );
                    parent = child;
                }

                (tree, parent)
            },
            |(mut tree, leaf)| {
                tree.backpropagate(leaf, 1.0);
                black_box(tree)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark policy extraction
    group.bench_function("root_policy", |b| {
        // Pre-build a tree with children
        let mut tree = MctsTree::new(vec![0u8; 16], vec![0u8; 32], 0b111111111);

        for i in 0..9u8 {
            let child_id = tree.add_child(
                tree.root(),
                i,
                1.0 / 9.0,
                vec![i; 16],
                vec![i; 32],
                0b111111111,
                false,
                0.0,
            );
            tree.get_mut(child_id).visit_count = (i as u32 + 1) * 50;
        }

        b.iter(|| black_box(tree.root_policy(9, 1.0)));
    });

    // Benchmark policy extraction with temperature scaling
    group.bench_function("root_policy_temperature", |b| {
        let mut tree = MctsTree::new(vec![0u8; 16], vec![0u8; 32], 0b111111111);

        for i in 0..9u8 {
            let child_id = tree.add_child(
                tree.root(),
                i,
                1.0 / 9.0,
                vec![i; 16],
                vec![i; 32],
                0b111111111,
                false,
                0.0,
            );
            tree.get_mut(child_id).visit_count = (i as u32 + 1) * 50;
        }

        b.iter(|| black_box(tree.root_policy(9, 0.5)));
    });

    group.finish();
}

// =============================================================================
// Configuration Comparison Benchmarks
// =============================================================================

fn bench_mcts_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_configs");
    let sims = 200u32;

    // Training config (with Dirichlet noise)
    group.bench_function("training_config", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_training().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Evaluation config (no noise, greedy)
    group.bench_function("evaluation_config", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_evaluation().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Different c_puct values
    for c_puct in [0.5, 1.25, 2.5, 4.0] {
        group.bench_with_input(BenchmarkId::new("c_puct", c_puct), &c_puct, |b, &c_puct| {
            let mut ctx = setup_tictactoe();
            let evaluator = UniformEvaluator::new();
            let config = MctsConfig::for_testing()
                .with_simulations(sims)
                .with_c_puct(c_puct);

            b.iter(|| {
                let reset = ctx.reset(42, &[]).unwrap();
                let legal_mask = 0b111111111u64;
                let mut rng = ChaCha20Rng::seed_from_u64(42);

                let mut search = MctsSearch::new(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    reset.state,
                    reset.obs,
                    legal_mask,
                )
                .unwrap();

                black_box(search.run(&mut rng).unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mcts_search_simulations,
    bench_mcts_game_phases,
    bench_tree_operations,
    bench_mcts_configs,
);

criterion_main!(benches);
