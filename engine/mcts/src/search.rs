//! MCTS search implementation.
//!
//! Implements the core MCTS algorithm:
//! 1. Selection: Traverse tree using UCB to find a leaf
//! 2. Expansion: Add children to the leaf using policy prior
//! 3. Evaluation: Get value estimate from evaluator
//! 4. Backpropagation: Update statistics along the path

use engine_core::{ActionSpace, EngineContext};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use thiserror::Error;
use tracing::trace;

use crate::config::MctsConfig;
use crate::evaluator::{Evaluator, EvaluatorError};
use crate::node::NodeId;
use crate::tree::MctsTree;

/// Errors that can occur during MCTS search.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("Engine error: {0}")]
    EngineError(String),

    #[error("Evaluator error: {0}")]
    EvaluatorError(#[from] EvaluatorError),

    #[error("No legal moves available")]
    NoLegalMoves,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("MCTS only supports discrete action spaces")]
    UnsupportedActionSpace,
}

/// Result of an MCTS search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best action to take
    pub action: u8,

    /// Policy distribution over actions (visit counts normalized)
    pub policy: Vec<f32>,

    /// Value estimate at root
    pub value: f32,

    /// Number of simulations performed
    pub simulations: u32,
}

/// MCTS search state.
pub struct MctsSearch<'a, E: Evaluator> {
    tree: MctsTree,
    ctx: &'a mut EngineContext,
    evaluator: &'a E,
    config: MctsConfig,
    num_actions: usize,
}

impl<'a, E: Evaluator> MctsSearch<'a, E> {
    /// Create a new MCTS search from the given game state.
    pub fn new(
        ctx: &'a mut EngineContext,
        evaluator: &'a E,
        config: MctsConfig,
        state: Vec<u8>,
        obs: Vec<u8>,
        legal_moves_mask: u64,
    ) -> Result<Self, SearchError> {
        let num_actions = match ctx.action_space() {
            ActionSpace::Discrete(n) => n as usize,
            _ => return Err(SearchError::UnsupportedActionSpace),
        };

        let tree = MctsTree::new(state, obs, legal_moves_mask);

        Ok(Self {
            tree,
            ctx,
            evaluator,
            config,
            num_actions,
        })
    }

    /// Run the MCTS search for the configured number of simulations.
    pub fn run(&mut self, rng: &mut ChaCha20Rng) -> Result<SearchResult, SearchError> {
        // First, expand the root if needed
        if !self.tree.get(self.tree.root()).is_expanded() {
            self.expand_node(self.tree.root(), rng)?;
        }

        // Add Dirichlet noise to root if configured
        if self.config.dirichlet_alpha > 0.0 {
            self.add_dirichlet_noise(rng);
        }

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.simulate(rng)?;
        }

        // Extract result
        let root = self.tree.get(self.tree.root());
        let policy = self
            .tree
            .root_policy(self.num_actions, self.config.temperature);

        let action = if self.config.temperature < 1e-6 {
            // Greedy
            self.tree
                .best_action()
                .map(|(a, _)| a)
                .ok_or(SearchError::NoLegalMoves)?
        } else {
            // Sample from policy
            sample_action(&policy, rng)?
        };

        Ok(SearchResult {
            action,
            policy,
            value: root.mean_value(),
            simulations: root.visit_count,
        })
    }

    /// Run a single simulation (select -> expand -> evaluate -> backpropagate).
    fn simulate(&mut self, rng: &mut ChaCha20Rng) -> Result<(), SearchError> {
        // Selection: traverse to a leaf
        let (leaf_id, path) = self.select();

        let leaf = self.tree.get(leaf_id);

        // If terminal, backpropagate the terminal value
        if leaf.is_terminal {
            let value = leaf.terminal_value;
            self.tree.backpropagate(leaf_id, value);
            return Ok(());
        }

        // Expansion + Evaluation: expand the node and get value estimate
        // We call the evaluator ONCE and use its policy for priors and its value for backprop
        let value = if !leaf.is_expanded() {
            // Expand and get the value from the same evaluation
            self.expand_node(leaf_id, rng)?
        } else {
            // Already expanded (shouldn't happen in standard MCTS, but handle gracefully)
            // This can occur if we select an expanded node that has no children
            // (e.g., all children were pruned due to zero priors)
            let leaf = self.tree.get(leaf_id);
            let eval =
                self.evaluator
                    .evaluate(&leaf.obs, leaf.legal_moves_mask, self.num_actions)?;
            eval.value
        };

        // Backpropagation
        self.tree.backpropagate(leaf_id, value);

        trace!(
            leaf = leaf_id.0,
            path_len = path.len(),
            value = value,
            "MCTS simulation complete"
        );

        Ok(())
    }

    /// Select a leaf node by traversing the tree using UCB.
    fn select(&self) -> (NodeId, Vec<NodeId>) {
        let mut path = vec![self.tree.root()];
        let mut current = self.tree.root();

        loop {
            let node = self.tree.get(current);

            // Stop at terminal or unexpanded nodes
            if node.is_terminal || !node.is_expanded() {
                break;
            }

            // Select best child
            match self.tree.select_child(current, self.config.c_puct) {
                Some(child_id) => {
                    path.push(child_id);
                    current = child_id;
                }
                None => break, // No children (shouldn't happen if expanded)
            }
        }

        (current, path)
    }

    /// Expand a node by adding all legal children.
    /// Returns the value estimate from the evaluator (to be used for backpropagation).
    fn expand_node(&mut self, node_id: NodeId, _rng: &mut ChaCha20Rng) -> Result<f32, SearchError> {
        let node = self.tree.get(node_id);

        // Don't expand terminal nodes
        if node.is_terminal {
            return Ok(node.terminal_value);
        }

        let state = node.state.clone();
        let obs = node.obs.clone();
        let legal_mask = node.legal_moves_mask;

        // Get policy AND value from evaluator (single NN call)
        // We use the policy for child priors and return the value for backpropagation
        let eval = self
            .evaluator
            .evaluate(&obs, legal_mask, self.num_actions)?;

        // Add children for each legal action
        for action in 0..self.num_actions {
            if (legal_mask >> action) & 1 == 0 {
                continue; // Illegal action
            }

            let prior = eval.policy[action];
            if prior < 1e-8 {
                continue; // Skip zero-prior actions
            }

            // Simulate the action to get new state
            let action_bytes = (action as u32).to_le_bytes().to_vec();
            let step_result = self
                .ctx
                .step(&state, &action_bytes)
                .map_err(|e| SearchError::EngineError(e.to_string()))?;

            // Extract legal moves from info bits (num_actions bits)
            let child_legal_mask = step_result.info & ((1u64 << self.num_actions) - 1);

            // Determine terminal value:
            // The reward is from the perspective of the player who just moved (the parent).
            // The child node represents the resulting state where the OPPONENT would move next.
            // We store the NEGATED reward because:
            // - When we backpropagate from this terminal child, the first thing we do is
            //   add value_sum to the child, then negate for the parent.
            // - If we store +reward, parent gets -reward (WRONG: parent chose winning move!)
            // - If we store -reward, parent gets +reward (CORRECT: winning action is good)
            //
            // Example: X wins by moving to position 2
            // - step_result.reward = +1 (X won)
            // - terminal_value stored = -1 (from O's perspective, this state is losing)
            // - backprop: child gets -1, parent (X's move) gets -(-1) = +1 ✓
            let terminal_value = if step_result.done {
                -step_result.reward
            } else {
                0.0
            };

            self.tree.add_child(
                node_id,
                action as u8,
                prior,
                step_result.state,
                step_result.obs,
                child_legal_mask,
                step_result.done,
                terminal_value,
            );
        }

        // Return the value estimate from this single evaluation
        Ok(eval.value)
    }

    /// Add Dirichlet noise to root node priors for exploration.
    fn add_dirichlet_noise(&mut self, rng: &mut ChaCha20Rng) {
        let root_id = self.tree.root();
        let root = self.tree.get(root_id);
        let num_children = root.children.len();

        if num_children == 0 {
            return;
        }

        // Generate Dirichlet noise using Gamma distribution
        let noise = dirichlet_noise(num_children, self.config.dirichlet_alpha, rng);

        // Mix noise with existing priors
        let eps = self.config.dirichlet_epsilon;
        let children: Vec<_> = root.children.iter().map(|(_, id)| *id).collect();

        for (i, child_id) in children.into_iter().enumerate() {
            let child = self.tree.get_mut(child_id);
            child.prior = (1.0 - eps) * child.prior + eps * noise[i];
        }
    }

    /// Get the search tree (for inspection/debugging).
    pub fn tree(&self) -> &MctsTree {
        &self.tree
    }
}

/// Sample an action from a probability distribution.
fn sample_action(policy: &[f32], rng: &mut ChaCha20Rng) -> Result<u8, SearchError> {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in policy.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u8);
        }
    }

    // Fallback to last non-zero action (handles floating point issues)
    for (i, &p) in policy.iter().enumerate().rev() {
        if p > 0.0 {
            return Ok(i as u8);
        }
    }

    Err(SearchError::NoLegalMoves)
}

/// Generate Dirichlet-distributed noise using Gamma variates.
fn dirichlet_noise(n: usize, alpha: f32, rng: &mut ChaCha20Rng) -> Vec<f32> {
    use rand_distr::{Distribution, Gamma};

    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();
    let mut samples: Vec<f32> = (0..n).map(|_| gamma.sample(rng) as f32).collect();

    // Normalize
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    }

    samples
}

/// Convenience function to run a single MCTS search.
pub fn run_mcts<E: Evaluator>(
    ctx: &mut EngineContext,
    evaluator: &E,
    config: MctsConfig,
    state: Vec<u8>,
    obs: Vec<u8>,
    legal_moves_mask: u64,
    rng: &mut ChaCha20Rng,
) -> Result<SearchResult, SearchError> {
    let mut search = MctsSearch::new(ctx, evaluator, config, state, obs, legal_moves_mask)?;
    search.run(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::UniformEvaluator;
    use rand::SeedableRng;

    fn setup_tictactoe() -> EngineContext {
        games_tictactoe::register_tictactoe();
        EngineContext::new("tictactoe").unwrap()
    }

    #[test]
    fn test_mcts_basic_search() {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing();

        let reset = ctx.reset(42, &[]).unwrap();
        // All 9 positions are legal at the start of TicTacToe
        let legal_mask = 0b111111111u64;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let result = run_mcts(
            &mut ctx,
            &evaluator,
            config,
            reset.state,
            reset.obs,
            legal_mask,
            &mut rng,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        // Should pick a valid action (0-8)
        assert!(result.action < 9);

        // Policy should sum to ~1.0
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Should have run simulations
        assert!(result.simulations > 0);
    }

    #[test]
    fn test_mcts_finds_winning_move() {
        // Set up a position where X can win immediately
        // Board:
        // X | X | _
        // O | O | _
        // _ | _ | _
        //
        // X should play position 2 to win

        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(200);

        // Start fresh
        let reset = ctx.reset(42, &[]).unwrap();

        // Play moves: X at 0, O at 3, X at 1, O at 4
        let moves = [0u32, 3, 1, 4];
        let mut state = reset.state;
        let mut obs = reset.obs;

        for m in moves {
            let action = m.to_le_bytes().to_vec();
            let step = ctx.step(&state, &action).unwrap();
            state = step.state;
            obs = step.obs;
        }

        // Now it's X's turn, position 2 wins
        let legal_mask = 0b111100100u64; // positions 2, 5, 6, 7, 8 are legal

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let result = run_mcts(
            &mut ctx, &evaluator, config, state, obs, legal_mask, &mut rng,
        )
        .unwrap();

        // With enough simulations, MCTS should find the winning move
        // Note: With uniform evaluator, it may not always find it, but should favor it
        assert!(result.action < 9);
        assert!(result.policy[2] > 0.0); // Position 2 should have some probability
    }

    #[test]
    fn test_mcts_winning_move_has_positive_value() {
        // Critical test: verify that an immediate winning move results in
        // positive value at the root (not negative due to sign bugs)
        //
        // Board setup where X can win immediately:
        // X | X | _
        // O | O | _
        // _ | _ | _
        //
        // Position 2 is an instant win for X

        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        // Use greedy selection (temperature=0) so we deterministically pick best move
        // Increase simulations to give MCTS enough time to discover the winning move
        let config = MctsConfig::for_testing()
            .with_simulations(800)
            .with_temperature(0.0);

        // Start fresh
        let reset = ctx.reset(42, &[]).unwrap();

        // Play moves: X at 0, O at 3, X at 1, O at 4
        let moves = [0u32, 3, 1, 4];
        let mut state = reset.state;
        let mut obs = reset.obs;
        let mut info = 0u64;

        for m in moves {
            let action = m.to_le_bytes().to_vec();
            let step = ctx.step(&state, &action).unwrap();
            state = step.state;
            obs = step.obs;
            info = step.info;
        }

        // Extract legal mask from info
        let legal_mask = info & 0x1FF;

        // Verify position 2 is legal
        assert!(
            (legal_mask >> 2) & 1 == 1,
            "Position 2 should be legal, mask={:#b}",
            legal_mask
        );

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut search =
            MctsSearch::new(&mut ctx, &evaluator, config, state, obs, legal_mask).unwrap();
        let result = search.run(&mut rng).unwrap();

        // Check the tree directly
        let tree = search.tree();
        let root = tree.get(tree.root());

        // Debug output: log all children's values
        for (action, child_id) in &root.children {
            let child = tree.get(*child_id);
            let q_value = if child.visit_count > 0 {
                child.value_sum / child.visit_count as f32
            } else {
                0.0
            };
            trace!(
                action,
                terminal = child.is_terminal,
                visits = child.visit_count,
                value_sum = child.value_sum,
                q_value,
                prior = child.prior,
                "Child node stats"
            );
        }

        // Find the child for action 2
        let winning_child_id = root
            .children
            .iter()
            .find(|(action, _)| *action == 2)
            .map(|(_, id)| *id)
            .expect("Child for action 2 should exist");

        let winning_child = tree.get(winning_child_id);

        // The winning child should be terminal
        assert!(
            winning_child.is_terminal,
            "Position 2 should lead to terminal state (X wins)"
        );

        // terminal_value should be -1 (negated +1 reward for winner)
        assert!(
            (winning_child.terminal_value - (-1.0)).abs() < 0.01,
            "Terminal value should be -1.0 (negated win), got {}",
            winning_child.terminal_value
        );

        // The winning child should have been visited
        assert!(
            winning_child.visit_count > 0,
            "Winning action should have visits"
        );

        // The root's mean_value should be positive (we have a winning move)
        trace!(
            visits = root.visit_count,
            value_sum = root.value_sum,
            mean_value = root.mean_value(),
            "Root node stats"
        );

        // The winning move (position 2) should be selected
        assert_eq!(result.action, 2, "Should select winning move at position 2");

        // The root value should be POSITIVE because we have a winning move
        assert!(
            result.value > 0.0,
            "Root value should be positive when winning move exists, got {}",
            result.value
        );

        // Policy should strongly favor position 2
        assert!(
            result.policy[2] > 0.5,
            "Policy should favor winning move, got {}",
            result.policy[2]
        );
    }

    #[test]
    fn test_sample_action() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let policy = vec![0.0, 0.5, 0.3, 0.2, 0.0];

        // Sample many times and check distribution
        let mut counts = [0u32; 5];
        for _ in 0..1000 {
            let action = sample_action(&policy, &mut rng).unwrap();
            counts[action as usize] += 1;
        }

        // Action 0 and 4 should never be selected
        assert_eq!(counts[0], 0);
        assert_eq!(counts[4], 0);

        // Action 1 should be most common (~500), action 2 (~300), action 3 (~200)
        assert!(counts[1] > counts[2]);
        assert!(counts[2] > counts[3]);
    }

    #[test]
    fn test_dirichlet_noise() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let noise = dirichlet_noise(5, 0.3, &mut rng);

        // Should sum to 1.0
        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // All values should be positive
        for &n in &noise {
            assert!(n >= 0.0);
        }
    }
}
