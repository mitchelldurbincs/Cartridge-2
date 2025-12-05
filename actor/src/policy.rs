//! Action selection policies for the actor

use anyhow::{anyhow, Result};
use engine_core::typed::ActionSpace;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

/// Trait for action selection policies.
/// Currently unused but kept for potential future policy abstractions.
#[allow(dead_code)]
pub trait Policy: Send + Sync {
    /// Select an action given an observation
    fn select_action(&mut self, observation: &[u8]) -> Result<Vec<u8>>;
}

/// Random policy that selects actions uniformly at random.
/// Currently unused (MctsPolicy handles random fallback internally),
/// but kept for testing action space encoding and potential future use.
#[allow(dead_code)]
#[derive(Debug)]
pub struct RandomPolicy {
    rng: ChaCha20Rng,
    action_space: ActionSpace,
}

#[allow(dead_code)]
impl RandomPolicy {
    pub fn new(action_space: ActionSpace) -> Self {
        let rng = ChaCha20Rng::from_entropy();
        Self { rng, action_space }
    }

    pub fn with_seed(action_space: ActionSpace, seed: u64) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        Self { rng, action_space }
    }
}

impl Policy for RandomPolicy {
    fn select_action(&mut self, _observation: &[u8]) -> Result<Vec<u8>> {
        match &self.action_space {
            ActionSpace::Discrete(n) => {
                if *n == 0 {
                    return Err(anyhow!("Discrete action space must have n > 0"));
                }
                let action = self.rng.gen_range(0..*n);
                Ok(action.to_le_bytes().to_vec())
            }
            ActionSpace::MultiDiscrete(nvec) => {
                let mut action_bytes = Vec::new();
                for &n in nvec {
                    if n == 0 {
                        return Err(anyhow!("Multi-discrete action space must have all n > 0"));
                    }
                    let action = self.rng.gen_range(0..n);
                    action_bytes.extend_from_slice(&action.to_le_bytes());
                }
                Ok(action_bytes)
            }
            ActionSpace::Continuous { low, high, .. } => {
                if low.len() != high.len() {
                    return Err(anyhow!(
                        "Continuous action space low and high bounds must have same length"
                    ));
                }
                let mut action_bytes = Vec::new();
                for (&low_val, &high_val) in low.iter().zip(high.iter()) {
                    if low_val >= high_val {
                        return Err(anyhow!(
                            "Continuous action space low bound must be less than high bound"
                        ));
                    }
                    let action: f32 = self.rng.gen_range(low_val..high_val);
                    action_bytes.extend_from_slice(&action.to_le_bytes());
                }
                Ok(action_bytes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_action_space() {
        let mut policy = RandomPolicy::with_seed(ActionSpace::Discrete(4), 42);

        for _ in 0..10 {
            let action_bytes = policy.select_action(&[]).unwrap();
            assert_eq!(action_bytes.len(), 4); // u32 = 4 bytes
            let action = u32::from_le_bytes(action_bytes.try_into().unwrap());
            assert!(action < 4);
        }
    }

    #[test]
    fn test_multi_discrete_action_space() {
        let mut policy = RandomPolicy::with_seed(ActionSpace::MultiDiscrete(vec![2, 3, 4]), 42);

        for _ in 0..10 {
            let action_bytes = policy.select_action(&[]).unwrap();
            assert_eq!(action_bytes.len(), 12); // 3 * u32 = 12 bytes

            let action1 = u32::from_le_bytes(action_bytes[0..4].try_into().unwrap());
            let action2 = u32::from_le_bytes(action_bytes[4..8].try_into().unwrap());
            let action3 = u32::from_le_bytes(action_bytes[8..12].try_into().unwrap());

            assert!(action1 < 2);
            assert!(action2 < 3);
            assert!(action3 < 4);
        }
    }

    #[test]
    fn test_continuous_action_space() {
        let mut policy = RandomPolicy::with_seed(
            ActionSpace::Continuous {
                low: vec![-1.0, 0.0],
                high: vec![1.0, 2.0],
                shape: vec![2],
            },
            42,
        );

        for _ in 0..10 {
            let action_bytes = policy.select_action(&[]).unwrap();
            assert_eq!(action_bytes.len(), 8); // 2 * f32 = 8 bytes

            let action1 = f32::from_le_bytes(action_bytes[0..4].try_into().unwrap());
            let action2 = f32::from_le_bytes(action_bytes[4..8].try_into().unwrap());

            assert!(action1 >= -1.0 && action1 < 1.0);
            assert!(action2 >= 0.0 && action2 < 2.0);
        }
    }

    #[test]
    fn test_policy_determinism_with_same_seed() {
        // Create two policies with the same seed
        let mut policy1 = RandomPolicy::with_seed(ActionSpace::Discrete(10), 12345);
        let mut policy2 = RandomPolicy::with_seed(ActionSpace::Discrete(10), 12345);

        // Both policies should produce the same sequence of actions
        for _ in 0..20 {
            let action1 = policy1.select_action(&[]).unwrap();
            let action2 = policy2.select_action(&[]).unwrap();
            assert_eq!(
                action1, action2,
                "policies with same seed should produce same actions"
            );
        }
    }

    #[test]
    fn test_policy_different_with_different_seeds() {
        let mut policy1 = RandomPolicy::with_seed(ActionSpace::Discrete(10), 11111);
        let mut policy2 = RandomPolicy::with_seed(ActionSpace::Discrete(10), 22222);

        // With different seeds, at least one action should differ in 20 samples
        let mut found_difference = false;
        for _ in 0..20 {
            let action1 = policy1.select_action(&[]).unwrap();
            let action2 = policy2.select_action(&[]).unwrap();
            if action1 != action2 {
                found_difference = true;
                break;
            }
        }
        assert!(
            found_difference,
            "policies with different seeds should produce different actions"
        );
    }

    #[test]
    fn test_discrete_action_space_with_n_zero_fails() {
        let mut policy = RandomPolicy::with_seed(ActionSpace::Discrete(0), 42);

        let result = policy.select_action(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must have n > 0"));
    }

    #[test]
    fn test_multi_discrete_with_zero_element_fails() {
        let mut policy = RandomPolicy::with_seed(ActionSpace::MultiDiscrete(vec![2, 0, 4]), 42);

        let result = policy.select_action(&[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must have all n > 0"));
    }

    #[test]
    fn test_continuous_action_space_invalid_bounds_fails() {
        let mut policy = RandomPolicy::with_seed(
            ActionSpace::Continuous {
                low: vec![1.0, 0.0], // low >= high for first element
                high: vec![0.0, 2.0],
                shape: vec![2],
            },
            42,
        );

        let result = policy.select_action(&[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("low bound must be less than high bound"));
    }

    #[test]
    fn test_continuous_action_space_mismatched_bounds_fails() {
        let mut policy = RandomPolicy::with_seed(
            ActionSpace::Continuous {
                low: vec![-1.0, 0.0],
                high: vec![1.0, 2.0, 3.0], // Different length
                shape: vec![2],
            },
            42,
        );

        let result = policy.select_action(&[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must have same length"));
    }
}
