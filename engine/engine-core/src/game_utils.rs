//! Shared utilities for two-player game implementations
//!
//! This module provides common functionality used across multiple game implementations
//! to reduce code duplication and ensure consistent behavior.

/// Calculate reward for a two-player zero-sum game.
///
/// Returns the reward from the perspective of the player who just moved.
///
/// # Arguments
/// * `winner` - Winner indicator: 0=ongoing, 1=player1 wins, 2=player2 wins, 3=draw
/// * `previous_player` - The player who made the move (1 or 2)
///
/// # Returns
/// * `1.0` if the previous player won
/// * `-1.0` if the previous player lost
/// * `0.0` for draws or ongoing games
///
/// # Example
/// ```
/// use engine_core::game_utils::calculate_reward;
///
/// // Player 1 wins, viewed from player 1's perspective
/// assert_eq!(calculate_reward(1, 1), 1.0);
///
/// // Player 1 wins, viewed from player 2's perspective
/// assert_eq!(calculate_reward(1, 2), -1.0);
///
/// // Draw
/// assert_eq!(calculate_reward(3, 1), 0.0);
///
/// // Game ongoing
/// assert_eq!(calculate_reward(0, 1), 0.0);
/// ```
#[inline]
pub fn calculate_reward(winner: u8, previous_player: u8) -> f32 {
    match winner {
        0 => 0.0, // Game ongoing
        1 => {
            if previous_player == 1 {
                1.0
            } else {
                -1.0
            }
        } // Player 1 wins
        2 => {
            if previous_player == 2 {
                1.0
            } else {
                -1.0
            }
        } // Player 2 wins
        3 => 0.0, // Draw
        _ => 0.0, // Shouldn't happen
    }
}

/// Standard bit-field layout constants for game info encoding.
///
/// The info u64 is laid out as follows (little endian bit numbering):
/// * Bits 0-15  : Legal move mask (varies by game: 9 bits for TicTacToe, 7 for Connect4)
/// * Bits 16-19 : Current player (1 = player1, 2 = player2)
/// * Bits 20-23 : Winner (0 = none, 1 = player1, 2 = player2, 3 = draw)
/// * Bits 24-31 : Moves played so far
pub mod info_bits {
    /// Bit position for current player field
    pub const CURRENT_PLAYER_SHIFT: u32 = 16;
    /// Bit position for winner field
    pub const WINNER_SHIFT: u32 = 20;
    /// Bit position for moves played counter
    pub const MOVES_PLAYED_SHIFT: u32 = 24;

    /// Pack game state auxiliary information into a u64 bit-field.
    ///
    /// # Arguments
    /// * `legal_moves_mask` - Bit mask of legal moves (game-specific width)
    /// * `current_player` - Current player (1 or 2)
    /// * `winner` - Winner indicator (0=ongoing, 1=p1, 2=p2, 3=draw)
    /// * `moves_played` - Number of moves played so far
    ///
    /// # Example
    /// ```
    /// use engine_core::game_utils::info_bits::compute_info_bits;
    ///
    /// // TicTacToe: all 9 positions legal, player 1 to move, no winner yet, 0 moves
    /// let info = compute_info_bits(0x1FF, 1, 0, 0);
    /// assert_eq!(info & 0x1FF, 0x1FF);  // Legal moves mask
    /// assert_eq!((info >> 16) & 0xF, 1); // Current player
    /// assert_eq!((info >> 20) & 0xF, 0); // Winner
    /// assert_eq!((info >> 24) & 0xFF, 0); // Moves played
    /// ```
    #[inline]
    pub fn compute_info_bits(
        legal_moves_mask: u64,
        current_player: u8,
        winner: u8,
        moves_played: u64,
    ) -> u64 {
        let mut info = legal_moves_mask;
        info |= (current_player as u64) << CURRENT_PLAYER_SHIFT;
        info |= (winner as u64) << WINNER_SHIFT;
        info |= moves_played << MOVES_PLAYED_SHIFT;
        info
    }

    /// Extract the legal moves mask from info bits
    #[inline]
    pub fn extract_legal_mask(info: u64, mask_width: u32) -> u64 {
        info & ((1u64 << mask_width) - 1)
    }

    /// Extract the current player from info bits
    #[inline]
    pub fn extract_current_player(info: u64) -> u8 {
        ((info >> CURRENT_PLAYER_SHIFT) & 0xF) as u8
    }

    /// Extract the winner from info bits
    #[inline]
    pub fn extract_winner(info: u64) -> u8 {
        ((info >> WINNER_SHIFT) & 0xF) as u8
    }

    /// Extract the moves played count from info bits
    #[inline]
    pub fn extract_moves_played(info: u64) -> u64 {
        (info >> MOVES_PLAYED_SHIFT) & 0xFF
    }
}

/// Encode multiple f32 slices to bytes in little-endian format.
///
/// This is a common pattern for encoding observations that consist of
/// multiple float arrays (board view, legal moves, current player).
///
/// # Arguments
/// * `out` - Output buffer to append bytes to
/// * `slices` - Iterator of f32 slices to encode
///
/// # Example
/// ```
/// use engine_core::game_utils::encode_f32_slices;
///
/// let board = [1.0f32, 0.0, 0.0];
/// let legal = [1.0f32, 1.0, 1.0];
/// let player = [1.0f32, 0.0];
///
/// let mut buf = Vec::new();
/// encode_f32_slices(&mut buf, [&board[..], &legal[..], &player[..]]);
///
/// // Should be 8 floats * 4 bytes = 32 bytes
/// assert_eq!(buf.len(), 32);
/// ```
pub fn encode_f32_slices<'a>(out: &mut Vec<u8>, slices: impl IntoIterator<Item = &'a [f32]>) {
    for slice in slices {
        for &value in slice {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_reward_player1_wins() {
        // Player 1 wins, from player 1's perspective
        assert_eq!(calculate_reward(1, 1), 1.0);
        // Player 1 wins, from player 2's perspective
        assert_eq!(calculate_reward(1, 2), -1.0);
    }

    #[test]
    fn test_calculate_reward_player2_wins() {
        // Player 2 wins, from player 2's perspective
        assert_eq!(calculate_reward(2, 2), 1.0);
        // Player 2 wins, from player 1's perspective
        assert_eq!(calculate_reward(2, 1), -1.0);
    }

    #[test]
    fn test_calculate_reward_draw() {
        assert_eq!(calculate_reward(3, 1), 0.0);
        assert_eq!(calculate_reward(3, 2), 0.0);
    }

    #[test]
    fn test_calculate_reward_ongoing() {
        assert_eq!(calculate_reward(0, 1), 0.0);
        assert_eq!(calculate_reward(0, 2), 0.0);
    }

    #[test]
    fn test_compute_info_bits_tictactoe() {
        // TicTacToe: all positions legal (9 bits), player 1, no winner, 0 moves
        let info = info_bits::compute_info_bits(0x1FF, 1, 0, 0);

        assert_eq!(info_bits::extract_legal_mask(info, 9), 0x1FF);
        assert_eq!(info_bits::extract_current_player(info), 1);
        assert_eq!(info_bits::extract_winner(info), 0);
        assert_eq!(info_bits::extract_moves_played(info), 0);
    }

    #[test]
    fn test_compute_info_bits_connect4() {
        // Connect4: all columns legal (7 bits), player 2, player 1 won, 10 moves
        let info = info_bits::compute_info_bits(0x7F, 2, 1, 10);

        assert_eq!(info_bits::extract_legal_mask(info, 7), 0x7F);
        assert_eq!(info_bits::extract_current_player(info), 2);
        assert_eq!(info_bits::extract_winner(info), 1);
        assert_eq!(info_bits::extract_moves_played(info), 10);
    }

    #[test]
    fn test_encode_f32_slices() {
        let board = [1.0f32, 0.0];
        let legal = [1.0f32];
        let player = [0.0f32, 1.0];

        let mut buf = Vec::new();
        encode_f32_slices(&mut buf, [&board[..], &legal[..], &player[..]]);

        // 5 floats * 4 bytes = 20 bytes
        assert_eq!(buf.len(), 20);

        // Verify first float is 1.0
        let first = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(first, 1.0);

        // Verify last float is 1.0
        let last = f32::from_le_bytes(buf[16..20].try_into().unwrap());
        assert_eq!(last, 1.0);
    }

    #[test]
    fn test_encode_f32_slices_empty() {
        let mut buf = Vec::new();
        encode_f32_slices(&mut buf, std::iter::empty::<&[f32]>());
        assert!(buf.is_empty());
    }
}
