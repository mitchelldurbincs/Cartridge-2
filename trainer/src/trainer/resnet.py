"""AlphaZero-style convolutional ResNet for spatially-structured games.

This module provides a convolutional neural network with residual blocks,
following the AlphaZero architecture. It's designed for games with spatial
structure like Connect4 (7x6) and Othello (8x8).

Architecture:
    Input: (batch, channels, height, width)
    -> Initial Conv Block: Conv2d -> BatchNorm -> ReLU
    -> Residual Tower: N residual blocks with skip connections
    -> Policy Head: Conv2d(1x1) -> BN -> ReLU -> Flatten -> Linear
    -> Value Head: Conv2d(1x1) -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_config import GameConfig


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection.

    Architecture:
        Input
        -> Conv2d(3x3, padding=1) -> BatchNorm -> ReLU
        -> Conv2d(3x3, padding=1) -> BatchNorm
        -> + skip connection
        -> ReLU
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        out = F.relu(out)
        return out


class ConvPolicyValueNetwork(nn.Module):
    """AlphaZero-style convolutional network with residual blocks.

    This network takes spatial input and uses convolutional layers to
    preserve spatial structure, which is beneficial for board games.

    The observation is expected to be a flat tensor that will be reshaped
    to (batch, input_channels, board_height, board_width).
    """

    def __init__(self, config: GameConfig):
        super().__init__()

        self.config = config
        self.obs_size = config.obs_size
        self.action_size = config.num_actions
        self.board_height = config.board_height
        self.board_width = config.board_width
        self.input_channels = config.input_channels
        self.num_filters = config.num_filters

        # Initial convolutional block
        self.initial_conv = nn.Conv2d(
            config.input_channels, config.num_filters, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm2d(config.num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(config.num_filters) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(config.num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        policy_flatten_size = 2 * config.board_height * config.board_width
        self.policy_fc = nn.Linear(policy_flatten_size, config.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(config.num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        value_flatten_size = config.board_height * config.board_width
        self.value_fc1 = nn.Linear(value_flatten_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights using He initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _reshape_observation(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat observation to spatial format.

        Args:
            x: Flat observation of shape (batch, obs_size)

        Returns:
            Spatial tensor of shape (batch, input_channels, height, width)

        The observation encoding is game-specific:
        - First board_size elements: Player 1's pieces (1 where piece present)
        - Next board_size elements: Player 2's pieces (1 where piece present)
        - Remaining elements: legal mask and player indicator (ignored for CNN input)
        """
        batch_size = x.shape[0]
        board_size = self.board_height * self.board_width

        # Extract board planes from the flat observation
        # Assumption: first input_channels * board_size elements are board planes
        planes = []
        for i in range(self.input_channels):
            start = i * board_size
            end = start + board_size
            plane = x[:, start:end].view(batch_size, self.board_height, self.board_width)
            planes.append(plane)

        # Stack planes along channel dimension: (batch, channels, height, width)
        spatial = torch.stack(planes, dim=1)
        return spatial

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Observation tensor of shape (batch, obs_size) - flat format

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (batch, action_size) - raw logits
                - value: Shape (batch, 1) - value in [-1, 1]
        """
        # Reshape flat observation to spatial format
        spatial = self._reshape_observation(x)

        # Initial conv block
        h = F.relu(self.initial_bn(self.initial_conv(spatial)))

        # Residual tower
        for res_block in self.res_blocks:
            h = res_block(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy probabilities and value.

        Args:
            x: Observation tensor of shape (batch, obs_size)
            legal_mask: Optional mask of shape (batch, action_size)
                        where 1.0 = legal, 0.0 = illegal

        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: Shape (batch, action_size) - probabilities summing to 1
                - value: Shape (batch, 1) - value in [-1, 1]
        """
        policy_logits, value = self.forward(x)

        # Apply legal move mask if provided
        if legal_mask is not None:
            # Set illegal moves to very negative value before softmax
            policy_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))

        policy_probs = F.softmax(policy_logits, dim=-1)

        return policy_probs, value


def create_resnet(config: GameConfig) -> ConvPolicyValueNetwork:
    """Factory function to create a ResNet for the specified game configuration.

    Args:
        config: Game configuration with CNN settings.

    Returns:
        ConvPolicyValueNetwork configured for the specified game.
    """
    return ConvPolicyValueNetwork(config)
