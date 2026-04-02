"""Lightweight Spatial-Temporal Graph Convolutional Network.

3-layer ST-GCN (64 channels, ~200K params) operating on raw 3D skeleton
input (T, 17, 3). Produces a 128D embedding per clip.

Architecture:
    Input: (B, T, 17, 3) -> reshape to (B, 3, T, 17)
    -> STGCNBlock(3 -> 64) -> STGCNBlock(64 -> 64) -> STGCNBlock(64 -> 128)
    -> Global average pooling over T and V (joints)
    -> Output: (B, 128)

Graph: COCO 17-joint skeleton with normalized adjacency matrix.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── COCO 17-joint skeleton graph ─────────────────────────────────────────

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head: nose-eyes-ears
    (5, 6),                                   # shoulder bridge
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso sides
    (11, 12),                                 # hip bridge
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

NUM_JOINTS = 17


def build_adjacency_matrix():
    """Build normalized adjacency matrix for COCO 17-joint graph.

    A_hat = D^(-1/2) @ (A + I) @ D^(-1/2)

    Returns:
        torch.FloatTensor of shape (17, 17).
    """
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Add self-loops
    A_hat = A + np.eye(NUM_JOINTS, dtype=np.float32)

    # Symmetric normalization: D^(-1/2) A_hat D^(-1/2)
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal() + 1e-8))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    return torch.from_numpy(A_norm)


# ── ST-GCN building blocks ──────────────────────────────────────────────

class SpatialGraphConv(nn.Module):
    """Spatial graph convolution: A @ X @ W.

    Applies the graph structure via the adjacency matrix and learns
    a linear transform of the features.

    Args:
        in_channels: input feature dimension.
        out_channels: output feature dimension.
        adjacency: (V, V) normalized adjacency matrix.
    """

    def __init__(self, in_channels, out_channels, adjacency):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) where V = num_joints.

        Returns:
            (B, C_out, T, V).
        """
        # Graph convolution: aggregate features from neighboring joints
        # x @ A^T: (B, C, T, V) @ (V, V) -> (B, C, T, V)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        # Channel transform: 1x1 conv over (T, V) spatial dims
        x = self.conv(x)
        return x


class TemporalConv(nn.Module):
    """Temporal convolution along the time axis.

    Applies a 1D convolution across the temporal dimension with padding
    to maintain sequence length.

    Args:
        in_channels: input feature dimension.
        out_channels: output feature dimension.
        kernel_size: temporal kernel size (default 9).
    """

    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V).

        Returns:
            (B, C_out, T, V).
        """
        return self.conv(x)


class STGCNBlock(nn.Module):
    """One ST-GCN block = SpatialGraphConv + TemporalConv + residual + BN + ReLU.

    Args:
        in_channels: input feature dimension.
        out_channels: output feature dimension.
        adjacency: (V, V) normalized adjacency matrix.
        temporal_kernel: temporal conv kernel size.
        dropout: dropout rate.
        stride: temporal stride (1 = no downsampling).
    """

    def __init__(self, in_channels, out_channels, adjacency,
                 temporal_kernel=9, dropout=0.3, stride=1):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.bn_spatial = nn.BatchNorm2d(out_channels)

        self.temporal = TemporalConv(out_channels, out_channels, temporal_kernel)
        self.bn_temporal = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection: match dimensions if needed
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T, V).

        Returns:
            (B, C_out, T, V).
        """
        res = self.residual(x)

        # Spatial graph convolution
        x = self.spatial(x)
        x = self.bn_spatial(x)
        x = self.relu(x)

        # Temporal convolution
        x = self.temporal(x)
        x = self.bn_temporal(x)

        # Residual + activation
        x = self.relu(x + res)
        x = self.dropout(x)
        return x


# ── Lite ST-GCN ─────────────────────────────────────────────────────────

class LiteSTGCN(nn.Module):
    """Lightweight ST-GCN encoder for 3D skeleton sequences.

    Config (from default.yaml):
        num_layers: 3
        channels: [3, 64, 64, 128]
        temporal_kernel: 9
        num_joints: 17
        dropout: 0.3

    Forward:
        Input: (B, T, 17, 3) raw 3D skeleton
        -> Reshape to (B, 3, T, 17) [channel-first for conv]
        -> 3x STGCNBlock
        -> Global average pooling over time and joints
        -> Output: (B, 128)

    Estimated parameters: ~200K
    """

    def __init__(self, in_channels=3, channels=None, temporal_kernel=9,
                 num_joints=17, dropout=0.3):
        """
        Args:
            in_channels: input channels (3 for xyz coordinates).
            channels: list of channel dimensions per layer.
                Default: [64, 64, 128] (3 layers).
            temporal_kernel: kernel size for temporal convolution.
            num_joints: number of skeleton joints (17 for COCO).
            dropout: dropout rate for ST-GCN blocks.
        """
        super().__init__()

        if channels is None:
            channels = [64, 64, 128]

        self.num_joints = num_joints
        self.output_dim = channels[-1]

        # Build adjacency matrix
        A = build_adjacency_matrix()

        # Input batch normalization
        self.bn_input = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN blocks
        self.blocks = nn.ModuleList()
        layer_channels = [in_channels] + channels
        for i in range(len(channels)):
            self.blocks.append(
                STGCNBlock(
                    in_channels=layer_channels[i],
                    out_channels=layer_channels[i + 1],
                    adjacency=A,
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
            )

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: (B, T, V, C) raw 3D skeleton coordinates.
                V = num_joints, C = 3 (xyz).
            mask: optional (B, T) padding mask. 1 = real, 0 = padding.
                Used to zero out padded frames before pooling.

        Returns:
            (B, output_dim) skeleton embedding.
        """
        B, T, V, C = x.shape

        # Reshape to channel-first: (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)

        # Input batch normalization
        # Reshape to (B, C*V, T) for BN1d, then back
        x_bn = x.reshape(B, C * V, T)
        x_bn = self.bn_input(x_bn)
        x = x_bn.reshape(B, C, T, V)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x)  # (B, C_out, T, V)

        # Apply mask before pooling to ignore padded frames
        if mask is not None:
            # mask: (B, T) -> (B, 1, T, 1) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(3)
            x = x * mask_expanded
            # Average over non-padded frames only
            n_real = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            # Pool over joints first: (B, C, T, V) -> (B, C, T)
            x = x.mean(dim=3)
            # Pool over time with mask: (B, C, T) -> (B, C)
            x = x.sum(dim=2) / (n_real * 1.0)  # n_real: (B, 1)
        else:
            # Global average pooling over time and joints
            x = x.mean(dim=[2, 3])  # (B, C)

        return x

    @classmethod
    def from_config(cls, config):
        """Create LiteSTGCN from config dict (default.yaml model.stgcn section).

        Args:
            config: dict with keys from configs/default.yaml model.stgcn.

        Returns:
            LiteSTGCN instance.
        """
        stgcn_cfg = config.get("stgcn", config)
        channels_full = stgcn_cfg.get("channels", [3, 64, 64, 128])
        in_channels = channels_full[0]
        layer_channels = channels_full[1:]

        return cls(
            in_channels=in_channels,
            channels=layer_channels,
            temporal_kernel=stgcn_cfg.get("temporal_kernel", 9),
            num_joints=stgcn_cfg.get("num_joints", 17),
            dropout=stgcn_cfg.get("dropout", 0.3),
        )
