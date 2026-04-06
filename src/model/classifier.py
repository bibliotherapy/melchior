"""Multi-stream fusion and hierarchical classification head.

Fuses five streams:
  A: Lite ST-GCN skeleton embedding (128D)
  B: Temporal-pooled skeleton features (30D = 15 features x mean+std)
  C: Temporal-pooled interaction features (20D = 10 features x mean+std)
  D: Context vector passthrough (18D)
  E: Walker-skeleton spatial features (5D per clip, no temporal pooling needed)
Concat (~201D) -> MLP(64) -> hierarchical classification:
    Stage 1: Ambulatory vs Non-ambulatory (2 classes, routed by w_status)
    Stage 2-A: L1 vs L2 vs L3-L4 (3 classes, ambulatory branch)
    Stage 2-B: L3-L4 vs L5 (2 classes, non-ambulatory branch)

Note: L3 and L4 are merged in both branches because ambulatory status does
not cleanly separate them. Some L3 patients (kku) cannot walk, and some L4
patients (hdi, jrh) can walk with assistance. Final L3/L4 resolution relies
on walker engagement (L3) vs caregiver assistance (L4) features.
"""

import torch
import torch.nn as nn


class MultiStreamClassifier(nn.Module):
    """Hierarchical multi-stream classifier for GMFCS level prediction.

    Supports two-stage hierarchical classification:
        Stage 1: Ambulatory (L1-L3) vs Non-ambulatory (L4-L5)
        Stage 2-A: L1 vs L2 vs L3
        Stage 2-B: L4 vs L5
    Or flat 5-class classification.
    """

    def __init__(self, stgcn_dim=128, skeleton_feature_dim=15,
                 interaction_feature_dim=10, context_vector_dim=18,
                 walker_feature_dim=5, hidden_dim=64, dropout=0.3,
                 num_classes=5, hierarchical=True):
        """
        Args:
            stgcn_dim: Output dimension from Lite ST-GCN (Stream A).
            skeleton_feature_dim: Number of Layer 1 skeleton features.
            interaction_feature_dim: Number of Layer 2 interaction features.
            context_vector_dim: Dimension of Layer 3 context vector (18D).
            walker_feature_dim: Number of walker spatial features (Stream E).
            hidden_dim: Hidden layer dimension in fusion MLP.
            dropout: Dropout rate.
            num_classes: Number of output classes (5 for flat, ignored if hierarchical).
            hierarchical: Use 2-stage hierarchical classification.
        """
        super().__init__()

        # Stream B/C use mean+std temporal pooling -> 2x feature dim
        pooled_skeleton_dim = skeleton_feature_dim * 2   # 30D
        pooled_interaction_dim = interaction_feature_dim * 2  # 20D

        # Total fusion input
        fusion_input_dim = (
            stgcn_dim +            # Stream A: 128D
            pooled_skeleton_dim +  # Stream B: 30D
            pooled_interaction_dim +  # Stream C: 20D
            context_vector_dim +   # Stream D: 18D
            walker_feature_dim     # Stream E: 5D
        )

        self.hierarchical = hierarchical

        if hierarchical:
            # Stage 1: Binary (ambulatory vs non-ambulatory)
            self.stage1_head = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            # Stage 2-A: L1 vs L2 vs L3-L4 (ambulatory branch)
            self.stage2a_head = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
            # Stage 2-B: L3-L4 vs L5 (non-ambulatory branch)
            self.stage2b_head = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.flat_head = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    @staticmethod
    def temporal_pool(features_seq):
        """Mean + std temporal pooling.

        Args:
            features_seq: (T, D) temporal feature sequence.

        Returns:
            (2*D,) concatenated mean and std vectors.
        """
        mean = features_seq.mean(dim=0)
        std = features_seq.std(dim=0)
        return torch.cat([mean, std], dim=0)

    def forward(self, stgcn_embedding, skeleton_features, interaction_features,
                context_vector, walker_features, stage=None):
        """Forward pass through multi-stream fusion.

        Args:
            stgcn_embedding: (B, 128) from Lite ST-GCN (Stream A).
            skeleton_features: (B, T, 15) Layer 1 features (Stream B).
            interaction_features: (B, T, 10) Layer 2 features (Stream C).
            context_vector: (B, 22) Layer 3 metadata (Stream D).
            walker_features: (B, 5) walker spatial features (Stream E).
            stage: For hierarchical mode — 1, "2a", or "2b".
                If None in hierarchical mode, returns all stage outputs.

        Returns:
            If hierarchical and stage specified: logits for that stage.
            If hierarchical and stage=None: dict of all stage logits.
            If flat: (B, num_classes) logits.
        """
        # Stream B: temporal pool skeleton features
        B = stgcn_embedding.shape[0]
        pooled_skeleton = torch.stack([
            self.temporal_pool(skeleton_features[i]) for i in range(B)
        ])

        # Stream C: temporal pool interaction features
        pooled_interaction = torch.stack([
            self.temporal_pool(interaction_features[i]) for i in range(B)
        ])

        # Fuse all streams
        fused = torch.cat([
            stgcn_embedding,     # A: 128D
            pooled_skeleton,     # B: 30D
            pooled_interaction,  # C: 20D
            context_vector,      # D: 22D
            walker_features,     # E: 5D
        ], dim=1)

        if self.hierarchical:
            if stage == 1:
                return self.stage1_head(fused)
            elif stage == "2a":
                return self.stage2a_head(fused)
            elif stage == "2b":
                return self.stage2b_head(fused)
            else:
                return {
                    "stage1": self.stage1_head(fused),
                    "stage2a": self.stage2a_head(fused),
                    "stage2b": self.stage2b_head(fused),
                }
        else:
            return self.flat_head(fused)
