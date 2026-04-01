"""Multi-stream fusion and classification head.

Fuses four streams:
  A: Lite ST-GCN skeleton embedding (128D)
  B: Temporal-pooled skeleton features (30D)
  C: Temporal-pooled interaction features (20D)
  D: Context vector passthrough (18D)
Concat (~196D) -> MLP(64) -> classification head.
"""
