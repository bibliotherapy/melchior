"""Training entry point for hierarchical GMFCS classifier.

Runs hierarchical 2-stage training with patient-level cross-validation.
Trains Stage 1 (ambulatory routing), Stage 2-A (L1/L2/L3-L4), and
Stage 2-B (L3-L4/L5) sequentially.

Usage:
    # Single GPU
    python scripts/05_train.py

    # Multi-GPU with DDP (recommended for V100x2)
    torchrun --nproc_per_node=2 scripts/05_train.py

    # Custom config
    python scripts/05_train.py --config configs/default.yaml

    # Train single stage
    python scripts/05_train.py --stage stage1

    # Quick test (fewer epochs/folds)
    python scripts/05_train.py --quick
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_ddp():
    """Initialize DDP if launched with torchrun.

    Returns:
        (rank, world_size, is_ddp)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        logger.info("DDP initialized: rank=%d, world_size=%d, local_rank=%d",
                     rank, world_size, local_rank)
        return rank, world_size, True

    return 0, 1, False


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical GMFCS classifier"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["stage1", "stage2a", "stage2b"],
                        help="Train single stage (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 epochs, 2 folds")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for models")
    parser.add_argument("--train-patients", type=str, nargs="+", default=None,
                        help="Fixed train set patient IDs (skip CV)")
    parser.add_argument("--val-patients", type=str, nargs="+", default=None,
                        help="Fixed validation set patient IDs")
    parser.add_argument("--test-patients", type=str, nargs="+", default=None,
                        help="Fixed test set patient IDs")
    args = parser.parse_args()

    # DDP setup
    rank, world_size, is_ddp = setup_ddp()
    is_main = rank == 0

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Quick mode overrides
    if args.quick:
        config.setdefault("training", {})
        config["training"]["epochs"] = 5
        config["training"]["cv_folds"] = 2
        config["training"]["patience"] = 3
        logger.info("Quick mode: 5 epochs, 2 folds, patience=3")

    # DDP: adjust batch size per GPU
    if is_ddp:
        total_batch = config.get("training", {}).get("batch_size", 32)
        per_gpu_batch = max(1, total_batch // world_size)
        config["training"]["batch_size"] = per_gpu_batch
        logger.info("DDP: per-GPU batch size = %d (total %d / %d GPUs)",
                     per_gpu_batch, total_batch, world_size)

    # DDP: use local_rank device
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        config["device"] = f"cuda:{local_rank}"

    if args.output_dir:
        config["_output_dir_override"] = args.output_dir

    # Import trainer (after sys.path setup)
    from src.model.train import HierarchicalTrainer

    # Initialize trainer
    trainer = HierarchicalTrainer(config)

    if args.output_dir:
        trainer.output_dir = Path(args.output_dir)
        trainer.log_dir = Path(args.output_dir) / "logs"

    # Build fixed split if specified
    fixed_split = None
    if args.train_patients:
        fixed_split = {
            "train": args.train_patients,
            "val": args.val_patients or [],
            "test": args.test_patients or [],
        }
        logger.info("Fixed split: train=%s, val=%s, test=%s",
                     fixed_split["train"], fixed_split["val"], fixed_split["test"])

    # Train
    if args.stage:
        # Single stage
        logger.info("Training single stage: %s", args.stage)
        results = trainer.train_stage(args.stage, fixed_split=fixed_split)
        if is_main:
            logger.info("Stage %s accuracy: %.1f%%",
                         args.stage, results.get("overall_acc", 0) * 100)
    else:
        # All stages
        results = trainer.train_all(fixed_split=fixed_split)

    # Cleanup DDP
    if is_ddp:
        torch.distributed.destroy_process_group()

    if is_main:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
