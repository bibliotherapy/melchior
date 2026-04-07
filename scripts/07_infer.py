"""Hierarchical inference on test patient(s).

Loads trained models for all three stages, runs the hierarchical cascade,
and prints predicted GMFCS levels.

Usage:
    python scripts/07_infer.py --patient mkj
    python scripts/07_infer.py --patient mkj ly --config configs/e2e_test.yaml
    python scripts/07_infer.py --patient mkj --model-dir outputs/e2e_test/models
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.classifier import MultiStreamClassifier
from src.model.dataset import GMFCSDataset, _load_ambulatory_status, _load_labels
from src.model.lite_stgcn import LiteSTGCN
from src.model.train import collate_fn
from src.utils.naming import clip_id_to_patient

# Stage 2A: class index -> GMFCS level
STAGE2A_RESOLVE = {0: 1, 1: 2, 2: "3/4"}
# Stage 2B: class index -> GMFCS level
STAGE2B_RESOLVE = {0: "3/4", 1: 5}


def load_stage_models(model_dir, stage_name, model_cfg, device):
    """Load ST-GCN + Classifier for a stage from fold_0."""
    fold_dir = Path(model_dir) / stage_name / "fold_0"
    if not fold_dir.exists():
        logger.warning("No model found for %s at %s", stage_name, fold_dir)
        return None, None

    num_classes = {"stage1": 2, "stage2a": 3, "stage2b": 2}[stage_name]

    stgcn = LiteSTGCN.from_config(model_cfg)
    stgcn.load_state_dict(
        torch.load(fold_dir / "stgcn.pt", map_location=device, weights_only=True)
    )

    fusion_cfg = model_cfg.get("fusion", {})
    classifier = MultiStreamClassifier(
        stgcn_dim=stgcn.output_dim,
        skeleton_feature_dim=model_cfg.get("skeleton_feature_dim", 15),
        interaction_feature_dim=model_cfg.get("interaction_feature_dim", 10),
        context_vector_dim=model_cfg.get("context_vector_dim", 18),
        walker_feature_dim=model_cfg.get("walker_feature_dim", 5),
        hidden_dim=fusion_cfg.get("hidden_dim", 64),
        dropout=fusion_cfg.get("dropout", 0.3),
        num_classes=num_classes,
        hierarchical=True,
    )
    classifier.load_state_dict(
        torch.load(fold_dir / "classifier.pt", map_location=device, weights_only=True)
    )

    stgcn = stgcn.to(device).eval()
    classifier = classifier.to(device).eval()
    return stgcn, classifier


@torch.no_grad()
def predict_clips(stgcn, classifier, dataset, stage_arg, device):
    """Run inference on a dataset, return {clip_id: predicted_class}."""
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    predictions = {}
    for batch in loader:
        skeleton = batch["skeleton"].to(device)
        skel_feat = batch["skeleton_features"].to(device)
        inter_feat = batch["interaction_features"].to(device)
        ctx_vec = batch["context_vector"].to(device)
        walker = batch["walker_features"].to(device)
        mask = batch["mask"].to(device)

        embedding = stgcn(skeleton, mask=mask)
        logits = classifier(
            embedding, skel_feat, inter_feat, ctx_vec, walker, stage=stage_arg
        )
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for cid, pred, prob in zip(batch["clip_id"], preds, probs):
            predictions[cid] = {"pred": int(pred), "probs": prob.tolist()}

    return predictions


def resolve_gmfcs(stage1_pred, stage2a_pred, stage2b_pred):
    """Resolve final GMFCS level from hierarchical predictions."""
    if stage1_pred == 0:  # ambulatory
        return STAGE2A_RESOLVE.get(stage2a_pred, "?")
    else:  # non-ambulatory
        return STAGE2B_RESOLVE.get(stage2b_pred, "?")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical GMFCS inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-dir", default="outputs/models",
                        help="Directory containing trained stage models")
    parser.add_argument("--patient", type=str, nargs="+", required=True,
                        help="Patient ID(s) to run inference on")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model_cfg = config.get("model", {})
    features_dir = config.get("features_dir", "data/features")
    skeleton_3d_dir = config.get("skeleton_3d_dir", "data/skeleton_3d")
    metadata_dir = config.get("metadata_dir", "data/metadata")

    labels_path = Path(metadata_dir) / "labels.json"
    ann_path = Path(metadata_dir) / "assistive_annotations.json"
    labels = _load_labels(str(labels_path))
    ambulatory_status = _load_ambulatory_status(str(labels_path), str(ann_path))

    # Discover feature files for requested patients
    feat_dir = Path(features_dir)
    all_clip_ids = sorted(f.stem for f in feat_dir.glob("*.npz"))
    test_clips = [c for c in all_clip_ids
                  if clip_id_to_patient(c) in args.patient]

    if not test_clips:
        logger.error("No feature files found for patients: %s", args.patient)
        sys.exit(1)

    logger.info("Running inference on %d clips: %s", len(test_clips), test_clips)

    # Load models for all stages
    models = {}
    stage_args = {"stage1": 1, "stage2a": "2a", "stage2b": "2b"}
    for stage_name in ["stage1", "stage2a", "stage2b"]:
        stgcn, classifier = load_stage_models(
            args.model_dir, stage_name, model_cfg, device
        )
        if stgcn is not None:
            models[stage_name] = (stgcn, classifier)

    if "stage1" not in models:
        logger.error("Stage 1 model not found — cannot run inference")
        sys.exit(1)

    # Stage 1: predict ambulatory vs non-ambulatory for all clips
    ds_s1 = GMFCSDataset(
        test_clips, features_dir, skeleton_3d_dir, labels,
        stage="stage1", ambulatory_status=ambulatory_status,
    )
    s1_preds = predict_clips(
        *models["stage1"], ds_s1, stage_args["stage1"], device
    )

    # Stage 2A: ambulatory clips
    s2a_preds = {}
    if "stage2a" in models:
        amb_clips = [c for c in test_clips if s1_preds[c]["pred"] == 0]
        if amb_clips:
            ds_s2a = GMFCSDataset(
                amb_clips, features_dir, skeleton_3d_dir, labels,
                stage="stage2a", ambulatory_status=ambulatory_status,
            )
            s2a_preds = predict_clips(
                *models["stage2a"], ds_s2a, stage_args["stage2a"], device
            )

    # Stage 2B: non-ambulatory clips
    s2b_preds = {}
    if "stage2b" in models:
        nonamb_clips = [c for c in test_clips if s1_preds[c]["pred"] == 1]
        if nonamb_clips:
            ds_s2b = GMFCSDataset(
                nonamb_clips, features_dir, skeleton_3d_dir, labels,
                stage="stage2b", ambulatory_status=ambulatory_status,
            )
            s2b_preds = predict_clips(
                *models["stage2b"], ds_s2b, stage_args["stage2b"], device
            )

    # Resolve and display results
    print()
    print("=" * 70)
    print("GMFCS Inference Results")
    print("=" * 70)
    header = f"{'Clip ID':<20} {'Patient':<8} {'True':<6} {'Predicted':<10} {'Route':<12}"
    print(header)
    print("-" * 70)

    for clip_id in test_clips:
        patient = clip_id_to_patient(clip_id)
        true_level = labels.get(patient, "?")
        s1 = s1_preds[clip_id]["pred"]
        route = "ambulatory" if s1 == 0 else "non-ambulatory"

        s2a = s2a_preds.get(clip_id, {}).get("pred", -1)
        s2b = s2b_preds.get(clip_id, {}).get("pred", -1)
        predicted = resolve_gmfcs(s1, s2a, s2b)

        print(f"{clip_id:<20} {patient:<8} L{true_level:<5} L{predicted!s:<9} {route}")

    print("=" * 70)


if __name__ == "__main__":
    main()
