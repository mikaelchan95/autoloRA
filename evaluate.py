#!/usr/bin/env python3
"""
AutoLoRA — Composite Evaluation Scorer

READ-ONLY. The optimization agent must NOT modify this file.

Scores generated images against reference images using a weighted composite
of CLIP-I (image similarity), CLIP-T (text-image alignment), aesthetic
prediction, ArcFace identity consistency, and a diversity penalty.

Prints a single float (the composite score) to stdout on the last line.
All other output goes to stderr.

Usage:
    python evaluate.py --generated outputs/eval_images/ --reference reference_images/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Weights for composite score — calibrate these with manual experiments before
# running overnight. The sum should equal 1.0.
# ---------------------------------------------------------------------------
WEIGHT_CLIP_I = 0.35  # Image-to-reference similarity
WEIGHT_CLIP_T = 0.25  # Text-image alignment
WEIGHT_AESTHETIC = 0.20  # General image quality
WEIGHT_IDENTITY = 0.15  # Face identity preservation (ArcFace)
WEIGHT_DIVERSITY = 0.05  # Penalize mode collapse

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
    return images


def extract_prompt_from_filename(
    path: Path, prompts_file: Path | None = None
) -> str | None:
    """Best-effort prompt recovery from eval_prompts.txt using seed in filename."""
    if prompts_file is None:
        prompts_file = Path(__file__).parent / "eval_prompts.txt"
    if not prompts_file.exists():
        return None

    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        return None
    target_seed = int(match.group(1))

    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d+)\s*\|\s*(.+)$", line)
            if m and int(m.group(1)) == target_seed:
                return m.group(2).strip()
    return None


# ---------------------------------------------------------------------------
# CLIP-I: Image-to-reference similarity
# ---------------------------------------------------------------------------


def score_clip_image_similarity(
    generated: list[Path],
    references: list[Path],
    model,
    preprocess,
    device: str,
) -> float:
    log("Computing CLIP-I (image similarity)...")
    ref_features = []
    for ref_path in references:
        from PIL import Image

        img = preprocess(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        ref_features.append(feat)
    ref_features = torch.cat(ref_features, dim=0)
    ref_centroid = ref_features.mean(dim=0, keepdim=True)
    ref_centroid = ref_centroid / ref_centroid.norm(dim=-1, keepdim=True)

    similarities = []
    for gen_path in generated:
        from PIL import Image

        img = preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = (feat @ ref_centroid.T).item()
        similarities.append(sim)

    mean_sim = float(np.mean(similarities))
    log(f"  CLIP-I mean similarity: {mean_sim:.4f}")
    return mean_sim


# ---------------------------------------------------------------------------
# CLIP-T: Text-image alignment
# ---------------------------------------------------------------------------


def score_clip_text_alignment(
    generated: list[Path],
    model,
    preprocess,
    tokenizer,
    device: str,
) -> float:
    log("Computing CLIP-T (text-image alignment)...")
    scores = []
    for gen_path in generated:
        prompt = extract_prompt_from_filename(gen_path)
        if prompt is None:
            continue

        from PIL import Image

        img = preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
        text = tokenizer([prompt]).to(device)

        with torch.no_grad(), torch.amp.autocast(device):
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(text)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).item()
        scores.append(sim)

    if not scores:
        log("  WARNING: No prompts matched — CLIP-T score 0.0")
        return 0.0

    mean_score = float(np.mean(scores))
    log(f"  CLIP-T mean alignment: {mean_score:.4f}")
    return mean_score


# ---------------------------------------------------------------------------
# Aesthetic predictor (LAION aesthetic-predictor-v2-5)
# ---------------------------------------------------------------------------


def score_aesthetic(generated: list[Path], device: str) -> float:
    log("Computing aesthetic scores...")
    try:
        from aesthetics_predictor import AestheticsPredictorV2Linear
        from transformers import CLIPProcessor

        aesthetic_model = AestheticsPredictorV2Linear.from_pretrained(
            "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        )
        aesthetic_model = aesthetic_model.to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except ImportError:
        log("  WARNING: aesthetics_predictor not installed — returning 0.5")
        return 0.5

    scores = []
    for gen_path in generated:
        from PIL import Image

        img = Image.open(gen_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            score = aesthetic_model(**inputs).logits.squeeze().item()
        scores.append(score)

    del aesthetic_model
    torch.cuda.empty_cache()

    # Normalize: aesthetic scores are roughly 1-10, map to 0-1
    mean_raw = float(np.mean(scores))
    normalized = np.clip((mean_raw - 1.0) / 9.0, 0.0, 1.0)
    log(f"  Aesthetic mean raw: {mean_raw:.2f}, normalized: {normalized:.4f}")
    return float(normalized)


# ---------------------------------------------------------------------------
# ArcFace identity consistency
# ---------------------------------------------------------------------------


def score_identity(generated: list[Path], references: list[Path]) -> float:
    log("Computing ArcFace identity consistency...")
    try:
        import cv2
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
    except (ImportError, Exception) as e:
        log(f"  WARNING: insightface unavailable ({e}) — returning 0.5")
        return 0.5

    def get_embedding(image_path: Path) -> np.ndarray | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # Use largest face by bbox area
        best = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return best.embedding

    ref_embeddings = [get_embedding(r) for r in references]
    ref_embeddings = [e for e in ref_embeddings if e is not None]
    if not ref_embeddings:
        log("  WARNING: No faces detected in references — returning 0.5")
        return 0.5
    ref_mean = np.mean(ref_embeddings, axis=0)
    ref_mean = ref_mean / np.linalg.norm(ref_mean)

    similarities = []
    for gen_path in generated:
        emb = get_embedding(gen_path)
        if emb is None:
            continue
        emb = emb / np.linalg.norm(emb)
        sim = float(np.dot(emb, ref_mean))
        similarities.append(sim)

    if not similarities:
        log("  WARNING: No faces in generated images — returning 0.0")
        return 0.0

    # ArcFace cosine sim is roughly -1 to 1; remap to 0-1
    mean_sim = float(np.mean(similarities))
    normalized = np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0)
    log(f"  Identity mean sim: {mean_sim:.4f}, normalized: {normalized:.4f}")
    return normalized


# ---------------------------------------------------------------------------
# Diversity penalty: penalize outputs that all look the same
# ---------------------------------------------------------------------------


def score_diversity(
    generated: list[Path],
    model,
    preprocess,
    device: str,
) -> float:
    log("Computing diversity score...")
    if len(generated) < 2:
        return 1.0

    features = []
    for gen_path in generated:
        from PIL import Image

        img = preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat)
    features = torch.cat(features, dim=0)

    # Pairwise cosine similarity matrix
    sim_matrix = features @ features.T
    n = sim_matrix.shape[0]
    # Mean of off-diagonal elements (exclude self-similarity)
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    mean_pairwise = sim_matrix[mask].mean().item()

    # High pairwise sim = low diversity = bad. Invert.
    # Mean pairwise sim for diverse images is ~0.6-0.8.
    # For mode-collapsed outputs it's ~0.95+.
    diversity = 1.0 - np.clip((mean_pairwise - 0.5) / 0.5, 0.0, 1.0)
    log(f"  Mean pairwise similarity: {mean_pairwise:.4f}, diversity: {diversity:.4f}")
    return float(diversity)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def compute_composite_score(
    generated_dir: Path,
    reference_dir: Path,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    generated = load_images(generated_dir)
    references = load_images(reference_dir)
    log(f"Generated images: {len(generated)}")
    log(f"Reference images: {len(references)}")

    if not generated:
        log("ERROR: No generated images found")
        return 0.0
    if not references:
        log("ERROR: No reference images found")
        return 0.0

    # Load CLIP once, reuse for CLIP-I, CLIP-T, and diversity
    import open_clip

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device, precision="fp16"
    )
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    clip_i = score_clip_image_similarity(
        generated, references, clip_model, clip_preprocess, device
    )
    clip_t = score_clip_text_alignment(
        generated, clip_model, clip_preprocess, clip_tokenizer, device
    )
    diversity = score_diversity(generated, clip_model, clip_preprocess, device)

    # Free CLIP before loading other models
    del clip_model
    torch.cuda.empty_cache()

    aesthetic = score_aesthetic(generated, device)
    identity = score_identity(generated, references)

    composite = (
        WEIGHT_CLIP_I * clip_i
        + WEIGHT_CLIP_T * clip_t
        + WEIGHT_AESTHETIC * aesthetic
        + WEIGHT_IDENTITY * identity
        + WEIGHT_DIVERSITY * diversity
    )

    log(f"\n{'=' * 50}")
    log(f"  CLIP-I:     {clip_i:.4f}  × {WEIGHT_CLIP_I} = {WEIGHT_CLIP_I * clip_i:.4f}")
    log(f"  CLIP-T:     {clip_t:.4f}  × {WEIGHT_CLIP_T} = {WEIGHT_CLIP_T * clip_t:.4f}")
    log(
        f"  Aesthetic:  {aesthetic:.4f}  × {WEIGHT_AESTHETIC} = {WEIGHT_AESTHETIC * aesthetic:.4f}"
    )
    log(
        f"  Identity:   {identity:.4f}  × {WEIGHT_IDENTITY} = {WEIGHT_IDENTITY * identity:.4f}"
    )
    log(
        f"  Diversity:  {diversity:.4f}  × {WEIGHT_DIVERSITY} = {WEIGHT_DIVERSITY * diversity:.4f}"
    )
    log(f"  ─────────────────────────────────")
    log(f"  COMPOSITE:  {composite:.6f}")
    log(f"{'=' * 50}")

    return composite


def main():
    parser = argparse.ArgumentParser(description="AutoLoRA Composite Evaluator")
    parser.add_argument(
        "--generated", type=str, required=True, help="Directory of generated images"
    )
    parser.add_argument(
        "--reference", type=str, required=True, help="Directory of reference images"
    )
    args = parser.parse_args()

    generated_dir = Path(args.generated)
    reference_dir = Path(args.reference)

    if not generated_dir.exists():
        log(f"ERROR: Generated dir not found: {generated_dir}")
        sys.exit(1)
    if not reference_dir.exists():
        log(f"ERROR: Reference dir not found: {reference_dir}")
        sys.exit(1)

    score = compute_composite_score(generated_dir, reference_dir)

    # Final line to stdout — this is what run_experiment.py reads
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
