# Reference Images

Place 8-20 high-quality images of your target subject here. These are used by `evaluate.py` to score how well the LoRA reproduces the subject.

## What Goes Here

- Clear, high-quality photos of the subject
- Varied angles and lighting (but always recognizable)
- No heavy edits, filters, or obstructions

## How They're Used

| Metric | How References Are Used |
|--------|------------------------|
| CLIP-I | Cosine similarity between generated images and the centroid of reference embeddings |
| ArcFace | Face identity vector compared against mean reference face embedding |

## Tips

- More references = more robust scoring (diminishing returns past ~20)
- Don't include images where the subject is barely visible
- These images are never modified — they're read-only inputs to the scorer
- You can reuse some images from `dataset/`, but ideally use separate high-quality shots
