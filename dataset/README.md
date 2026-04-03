# Dataset

Place your training images and captions here.

## Format

Each image needs a matching `.txt` caption file with the same name:

```
dataset/
├── 001.png
├── 001.txt
├── 002.jpg
├── 002.txt
└── ...
```

## Requirements

- **Images**: `.jpg`, `.jpeg`, or `.png`
- **Captions**: `.txt` file per image, same filename stem
- **Count**: 10-30 images is typical for a subject/character LoRA
- **Quality**: High-resolution, well-lit, varied poses/angles/backgrounds
- **Trigger word**: Include your trigger token (default: `sks`) in every caption

## Caption Examples

```
a photo of sks person smiling, indoor lighting
sks person sitting at a desk, side view
close-up portrait of sks person, natural light, shallow depth of field
```

## Tips

- Crop images to focus on the subject
- Include variety: different lighting, angles, expressions, clothing
- Avoid watermarks, heavy filters, or low-resolution images
- Keep captions factual — describe what's visible, not artistic intent
- The trigger word must appear in every caption
