#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare SFT training data from the HuggingFace MMAD dataset.

Usage:
    1. Download the dataset from https://huggingface.co/datasets/jiang-cc/MMAD
    2. Run this script to crop images and update JSON paths

    python prepare_sft_data.py \
        --input dataset/train-sft1600.json \
        --mmap_root /path/to/MMAD \
        --output dataset/train-sft1600-ready.json \
        --crop_dir dataset/sft_crops
"""

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PIL.Image

# Match crop coordinates in tool calls
_CROP_RE = re.compile(
    r'<crop>\s*\[\s*([\d\s.,\-]+)\s*\]\s*</crop>',
    re.IGNORECASE,
)


def parse_crop_from_messages(messages: List[Dict]) -> List[List[float]]:
    """Extract all crop coordinates from assistant turns, normalized to [0,1]."""
    crops = []
    for msg in messages:
        if msg['role'] != 'assistant':
            continue
        content = msg['content']
        for m in _CROP_RE.finditer(content):
            try:
                raw = [float(v.strip()) for v in m.group(1).split(',')]
                if len(raw) == 4:
                    crop = [max(0.0, min(1.0, v / 1000.0)) for v in raw]
                    crops.append(crop)
            except ValueError:
                pass
    return crops


def crop_image(image: PIL.Image.Image, bbox: List[float],
               max_aspect_ratio: float = 150) -> Optional[PIL.Image.Image]:
    """Crop image with aspect ratio constraint."""
    try:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(1., x2), min(1., y2)
        if x1 >= x2 or y1 >= y2:
            return None

        w, h = image.size
        px1, py1 = math.floor(x1 * w), math.floor(y1 * h)
        px2, py2 = max(px1 + 1, math.ceil(x2 * w)), max(py1 + 1, math.ceil(y2 * h))

        crop_w, crop_h = px2 - px1, py2 - py1
        aspect = max(crop_w, crop_h) / max(1, min(crop_w, crop_h))

        if aspect > max_aspect_ratio:
            if crop_w > crop_h:
                target_h = int(crop_w / max_aspect_ratio)
                expand = target_h - crop_h
                py1 = max(0, py1 - expand // 2)
                py2 = min(h, py2 + (expand - (py1 - max(0, py1 - expand // 2))))
            else:
                target_w = int(crop_h / max_aspect_ratio)
                expand = target_w - crop_w
                px1 = max(0, px1 - expand // 2)
                px2 = min(w, px2 + (expand - (px1 - max(0, px1 - expand // 2))))

        cropped = image.crop((px1, py1, px2, py2))
        if max(cropped.size) > 2048:
            scale = 2048 / max(cropped.size)
            cropped = cropped.resize(
                (max(1, int(cropped.width * scale)),
                 max(1, int(cropped.height * scale))),
                PIL.Image.LANCZOS,
            )
        return cropped
    except Exception:
        return None


def process_record(args: Tuple) -> Dict:
    """Crop images and update paths for a single record."""
    record, mmap_root, crop_dir = args
    messages = record['messages']
    old_images = record.get('images', [])

    if not old_images:
        return {'status': 'skip', 'reason': 'no images'}

    original_image = old_images[0]
    # Extract relative path
    rel_path = None
    for marker in ['MMAD_resize_1024/', 'MMAD/', 'AgentIAD/']:
        idx = original_image.find(marker)
        if idx >= 0:
            rel_path = original_image[idx + len(marker):]
            break
    if rel_path is None:
        rel_path = Path(original_image).name

    actual_image_path = os.path.join(mmap_root, rel_path)
    if not os.path.exists(actual_image_path):
        return {'status': 'skip', 'reason': f'image not found: {actual_image_path}'}

    crops = parse_crop_from_messages(messages)

    try:
        image = PIL.Image.open(actual_image_path).convert('RGB')
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}

    new_images = [actual_image_path]
    stem = Path(rel_path).stem
    parent = str(Path(rel_path).parent)

    for i, crop in enumerate(crops):
        cropped = crop_image(image, crop)
        if cropped is None:
            new_images.append(actual_image_path)
            continue

        bbox_str = f"{crop[0]:.2f}_{crop[1]:.2f}_{crop[2]:.2f}_{crop[3]:.2f}"
        crop_filename = f"{i}_{bbox_str}.png"
        crop_save_dir = os.path.join(crop_dir, parent, stem)
        os.makedirs(crop_save_dir, exist_ok=True)
        crop_path = os.path.join(crop_save_dir, crop_filename)
        cropped.save(crop_path, "PNG")
        new_images.append(crop_path)

    record['images'] = new_images
    return {'status': 'ok', 'num_crops': len(crops)}


def main():
    ap = argparse.ArgumentParser(description='Prepare SFT data with crop images')
    ap.add_argument('--input', default='dataset/train-sft1600.json',
                    help='Input SFT JSON file')
    ap.add_argument('--mmap_root', required=True,
                    help='Root directory of downloaded MMAD dataset')
    ap.add_argument('--output', default=None,
                    help='Output JSON file (default: <input>-ready.json)')
    ap.add_argument('--crop_dir', default=None,
                    help='Directory to save crop images (default: <output_dir>/sft_crops)')
    ap.add_argument('--num_workers', type=int, default=8)
    args = ap.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        args.output = str(input_path.with_suffix('')) + '-ready.json'
    if args.crop_dir is None:
        args.crop_dir = str(Path(args.output).parent / 'sft_crops')

    print(f"Input: {args.input}")
    print(f"MMAD root: {args.mmap_root}")
    print(f"Output: {args.output}")
    print(f"Crop dir: {args.crop_dir}")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    tasks = [(record, args.mmap_root, args.crop_dir) for record in data]
    ok_count = 0
    skip_count = 0
    error_count = 0
    total_crops = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for i, result in enumerate(pool.map(process_record, tasks)):
            if result['status'] == 'ok':
                ok_count += 1
                total_crops += result['num_crops']
            elif result['status'] == 'skip':
                skip_count += 1
            else:
                error_count += 1
            if (i + 1) % 100 == 0 or i + 1 == len(data):
                print(f"\r  Processing: {i+1}/{len(data)} "
                      f"(ok={ok_count}, skip={skip_count}, err={error_count})",
                      end='', flush=True)

    print(f"\nDone: ok={ok_count}, skip={skip_count}, error={error_count}, "
          f"total_crops={total_crops}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
