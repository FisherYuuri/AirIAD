#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compress MMAD images: resize so that max side ≤ 1024 pixels (keep aspect ratio).
Multi-threaded with progress display.
"""

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import PIL.Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def resize_image_by_pixels(image: PIL.Image.Image,
                           max_pixels: int = 1048576,
                           min_pixels: int = 1024) -> PIL.Image.Image:
    """Resize image by pixel count."""
    w, h = image.size
    pixels = w * h

    if pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / pixels)
        w, h = int(w * resize_factor), int(h * resize_factor)
        image = image.resize((w, h))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        w, h = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((w, h))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_single_image(args: tuple) -> Dict:
    """Process a single image (for multi-threading)."""
    img_path, input_root, output_root, is_mask = args

    if input_root in img_path:
        rel_path = img_path.replace(input_root, "").lstrip("/")
        output_path = os.path.join(output_root, rel_path)
    else:
        return {'original_path': img_path, 'status': 'skipped'}

    if os.path.exists(output_path):
        return {'original_path': img_path, 'status': 'exists', 'output_path': output_path}

    if not os.path.exists(img_path):
        return {'original_path': img_path, 'status': 'not_found'}

    try:
        img = PIL.Image.open(img_path)
        orig_w, orig_h = img.size
        orig_pixels = orig_w * orig_h
        orig_mode = img.mode
        orig_size_kb = os.path.getsize(img_path) / 1024

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if is_mask:
            pixels = orig_w * orig_h
            max_pixels = 1024 * 1024

            if pixels > max_pixels:
                resize_factor = math.sqrt(max_pixels / pixels)
                new_w, new_h = int(orig_w * resize_factor), int(orig_h * resize_factor)
                img = img.resize((new_w, new_h), PIL.Image.NEAREST)

            comp_w, comp_h = img.size
            comp_pixels = comp_w * comp_h
            img.save(output_path, "PNG")
        else:
            compressed = resize_image_by_pixels(img, max_pixels=1024*1024, min_pixels=1024)
            comp_w, comp_h = compressed.size
            comp_pixels = comp_w * comp_h

            if img_path.lower().endswith('.png'):
                compressed.save(output_path, "PNG")
            else:
                compressed.save(output_path, "JPEG", quality=95)

        comp_size_kb = os.path.getsize(output_path) / 1024

        return {
            'original_path': img_path,
            'output_path': output_path,
            'status': 'success',
            'is_mask': is_mask,
            'original_size': f"{orig_w}x{orig_h}",
            'original_pixels': orig_pixels,
            'original_mode': orig_mode,
            'original_file_kb': round(orig_size_kb, 2),
            'compressed_size': f"{comp_w}x{comp_h}",
            'compressed_pixels': comp_pixels,
            'compressed_file_kb': round(comp_size_kb, 2),
            'pixel_ratio': round(comp_pixels / orig_pixels, 4) if orig_pixels > 0 else 1,
            'file_ratio': round(comp_size_kb / orig_size_kb, 4) if orig_size_kb > 0 else 1,
        }

    except Exception as e:
        return {
            'original_path': img_path,
            'status': 'error',
            'is_mask': is_mask,
            'error': str(e),
        }


class SimpleProgress:
    """Simple progress bar (no tqdm dependency)."""
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.count = 0
        self.lock = threading.Lock()

    def update(self, n: int = 1):
        with self.lock:
            self.count += n
            pct = self.count / self.total * 100
            bar_len = 40
            filled = int(bar_len * self.count / self.total)
            bar = '=' * filled + '-' * (bar_len - filled)
            print(f"\r{self.desc}: |{bar}| {self.count}/{self.total} ({pct:.1f}%)", end='', flush=True)

    def close(self):
        print()


def main():
    input_root = "<MMAD_ROOT>"
    output_root = "<MMAD_ROOT>_resize_1024"
    report_path = "<MMAD_ROOT>/compression_report.txt"

    json_paths = [
        "<DATASET_DIR>/mmad-train-sft-os.json",
        "<DATASET_DIR>/mmad-train-sft-zs.json",
        "<DATASET_DIR>/mix-train-grpo.json",
    ]

    num_workers = 16

    all_images = set()
    all_masks = set()

    for json_path in json_paths:
        print(f"Reading dataset: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            mask_path = item.get('mask_path', '')
            if mask_path:
                all_masks.add(mask_path)

            # SFT format: images array
            images = item.get('images') or []
            for img_path in images:
                if img_path and img_path not in all_masks:
                    all_images.add(img_path)

            # GRPO format: single image field
            img_path = item.get('image', '')
            if img_path and img_path not in all_masks:
                all_images.add(img_path)

            similar_templates = item.get('similar_templates') or []
            for template in similar_templates:
                if template and template not in all_masks:
                    all_images.add(template)

            crop_images = item.get('crop_image') or []
            for crop_img in crop_images:
                if crop_img and crop_img not in all_masks:
                    all_images.add(crop_img)

    total_images = len(all_images)
    total_masks = len(all_masks)
    print(f"Found {total_images} unique images, {total_masks} masks")
    print(f"Using {num_workers} workers...")

    tasks = [
        (img_path, input_root, output_root, False)
        for img_path in sorted(all_images)
    ] + [
        (mask_path, input_root, output_root, True)
        for mask_path in sorted(all_masks)
    ]

    total_tasks = len(tasks)

    stats: List[Dict] = []
    success_count = 0
    exists_count = 0
    not_found_count = 0
    error_count = 0

    if HAS_TQDM:
        progress = tqdm(total=total_tasks, desc="Compressing", unit="img")
    else:
        progress = SimpleProgress(total_tasks, desc="Compressing")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            stats.append(result)

            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'exists':
                exists_count += 1
            elif result['status'] == 'not_found':
                not_found_count += 1
            elif result['status'] == 'error':
                error_count += 1

            progress.update(1)

    progress.close()

    # Generate report
    print(f"\nGenerating report: {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Image Compression Report (max side ≤ 1024, keep aspect ratio)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total: {total_tasks} (images: {total_images}, masks: {total_masks})\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Skipped (exists): {exists_count}\n")
        f.write(f"Not found: {not_found_count}\n")
        f.write(f"Errors: {error_count}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Original Size Distribution:\n")
        f.write("-" * 80 + "\n")
        orig_sizes = [s for s in stats if s['status'] == 'success']
        if orig_sizes:
            widths = [int(s['original_size'].split('x')[0]) for s in orig_sizes]
            heights = [int(s['original_size'].split('x')[1]) for s in orig_sizes]
            pixels = [s['original_pixels'] for s in orig_sizes]
            file_sizes = [s['original_file_kb'] for s in orig_sizes]

            f.write(f"Width: {min(widths)} ~ {max(widths)}, avg: {sum(widths)/len(widths):.1f}\n")
            f.write(f"Height: {min(heights)} ~ {max(heights)}, avg: {sum(heights)/len(heights):.1f}\n")
            f.write(f"Pixels: {min(pixels):,} ~ {max(pixels):,}, avg: {sum(pixels)/len(pixels):,.1f}\n")
            f.write(f"File size: {min(file_sizes):.2f}KB ~ {max(file_sizes):.2f}KB, avg: {sum(file_sizes)/len(file_sizes):.2f}KB\n")
            f.write(f"Images >1024x1024: {sum(1 for p in pixels if p > 1024*1024)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Compressed Size Distribution:\n")
        f.write("-" * 80 + "\n")
        if orig_sizes:
            widths = [int(s['compressed_size'].split('x')[0]) for s in orig_sizes]
            heights = [int(s['compressed_size'].split('x')[1]) for s in orig_sizes]
            pixels = [s['compressed_pixels'] for s in orig_sizes]
            file_sizes = [s['compressed_file_kb'] for s in orig_sizes]

            f.write(f"Width: {min(widths)} ~ {max(widths)}, avg: {sum(widths)/len(widths):.1f}\n")
            f.write(f"Height: {min(heights)} ~ {max(heights)}, avg: {sum(heights)/len(heights):.1f}\n")
            f.write(f"Pixels: {min(pixels):,} ~ {max(pixels):,}, avg: {sum(pixels)/len(pixels):,.1f}\n")
            f.write(f"File size: {min(file_sizes):.2f}KB ~ {max(file_sizes):.2f}KB, avg: {sum(file_sizes)/len(file_sizes):.2f}KB\n\n")

        f.write("-" * 80 + "\n")
        f.write("Compression Ratio:\n")
        f.write("-" * 80 + "\n")
        if orig_sizes:
            pixel_ratios = [s['pixel_ratio'] for s in orig_sizes]
            file_ratios = [s['file_ratio'] for s in orig_sizes]
            f.write(f"Pixel ratio: {min(pixel_ratios):.4f} ~ {max(pixel_ratios):.4f}, avg: {sum(pixel_ratios)/len(pixel_ratios):.4f}\n")
            f.write(f"File ratio: {min(file_ratios):.4f} ~ {max(file_ratios):.4f}, avg: {sum(file_ratios)/len(file_ratios):.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Compressed Images (original > 1024x1024):\n")
        f.write("-" * 80 + "\n")
        compressed_images = [s for s in orig_sizes if s['pixel_ratio'] < 1.0]
        for s in compressed_images[:100]:
            f.write(f"{s['original_size']} -> {s['compressed_size']} "
                   f"(pixel: {s['pixel_ratio']:.4f}, file: {s['file_ratio']:.4f})\n"
                   f"  {s['original_path']}\n")
        if len(compressed_images) > 100:
            f.write(f"\n... {len(compressed_images) - 100} more compressed images\n")

        if error_count > 0:
            f.write("\n" + "-" * 80 + "\n")
            f.write("Errors:\n")
            f.write("-" * 80 + "\n")
            for s in stats:
                if s['status'] == 'error':
                    f.write(f"ERROR: {s['original_path']} - {s.get('error', 'unknown')}\n")

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"Images: {total_images}, Masks: {total_masks}")
    print(f"Success: {success_count}, Skipped: {exists_count}, Not found: {not_found_count}, Errors: {error_count}")
    print(f"Output: {output_root}")
    print(f"Report: {report_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
