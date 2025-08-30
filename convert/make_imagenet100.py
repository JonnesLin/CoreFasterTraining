#!/usr/bin/env python3
"""
Create an ImageNet-100 subset from an ImageNet-1k ImageFolder.

Assumptions
- Source is in standard ImageFolder layout:
  <src>/train/<class>/*, and <src>/(val|validation)/<class>/* (if validation split present).
- Class folder names are used as labels (e.g., WordNet wnids like 'n01440764').

Features
- Select K classes randomly or from a provided list (one class per line).
- Link/copy files to destination (symlink, hardlink, or copy).
- Keeps existing files (idempotent) and skips missing classes with a warning.

Examples
  # Random 100 classes using symlinks
  python convert/make_imagenet100.py \
    --src /path/imagenet-1k --dst /path/imagenet-100

  # Use class list file (one class per line)
  python convert/make_imagenet100.py \
    --src /path/imagenet-1k --dst /path/imagenet-100 \
    --classes classes_100.txt --mode symlink

  # Distributed-friendly hardlinks
  python convert/make_imagenet100.py --src <src> --dst <dst> --mode hardlink

Notes
- If validation split is not found or not organized by class directories, the script will only
  process the training split and warn about validation.
- To restrict per-class image counts, you can add --max-per-class <N>.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp', '.JPEG', '.JPG'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Make ImageNet-100 subset from ImageNet-1k ImageFolder')
    p.add_argument('--src', required=True, type=str, help='Source ImageNet root (contains train/ and val|validation/)')
    p.add_argument('--dst', required=True, type=str, help='Destination root to create ImageNet-100 subset')
    p.add_argument('--k', type=int, default=100, help='Number of classes to keep (default: 100)')
    p.add_argument('--classes', type=str, default=None,
                   help='Path to text file listing class folder names (one per line). If not set, pick randomly.')
    p.add_argument('--seed', type=int, default=0, help='Random seed when sampling classes')
    p.add_argument('--mode', type=str, default='symlink', choices=['symlink', 'hardlink', 'copy'],
                   help='How to populate destination files (default: symlink)')
    p.add_argument('--max-per-class', type=int, default=None,
                   help='Limit number of images per class for each split (default: keep all)')
    p.add_argument('--dry-run', action='store_true', help='Print planned actions without touching filesystem')
    return p.parse_args()


def _exists_dir(p: Path) -> bool:
    return p.exists() and p.is_dir()


def resolve_split_root(src: Path, name: str) -> Optional[Path]:
    """Return split path for 'train' or 'val'/'validation'."""
    if name == 'train':
        cand = src / 'train'
        return cand if _exists_dir(cand) else None
    # val / validation
    for n in ('val', 'validation'):
        cand = src / n
        if _exists_dir(cand):
            return cand
    return None


def list_class_dirs(split_root: Path) -> List[str]:
    if not _exists_dir(split_root):
        return []
    cls = [d.name for d in sorted(split_root.iterdir()) if d.is_dir() and not d.name.startswith('.')]
    return cls


def is_split_structured(split_root: Path) -> bool:
    """Return True if split is organized as <split>/<class>/*."""
    if not _exists_dir(split_root):
        return False
    # Consider it structured if there is at least one subdirectory
    return any(p.is_dir() for p in split_root.iterdir())


def read_class_list(path: Path) -> List[str]:
    classes = []
    with path.open('r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                continue
            classes.append(s)
    return classes


def iter_images(cls_dir: Path, max_count: Optional[int] = None) -> Iterable[Path]:
    count = 0
    for p in sorted(cls_dir.iterdir()):
        if p.is_file() and p.suffix in IMG_EXTS:
            yield p
            count += 1
            if max_count is not None and count >= max_count:
                break


def ensure_dir(p: Path, dry_run: bool = False) -> None:
    if dry_run:
        return
    p.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, dry_run: bool = False) -> None:
    if dry_run:
        return
    if dst.exists():
        return
    if mode == 'symlink':
        # Use relative symlink for portability (no fallback)
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def populate_split(src_root: Path, dst_root: Path, classes: Sequence[str], mode: str,
                   max_per_class: Optional[int], dry_run: bool) -> Tuple[int, int]:
    if src_root is None or not _exists_dir(src_root):
        return 0, 0
    if not is_split_structured(src_root):
        print(f'[WARN] Split not organized by class directories: {src_root}. Skipping this split.')
        return 0, 0

    n_files = 0
    n_classes_present = 0
    for c in classes:
        src_cls = src_root / c
        if not _exists_dir(src_cls):
            print(f'[WARN] Class not found in split: {c} (root={src_root})')
            continue
        n_classes_present += 1
        dst_cls = dst_root / c
        ensure_dir(dst_cls, dry_run=dry_run)
        for img in iter_images(src_cls, max_count=max_per_class):
            dst_img = dst_cls / img.name
            place_file(img, dst_img, mode=mode, dry_run=dry_run)
            n_files += 1
    return n_classes_present, n_files


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    src_train = resolve_split_root(src, 'train')
    src_val = resolve_split_root(src, 'val')
    if src_train is None:
        print(f'[ERROR] Could not find train/ under: {src}')
        sys.exit(1)

    all_classes = list_class_dirs(src_train)
    if not all_classes:
        print(f'[ERROR] No class directories found under train/: {src_train}')
        sys.exit(1)

    if args.classes:
        wanted = read_class_list(Path(args.classes))
        classes = [c for c in wanted if c in all_classes]
        missing = [c for c in wanted if c not in all_classes]
        if missing:
            print(f'[WARN] {len(missing)} classes not found in train/: {missing[:10]}{' , ' if len(missing) > 10 else ''}')
    else:
        if args.k <= 0 or args.k > len(all_classes):
            print(f'[ERROR] --k must be within (0, {len(all_classes)}]')
            sys.exit(1)
        random.seed(args.seed)
        classes = sorted(random.sample(all_classes, args.k))

    # Create destination roots
    dst_train = dst / 'train'
    dst_val = dst / 'val'
    if args.dry_run:
        print(f'[DRY-RUN] Will create ImageNet-{len(classes)} at: {dst}')
        print(f'[DRY-RUN] Mode={args.mode}, max_per_class={args.max_per_class}')
        print(f'[DRY-RUN] Selected classes ({len(classes)}): {classes[:10]}{' , ' if len(classes) > 10 else ''}')
    else:
        ensure_dir(dst_train)
        # create val root only if we will populate it
        if src_val is not None and is_split_structured(src_val):
            ensure_dir(dst_val)

    # Populate splits
    n_cls_tr, n_files_tr = populate_split(src_train, dst_train, classes, args.mode, args.max_per_class, args.dry_run)
    n_cls_va, n_files_va = (0, 0)
    if src_val is not None:
        if is_split_structured(src_val):
            n_cls_va, n_files_va = populate_split(src_val, dst_val, classes, args.mode, args.max_per_class, args.dry_run)
        else:
            print(f'[WARN] Validation split is not class-structured: {src_val}. Skipping validation.')
    else:
        print('[INFO] No validation split found. Only train/ will be created.')

    print('[DONE] Summary:')
    print(f'  Train: classes={n_cls_tr}/{len(classes)}, files={n_files_tr}, src={src_train}')
    if src_val is not None:
        print(f'  Val  : classes={n_cls_va}/{len(classes)}, files={n_files_va}, src={src_val}')
    print(f'  Output: {dst}')


if __name__ == '__main__':
    main()
