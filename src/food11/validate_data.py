"""Check a processed Food-11 dataset folder against the data contract.

    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 100

Prints one PASS/FAIL line per check and exits with status 1 if any check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

from food11.data import CLASSES, TARGET_SIZE

SPLITS = ("training", "validation", "evaluation")
IMAGE_SIZE = (TARGET_SIZE, TARGET_SIZE)
MAX_DETAILS = 5


def images_by_split(root: Path) -> dict[str, dict[str, list[Path]]]:
    """{split: {class folder name: [image files]}} for whatever is actually on disk."""
    found: dict[str, dict[str, list[Path]]] = {}
    for split in SPLITS:
        split_dir = root / split
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []
        found[split] = {d.name: sorted(p for p in d.iterdir() if p.is_file()) for d in class_dirs}
    return found


def check_class_folders(data: dict[str, dict[str, list[Path]]]) -> list[str]:
    errors = []
    for split, classes in data.items():
        missing = sorted(set(CLASSES) - set(classes))
        extra = sorted(set(classes) - set(CLASSES))
        if missing:
            errors.append(f"{split}: missing {missing}")
        if extra:
            errors.append(f"{split}: unexpected {extra}")
    return errors


def check_images(data: dict[str, dict[str, list[Path]]], root: Path) -> list[str]:
    errors = []
    for classes in data.values():
        for paths in classes.values():
            for path in paths:
                name = path.relative_to(root).as_posix()
                try:
                    with Image.open(path) as img:
                        img.load()  # force a full decode, so truncated files are caught too
                        if img.mode != "RGB" or img.size != IMAGE_SIZE:
                            errors.append(f"{name}: {img.mode} {img.size}, expected RGB {IMAGE_SIZE}")
                except (OSError, ValueError, SyntaxError) as exc:  # what PIL raises on unreadable or truncated files
                    errors.append(f"{name}: cannot be opened ({exc})")
    return errors


def check_min_per_class(data: dict[str, dict[str, list[Path]]], min_per_class: int) -> list[str]:
    errors = []
    for split, classes in data.items():
        for name in CLASSES:
            count = len(classes.get(name, []))
            if count < min_per_class:
                errors.append(f"{split}/{name}: {count} images, expected at least {min_per_class}")
    return errors


def check_no_cross_split_duplicates(data: dict[str, dict[str, list[Path]]], root: Path) -> list[str]:
    by_hash: dict[str, list[tuple[str, Path]]] = {}
    for split, classes in data.items():
        for paths in classes.values():
            for path in paths:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                by_hash.setdefault(digest, []).append((split, path))

    errors = []
    for entries in by_hash.values():
        if len({split for split, _ in entries}) > 1:
            names = ", ".join(path.relative_to(root).as_posix() for _, path in entries)
            errors.append(f"same content in several splits: {names}")
    return errors


def check_class_shares(data: dict[str, dict[str, list[Path]]], tau: float) -> list[str]:
    def shares(split: str) -> dict[str, float]:
        counts = {name: len(data[split].get(name, [])) for name in CLASSES}
        total = sum(counts.values())
        return {name: (count / total if total else 0.0) for name, count in counts.items()}

    train, val = shares("training"), shares("validation")
    errors = []
    for name in CLASSES:
        gap = abs(train[name] - val[name])
        if gap > tau:
            errors.append(f"{name}: training {train[name]:.3f} vs validation {val[name]:.3f} (gap {gap:.3f} > {tau})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset against the data contract")
    parser.add_argument("root", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"FAIL  dataset folder not found: {args.root}")
        sys.exit(1)

    data = images_by_split(args.root)
    checks = [
        ("1. each split has exactly the 11 class folders", check_class_folders(data)),
        ("2. every image opens, is RGB and 128x128", check_images(data, args.root)),
        (f"3. at least {args.min_per_class} images per class", check_min_per_class(data, args.min_per_class)),
        ("4. no image appears in two splits", check_no_cross_split_duplicates(data, args.root)),
        (f"5. train/validation class shares within {args.tau}", check_class_shares(data, args.tau)),
    ]

    total = sum(len(paths) for classes in data.values() for paths in classes.values())
    print(f"Data contract for {args.root.as_posix()} ({total} images)")
    for name, errors in checks:
        print(f"{'FAIL' if errors else 'PASS'}  {name}")
        for error in errors[:MAX_DETAILS]:
            print(f"        - {error}")
        if len(errors) > MAX_DETAILS:
            print(f"        ... and {len(errors) - MAX_DETAILS} more")

    sys.exit(1 if any(errors for _, errors in checks) else 0)


if __name__ == "__main__":
    main()
