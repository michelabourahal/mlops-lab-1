from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "food11_raw"
PROCESSED_DIR = ROOT / "data" / "food11_processed"
PROCESSED_MINI_DIR = ROOT / "data" / "food11_processed_mini"
SPLITS = ("training", "evaluation", "validation")
TARGET_SIZE = (128, 128)
MAX_IMAGES_PER_CLASS = 100


def _iter_image_files(path: Path):
    return sorted(
        [entry for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )


def _class_name_from_path(path: Path) -> str:
    if path.stem and "_" in path.stem:
        return path.stem.split("_", 1)[0]
    return path.stem


def _copy_images(source_root: Path, destination_root: Path, max_per_class: int | None = None) -> dict[str, int]:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split in SPLITS:
        split_source = source_root / split
        if not split_source.exists():
            continue

        split_target = destination_root / split
        split_target.mkdir(parents=True, exist_ok=True)

        class_to_images: dict[str, list[Path]] = {}

        for entry in sorted(split_source.iterdir()):
            if entry.is_dir():
                for image_path in _iter_image_files(entry):
                    class_name = entry.name
                    class_to_images.setdefault(class_name, []).append(image_path)
            elif entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                class_name = _class_name_from_path(entry)
                class_to_images.setdefault(class_name, []).append(entry)

        for class_name in sorted(class_to_images):
            class_target = split_target / class_name
            class_target.mkdir(parents=True, exist_ok=True)

            image_files = class_to_images[class_name]
            selected_files = image_files[:max_per_class] if max_per_class is not None else image_files

            for image_path in selected_files:
                with Image.open(image_path) as image:
                    resized = image.convert("RGB").resize(TARGET_SIZE)
                    resized.save(class_target / image_path.name, format="JPEG")

            counts[f"{split}/{class_name}"] = len(selected_files)

    return counts


def main() -> None:
    print(f"Processing raw dataset from: {RAW_DATA_DIR}")

    full_counts = _copy_images(RAW_DATA_DIR, PROCESSED_DIR)
    mini_counts = _copy_images(RAW_DATA_DIR, PROCESSED_MINI_DIR, max_per_class=MAX_IMAGES_PER_CLASS)

    total_full = sum(full_counts.values())
    total_mini = sum(mini_counts.values())

    print(f"Created {PROCESSED_DIR} with {total_full} images")
    print(f"Created {PROCESSED_MINI_DIR} with {total_mini} images")

    print("Sample per split/class counts:")
    for key in sorted(full_counts):
        print(f"  {key}: {full_counts[key]} -> {mini_counts.get(key, 0)} (mini)")


if __name__ == "__main__":
    main()
