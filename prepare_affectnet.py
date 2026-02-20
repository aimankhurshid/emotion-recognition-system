"""
prepare_affectnet.py
====================
Organizes AffectNet+ dataset (flat images + JSON annotations) into the
class-folder structure expected by data_loader.py, using HARD LINKS
so NO extra disk space is used.

Expected AffectNet+ source layout:
    E:/AffectNet/AffectNet+/human_annotated/
        train_set/
            images/  (0.jpg, 1.jpg, ...)
            annotations/  (0.json, 1.json, ...)
        validation_set/
            images/
            annotations/

Output layout (inside project data/ folder):
    data/
        train/  -> 0_neutral/ 1_happy/ 2_sad/ ... 7_contempt/
        val/    -> same class folders
        test/   -> same class folders (50% split from validation_set)
"""

import os
import json
import random
import shutil
from pathlib import Path

# ─────────────────────────── CONFIGURE THESE ──────────────────────────────────
AFFECTNET_ROOT = r"E:\AffectNet\AffectNet+\human_annotated"
PROJECT_DATA   = r"C:\Users\Naman\emotion-recognition-system\data"
VAL_TEST_SPLIT = 0.5   # 50% of validation_set → test, 50% → val
RANDOM_SEED    = 42
USE_HARDLINKS  = True  # False = copy files (uses more space but works cross-drive)
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_LABELS = {
    0: "neutral",  1: "happy",   2: "sad",      3: "surprise",
    4: "fear",     5: "disgust", 6: "anger",    7: "contempt"
}


def make_class_dirs(base_dir):
    for eid, ename in EMOTION_LABELS.items():
        os.makedirs(os.path.join(base_dir, f"{eid}_{ename}"), exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    if USE_HARDLINKS:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # fallback to copy if hardlink fails (e.g. cross-drive)
    shutil.copy2(src, dst)


def process_split(images_dir: Path, annotations_dir: Path, output_dirs: dict):
    json_files = sorted(annotations_dir.glob("*.json"))
    print(f"  Found {len(json_files):,} annotation files in {annotations_dir.name}")

    keys = list(output_dirs.keys())
    if len(keys) == 2:
        random.seed(RANDOM_SEED)
        shuffled = json_files[:]
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * VAL_TEST_SPLIT)
        assignments = (
            [(f, output_dirs[keys[0]]) for f in shuffled[:split_idx]] +
            [(f, output_dirs[keys[1]]) for f in shuffled[split_idx:]]
        )
    else:
        assignments = [(f, output_dirs[keys[0]]) for f in json_files]

    ok = skipped = errors = 0
    for i, (json_path, out_dir) in enumerate(assignments):
        if i % 5000 == 0 and i > 0:
            print(f"    ... processed {i:,} / {len(assignments):,}")

        stem = json_path.stem
        img_path = images_dir / f"{stem}.jpg"

        if not img_path.exists():
            skipped += 1
            continue

        try:
            with open(json_path, "r") as f:
                ann = json.load(f)
            label = int(ann["human-label"])
        except Exception as e:
            errors += 1
            print(f"  [WARN] Could not read {json_path.name}: {e}")
            continue

        if label not in EMOTION_LABELS:
            skipped += 1
            continue

        class_folder = out_dir / f"{label}_{EMOTION_LABELS[label]}"
        dst_path = class_folder / img_path.name

        try:
            link_or_copy(img_path, dst_path)
            ok += 1
        except Exception as e:
            errors += 1
            print(f"  [WARN] Failed {img_path.name}: {e}")

    return ok, skipped, errors


def main():
    affectnet = Path(AFFECTNET_ROOT)
    data_root  = Path(PROJECT_DATA)

    print("=" * 60)
    print("  AffectNet+ Dataset Preparation Script")
    print("=" * 60)
    print(f"  Source : {affectnet}")
    print(f"  Output : {data_root}")
    print(f"  Mode   : {'Hard Links (no extra space)' if USE_HARDLINKS else 'File Copy'}")
    print()

    # ── TRAIN ────────────────────────────────────────────────────────
    train_images = affectnet / "train_set" / "images"
    train_annots = affectnet / "train_set" / "annotations"
    train_out    = data_root / "train"

    print("[1/2] Processing TRAIN split...")
    make_class_dirs(train_out)
    ok, sk, er = process_split(train_images, train_annots, {"train": train_out})
    print(f"  ✅ Train  — linked: {ok:,}  |  skipped: {sk:,}  |  errors: {er:,}\n")

    # ── VAL + TEST (from validation_set) ─────────────────────────────
    val_images = affectnet / "validation_set" / "images"
    val_annots = affectnet / "validation_set" / "annotations"
    val_out    = data_root / "val"
    test_out   = data_root / "test"

    print("[2/2] Processing VAL + TEST split (from validation_set)...")
    make_class_dirs(val_out)
    make_class_dirs(test_out)
    ok, sk, er = process_split(val_images, val_annots,
                               {"val": val_out, "test": test_out})
    print(f"  ✅ Val+Test — linked: {ok:,}  |  skipped: {sk:,}  |  errors: {er:,}\n")

    # ── SUMMARY ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  CLASS DISTRIBUTION SUMMARY")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        print(f"\n  [{split.upper()}]")
        total = 0
        for eid, ename in EMOTION_LABELS.items():
            class_dir = split_dir / f"{eid}_{ename}"
            count = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
            total += count
            print(f"    {eid}_{ename:<12}: {count:>7,}")
        print(f"    {'TOTAL':<16}: {total:>7,}")

    print()
    print("✅ Done! Run training with:")
    print("  python training/train.py --data_dir data --epochs 30 --batch_size 16 \\")
    print("    --model_type full --backbone efficientnet_b4 --lstm_hidden 512 \\")
    print("    --use_class_weights --checkpoint_dir results/phase1_laptop_benchmark")


if __name__ == "__main__":
    main()
