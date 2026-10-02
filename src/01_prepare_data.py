"""Step 1 - Unpack the dataset, sanity-check it and build the metadata table + CV folds.

Output: features/meta.csv with one row per audio file (train + test).
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold

from common import DATA_DIR, META_CSV, N_FOLDS, ROOT, SEED

DEFAULT_ZIP = Path.home() / "Downloads" / "shl-hiring-assessment-2026.zip"
# Label bins used to stratify the folds: 0 | 1-2 | 2.5-3 | 3.5-4 | 4.5-5
LABEL_BIN_EDGES = [0.5, 2.25, 3.25, 4.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to the competition zip")
    args = ap.parse_args()

    if not (DATA_DIR / "train.csv").exists():
        print(f"Extracting {args.zip} -> data/ ...")
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(ROOT / "data")

    train = pd.read_csv(DATA_DIR / "train.csv").assign(split="train")
    test = pd.read_csv(DATA_DIR / "test.csv").assign(split="test", label=np.nan)  # test labels are dummies
    meta = pd.concat([train, test], ignore_index=True)
    # The train and test folders reuse file names (audio_12.wav exists in both), so key on split + name.
    meta["key"] = meta["split"] + "/" + meta["filename"]

    paths = [DATA_DIR / s / f for s, f in zip(meta["split"], meta["filename"])]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} audio files are missing, e.g. {missing[:3]}")
    meta["duration"] = [round(sf.info(str(p)).duration, 2) for p in paths]

    # Stratified K-fold on binned labels so every fold sees the rare label-0 clips.
    bins = np.digitize(train["label"], LABEL_BIN_EDGES)
    folds = np.full(len(meta), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (_, val_idx) in enumerate(skf.split(np.zeros(len(train)), bins)):
        folds[val_idx] = f  # train rows come first in `meta`, so indices line up
    meta["fold"] = folds

    meta = meta[["key", "split", "filename", "label", "duration", "fold"]]
    meta.to_csv(META_CSV, index=False)

    print(f"train: {len(train)} files | test: {len(test)} files")
    print(f"label mean {train['label'].mean():.3f}, std {train['label'].std():.3f}")
    print("label counts:", train["label"].value_counts().sort_index().to_dict())
    print(f"duration (s): min {meta['duration'].min():.1f}, median {meta['duration'].median():.1f}, "
          f"max {meta['duration'].max():.1f}")
    print(f"saved {META_CSV}")


if __name__ == "__main__":
    main()
