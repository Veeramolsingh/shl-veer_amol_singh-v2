"""Shared helpers for the v2 pipeline.

v2 reuses everything v1 already computed (features/meta.csv, transcripts, WavLM, CoLA, GEC, text features)
and only adds new feature files in features/v2/. Outputs go to outputs/v2/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (FEAT_DIR, N_FOLDS, OUT_DIR, SEED, SR, audio_path, get_device, load_audio,  # noqa: E402,F401
                    load_meta, load_transcripts, split_sentences, strip_fillers, words_of)

# V2_FEAT_DIR / V2_OUT_DIR can point somewhere else for testing; defaults are features/v2/ and outputs/v2/
V2_FEAT = Path(os.environ.get("V2_FEAT_DIR", FEAT_DIR / "v2"))
V2_OUT = Path(os.environ.get("V2_OUT_DIR", OUT_DIR / "v2"))
V2_FEAT.mkdir(parents=True, exist_ok=True)
V2_OUT.mkdir(parents=True, exist_ok=True)

WHISPER_FILE = V2_FEAT / "whisper_encoder_layers.npy"
QWEN_FILE = V2_FEAT / "qwen_layers.npy"
ERRTYPES_FILE = V2_FEAT / "error_types.csv"
DEBERTA_FILE = V2_FEAT / "deberta_nz.csv"


def task_of(duration) -> np.ndarray:
    """Recording batch, read off the clip length: '45s' (44-46 s), '60s' (> 59 s) or 'other'.

    The 45 s clips are a separate batch: on the same prompts they get ~0.65 lower scores, and they make up
    56% of the test set but only 17% of train.
    """
    d = np.asarray(duration, dtype=float)
    return np.where((d >= 44) & (d <= 46), "45s", np.where(d > 59, "60s", "other"))


def task_flags(meta: pd.DataFrame) -> np.ndarray:
    """Two 0/1 columns (is_45s, is_other) appended to every base model's input."""
    t = task_of(meta["duration"])
    return np.column_stack([t == "45s", t == "other"]).astype(np.float32)


def test_mix_weights(meta: pd.DataFrame, train_rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-train-row weights that make the label>0 train rows look like the test set's batch mix.

    Used only for reporting the 'test-weighted' RMSE; label-0 rows get weight 0 (none are in the test set).
    """
    t_all = task_of(meta["duration"])
    t_tr = t_all[train_rows]
    test_mix = pd.Series(t_all[(meta["split"] == "test").to_numpy()]).value_counts(normalize=True)
    train_mix = pd.Series(t_tr[y > 0]).value_counts(normalize=True)
    w = pd.Series(t_tr).map(test_mix / train_mix).fillna(0).to_numpy()
    return np.where(y > 0, w, 0.0)
