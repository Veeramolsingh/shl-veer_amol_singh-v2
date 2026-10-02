"""Shared paths, constants and small helpers used by every pipeline step."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Dataset_Final"
FEAT_DIR = ROOT / "features"
OUT_DIR = ROOT / "outputs"
FEAT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

META_CSV = FEAT_DIR / "meta.csv"
TRANSCRIPTS = FEAT_DIR / "transcripts.jsonl"

# ----------------------------------------------------------------------------- constants
SR = 16_000
SEED = 42
N_FOLDS = 5


# ----------------------------------------------------------------------------- data access
def load_meta() -> pd.DataFrame:
    """One row per audio file: key, split, filename, label (NaN for test), duration, fold (-1 for test).

    Train and test folders reuse the same file names, so `key` = "<split>/<filename>" is the
    unique id used by every feature table.
    """
    if not META_CSV.exists():
        raise SystemExit("features/meta.csv not found - run src/01_prepare_data.py first.")
    return pd.read_csv(META_CSV)


def audio_path(split: str, filename: str) -> Path:
    return DATA_DIR / split / filename


def load_audio(path) -> np.ndarray:
    """Mono float32 waveform at 16 kHz."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_transcripts(missing_ok: bool = False) -> dict[str, dict]:
    """Whisper output of step 2, keyed by `key`."""
    if not TRANSCRIPTS.exists():
        if missing_ok:
            return {}
        raise SystemExit("features/transcripts.jsonl not found - run src/02_transcribe.py first.")
    records = {}
    with open(TRANSCRIPTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["key"]] = rec
    return records


# ----------------------------------------------------------------------------- hardware
def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expose_cuda_dlls() -> None:
    """Windows only: let CTranslate2 (faster-whisper) find the cuBLAS / cuDNN DLLs bundled with PyTorch."""
    if os.name != "nt":
        return
    import site

    import torch  # noqa: F401  (importing torch also loads its CUDA DLLs into the process)

    candidates = [Path(torch.__file__).parent / "lib"]
    for sp in site.getsitepackages():
        candidates += [Path(sp) / "nvidia" / pkg / "bin" for pkg in ("cublas", "cudnn", "cuda_runtime")]
    for d in candidates:
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


# ----------------------------------------------------------------------------- text helpers
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mmm", "mhm", "eh"}
_FILLER_RE = re.compile(r",?\s*\b(?:" + "|".join(sorted(FILLERS, key=len, reverse=True)) + r")\b[,.!?]*", re.I)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words_of(text: str) -> list[str]:
    """Lower-cased word tokens without punctuation (keeps contractions like don't)."""
    return _WORD_RE.findall(text.lower())


def strip_fillers(text: str) -> str:
    """Remove um/uh-style fillers: they are a fluency signal, not a grammar error."""
    text = _FILLER_RE.sub(" ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"(^|[.!?]\s+)[,\s]+", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def split_sentences(text: str, max_words: int = 40) -> list[str]:
    """Split on Whisper's sentence punctuation; very long run-ons are cut into max_words chunks."""
    out = []
    for part in re.split(r"(?<=[.!?])\s+", text.strip()):
        tokens = part.split()
        for i in range(0, len(tokens), max_words):
            out.append(" ".join(tokens[i : i + max_words]))
    return [s for s in out if s]


def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two word sequences (insertions, deletions, substitutions)."""
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]
