"""Step 3 - Acoustic features straight from the waveform.

(a) WavLM-Large embeddings. All 25 hidden layers are mean-pooled over time and stored, so
    step 6 can pick the layer that best predicts the score (middle layers carry the most
    phonetic / linguistic information). Speaking proficiency - fluency, rhythm, accent -
    is strongly correlated with grammar proficiency, and WavLM captures it without a transcript.
(b) Hand-crafted prosody: loudness-based pause statistics and pitch variation (librosa).

Outputs: features/wavlm_layers.npy  (n_files, 25, 1024) float16, rows in meta.csv order
         features/acoustic.csv
"""
import contextlib

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WavLMModel

from common import FEAT_DIR, SR, audio_path, get_device, load_audio, load_meta

MODEL = "microsoft/wavlm-large"
CHUNK_S = 30          # WavLM is run on <=30 s chunks to bound GPU memory
HOP = 160             # 10 ms analysis hop for prosody
PAUSE_MIN_FRAMES = 30   # a silence >= 0.3 s counts as a pause
LONG_PAUSE_FRAMES = 100  # >= 1 s is a long pause


@torch.inference_mode()
def wavlm_layer_means(model, wav, device, use_amp=True):
    """Time-averaged hidden state of every WavLM layer -> (n_layers + 1, hidden)."""
    n_chunks = max(1, int(np.ceil(len(wav) / (CHUNK_S * SR))))
    sums, frames = None, 0
    for chunk in np.array_split(wav, n_chunks):
        x = (chunk - chunk.mean()) / (chunk.std() + 1e-7)  # WavLM-Large was trained on normalised audio
        x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(device)
        amp = (torch.autocast("cuda", dtype=torch.float16) if use_amp and device.type == "cuda"
               else contextlib.nullcontext())
        with amp:
            hidden = model(x, output_hidden_states=True).hidden_states
        h = torch.stack(hidden, 0)[:, 0].float()  # (layers, T, hidden)
        sums = h.sum(1) if sums is None else sums + h.sum(1)
        frames += h.shape[1]
    out = (sums / frames).cpu().numpy()
    if use_amp and not np.isfinite(out).all():  # fp16 overflow -> redo this clip in fp32
        return wavlm_layer_means(model, wav, device, use_amp=False)
    return out


def run_lengths(mask):
    """Lengths of consecutive True runs in a boolean array."""
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def acoustic_features(wav):
    rms = librosa.feature.rms(y=wav, frame_length=400, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    active = db > np.percentile(db, 95) - 35  # frames within 35 dB of the loud part = speech
    if active.sum() < 10:
        return {"ac_active_ratio": float(active.mean())}

    first, last = np.flatnonzero(active)[[0, -1]]
    inner_silence = ~active[first : last + 1]  # ignore leading / trailing silence
    pauses = run_lengths(inner_silence)
    pauses = pauses[pauses >= PAUSE_MIN_FRAMES]
    span_min = (last - first + 1) * HOP / SR / 60

    f0 = librosa.yin(wav, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(len(f0), len(active))
    f0 = f0[:n][active[:n]]
    semitones = 12 * np.log2(f0 / np.median(f0))

    return {
        "ac_active_ratio": float(active.mean()),
        "ac_pauses_per_min": len(pauses) / span_min,
        "ac_long_pauses_per_min": float((pauses >= LONG_PAUSE_FRAMES).sum()) / span_min,
        "ac_mean_pause_s": float(pauses.mean() * HOP / SR) if len(pauses) else 0.0,
        "ac_loudness_mean": float(db[active].mean()),
        "ac_loudness_std": float(db[active].std()),
        "ac_f0_median": float(np.median(f0)),
        "ac_pitch_std_st": float(semitones.std()),
        "ac_pitch_range_st": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
    }


def main():
    meta = load_meta()
    device = get_device()
    print(f"Loading {MODEL} on {device} ...")
    model = WavLMModel.from_pretrained(MODEL).to(device).eval()

    n_layers = model.config.num_hidden_layers + 1
    emb = np.zeros((len(meta), n_layers, model.config.hidden_size), dtype=np.float16)
    rows = []
    for i, row in enumerate(tqdm(meta.itertuples(), total=len(meta), desc="audio features")):
        wav = load_audio(audio_path(row.split, row.filename))
        emb[i] = wavlm_layer_means(model, wav, device)
        rows.append({"key": row.key, **acoustic_features(wav)})

    np.save(FEAT_DIR / "wavlm_layers.npy", emb)
    pd.DataFrame(rows).to_csv(FEAT_DIR / "acoustic.csv", index=False)
    print(f"saved wavlm_layers.npy {emb.shape} and acoustic.csv")


if __name__ == "__main__":
    main()
