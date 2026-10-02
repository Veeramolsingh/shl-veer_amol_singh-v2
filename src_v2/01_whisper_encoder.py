"""v2 Step 1 - Whisper-large-v3-turbo encoder embeddings.

A second, differently-trained audio view next to WavLM. Every encoder layer is mean-pooled over the frames that
contain audio (Whisper pads each 30 s window; padding frames are ignored). Clips longer than 30 s are processed
in 30 s windows and pooled together. Step 5 uses the average of layers 27-29 (best by CV).

Output: features/v2/whisper_encoder_layers.npy  (n_files, 33, 1280) float16, rows in meta.csv order
Runtime: ~8 min on a 4 GB GPU (first run also downloads ~1.6 GB).
"""
import argparse
import time

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from common_v2 import SR, WHISPER_FILE, audio_path, load_audio, load_meta

MODEL = "openai/whisper-large-v3-turbo"
WIN = 30 * SR
FRAMES_PER_S = 50  # Whisper encoder output rate (1500 frames per 30 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N clips (nothing is saved)")
    args = ap.parse_args()

    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    enc = WhisperModel.from_pretrained(MODEL, dtype=torch.float16).get_encoder().cuda().eval()
    meta = load_meta()
    if args.limit:
        meta = meta.head(args.limit)
    means = np.zeros((len(meta), enc.config.encoder_layers + 1, enc.config.d_model), np.float16)

    t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(meta.itertuples()):
            wav = load_audio(audio_path(row.split, row.filename))
            total, n = None, 0
            for start in range(0, len(wav), WIN):
                chunk = wav[start:start + WIN]
                if start > 0 and len(chunk) < SR // 2:  # ignore a sub-0.5 s tail
                    continue
                feats = fe(chunk, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
                hidden = enc(feats, output_hidden_states=True).hidden_states  # layers x (1, 1500, dim)
                valid = min(1500, int(np.ceil(len(chunk) / SR * FRAMES_PER_S)))
                h = torch.stack(hidden, 0)[:, 0, :valid].float()  # (layers, valid, dim)
                total = h.sum(1) if total is None else total + h.sum(1)
                n += valid
            means[i] = (total / n).cpu().numpy()
            if i % 100 == 0:
                print(f"{i}/{len(meta)}  {time.time() - t0:.0f}s", flush=True)

    if args.limit:
        print(f"smoke test ok: {means.shape}, finite={np.isfinite(means.astype(np.float32)).all()}")
        return
    np.save(WHISPER_FILE, means)
    print(f"saved {WHISPER_FILE} {means.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
