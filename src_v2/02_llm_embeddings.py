"""v2 Step 2 - Transcript embeddings from a 1.5B-parameter language model (Qwen2.5-1.5B).

Each transcript (fillers removed) is run through the LM and every layer's hidden states are mean-pooled.
A ridge regression on the middle layers (17-19) is the best text model on prompts that were not seen in
training - better than the 0.5B model, so model size helps here.

Output: features/v2/qwen_layers.npy  (n_files, 29, 1536) float16, rows in meta.csv order
Runtime: ~2-3 min on a 4 GB GPU (first run also downloads ~3 GB).
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common_v2 import QWEN_FILE, load_meta, load_transcripts, strip_fillers

MODEL = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N clips (nothing is saved)")
    args = ap.parse_args()

    meta = load_meta()
    if args.limit:
        meta = meta.head(args.limit)
    transcripts = load_transcripts()
    texts = [strip_fillers(transcripts[k]["text"]) for k in meta["key"]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    out = np.zeros((len(texts), model.config.num_hidden_layers + 1, model.config.hidden_size), np.float16)

    t0 = time.time()
    with torch.inference_mode():
        for i, t in enumerate(texts):
            ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.cuda()
            hidden = model(ids, output_hidden_states=True).hidden_states  # layers x (1, T, H)
            out[i] = torch.stack(hidden, 0)[:, 0].float().mean(1).cpu().numpy()
            if i % 200 == 0:
                print(f"{i}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)

    if args.limit:
        print(f"smoke test ok: {out.shape}, finite={np.isfinite(out.astype(np.float32)).all()}")
        return
    np.save(QWEN_FILE, out)
    print(f"saved {QWEN_FILE} {out.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
