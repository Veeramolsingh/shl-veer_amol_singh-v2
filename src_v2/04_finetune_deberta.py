"""v2 Step 4 - DeBERTa-v3-base fine-tuned on the transcript, trained WITHOUT the label-0 clips, 3 seeds averaged.

The 37 label-0 clips are a separate batch that is not in the test set; training on them distorted v1's DeBERTa.
Dropping them improves its out-of-fold RMSE on real (label > 0) clips from 0.636 to 0.584.
Same 5 folds as the ensemble; the label-0 rows still get an OOF prediction (they are just never trained on).
Model setup is the same as v1 (frozen embeddings, gradient checkpointing, fp16, no dropout) and fits a 4 GB GPU.

Output: features/v2/deberta_nz.csv  (key, deberta_nz) - OOF for train rows, mean of all fold models for test rows
Runtime: ~9 min per seed on a 4 GB GPU (~27 min for the default 3 seeds).
"""
import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from common_v2 import DEBERTA_FILE, N_FOLDS, load_meta, load_transcripts

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 384


class TextRegressor(nn.Module):
    def __init__(self, init_bias):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.backbone.float()  # AMP needs fp32 master weights
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0)).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8, help="lower to 4 if you get CUDA out-of-memory")
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    dev = torch.device("cuda")
    meta = load_meta()
    transcripts = load_transcripts()
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok([transcripts[k]["text"] for k in meta["key"]], truncation=True, max_length=MAX_LEN)

    y = meta["label"].to_numpy()
    folds = meta["fold"].to_numpy()
    train_idx = np.where(meta["split"] == "train")[0]
    test_idx = np.where(meta["split"] == "test")[0]

    def make_batch(idx):
        b = tok.pad({"input_ids": [enc["input_ids"][j] for j in idx],
                     "attention_mask": [enc["attention_mask"][j] for j in idx]}, return_tensors="pt")
        return b["input_ids"].to(dev), b["attention_mask"].to(dev)

    @torch.inference_mode()
    def predict(model, idx):
        model.eval()
        order = sorted(idx, key=lambda j: len(enc["input_ids"][j]))  # length-sorted -> little padding
        preds = {}
        for i in range(0, len(order), 32):
            chunk = order[i:i + 32]
            ids, mask = make_batch(chunk)
            with torch.autocast("cuda", dtype=torch.float16):
                preds.update(zip(chunk, model(ids, mask).float().cpu().numpy()))
        return np.array([preds[j] for j in idx])

    oof = np.zeros(len(meta))
    test_pred = np.zeros(len(test_idx))
    t0 = time.time()
    for seed in args.seeds:
        for fold in range(N_FOLDS):
            seed_everything(seed + fold)
            fit_idx = train_idx[(folds[train_idx] != fold) & (y[train_idx] > 0)]  # label-0 clips never trained on
            val_idx = train_idx[folds[train_idx] == fold]

            model = TextRegressor(float(y[fit_idx].mean())).to(dev)
            opt = torch.optim.AdamW(
                [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
            steps = args.epochs * math.ceil(len(fit_idx) / args.batch_size)
            sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler("cuda")
            rng = np.random.default_rng(seed + fold)

            for _ in range(args.epochs):  # fixed epochs, no best-epoch picking -> honest OOF
                model.train()
                perm = rng.permutation(fit_idx)
                for i in range(0, len(perm), args.batch_size):
                    b = perm[i:i + args.batch_size]
                    ids, mask = make_batch(b)
                    target = torch.tensor(y[b], dtype=torch.float32, device=dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(ids, mask)
                    loss = F.mse_loss(pred.float(), target)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()

            p = predict(model, val_idx)
            oof[val_idx] += p / len(args.seeds)
            test_pred += predict(model, test_idx) / (N_FOLDS * len(args.seeds))
            nz = y[val_idx] > 0
            print(f"seed {seed} fold {fold}: val RMSE (label>0) {np.sqrt(np.mean((p[nz] - y[val_idx][nz]) ** 2)):.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
            del model, opt
            torch.cuda.empty_cache()

    nz = y[train_idx] > 0
    print(f"DeBERTa OOF RMSE (label>0, {len(args.seeds)} seeds): "
          f"{np.sqrt(np.mean((oof[train_idx][nz] - y[train_idx][nz]) ** 2)):.4f}")
    out = pd.DataFrame({"key": meta["key"], "deberta_nz": oof})
    out.loc[test_idx, "deberta_nz"] = test_pred
    out.to_csv(DEBERTA_FILE, index=False)
    print(f"saved {DEBERTA_FILE}")


if __name__ == "__main__":
    main()
