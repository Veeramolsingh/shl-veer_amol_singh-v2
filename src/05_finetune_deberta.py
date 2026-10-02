"""Step 5 - Fine-tune DeBERTa-v3-base to regress the grammar score directly from the transcript.

Same 5 folds as the ensemble, so the out-of-fold (OOF) predictions can be stacked in step 6.
Sized for a 4 GB GPU: the 98M-parameter embedding matrix is frozen (saves ~1.2 GB of
gradients + Adam state), gradient checkpointing is on, fp16 autocast, dynamic padding.
Dropout is disabled - a well-known trick that helps transformer regression.

Output: features/deberta_preds.csv
  deberta          OOF prediction for train rows, mean of the 5 fold models for test rows
  deberta_insample mean of the 5 fold models on train rows (only used to report training RMSE)
"""
import argparse
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from common import FEAT_DIR, N_FOLDS, SEED, get_device, load_meta, load_transcripts

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 384


class TextRegressor(nn.Module):
    def __init__(self, init_bias: float):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        # newer transformers keep the checkpoint's fp16 dtype; AMP needs fp32 master weights
        self.backbone.float()
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, init_bias)  # start at the mean label -> stable first steps

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # mean pooling over real tokens
        return self.head(pooled).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(tok, enc, idx, device):
    batch = tok.pad({"input_ids": [enc["input_ids"][j] for j in idx],
                     "attention_mask": [enc["attention_mask"][j] for j in idx]}, return_tensors="pt")
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.inference_mode()
def predict(model, tok, enc, idx, device, batch_size=32):
    model.eval()
    order = sorted(idx, key=lambda j: len(enc["input_ids"][j]))  # length-sorted -> little padding
    preds = {}
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        ids, mask = make_batch(tok, enc, chunk, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(ids, mask).float().cpu().numpy()
        preds.update(zip(chunk, out))
    return np.array([preds[j] for j in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8, help="lower to 4 if you get CUDA out-of-memory")
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    device = get_device()
    use_amp = device.type == "cuda"
    meta = load_meta()
    transcripts = load_transcripts()
    texts = [transcripts[k]["text"] for k in meta["key"]]
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(texts, truncation=True, max_length=MAX_LEN)

    train_idx_all = np.where(meta["split"] == "train")[0]
    test_idx = np.where(meta["split"] == "test")[0]
    y = meta["label"].to_numpy()
    folds = meta["fold"].to_numpy()

    oof = np.full(len(meta), np.nan)
    test_pred = np.zeros(len(test_idx))
    insample = np.zeros(len(train_idx_all))

    for fold in range(N_FOLDS):
        seed_everything(SEED + fold)
        tr_idx = train_idx_all[folds[train_idx_all] != fold]
        va_idx = train_idx_all[folds[train_idx_all] == fold]

        model = TextRegressor(float(y[tr_idx].mean())).to(device)
        opt = torch.optim.AdamW(
            [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
             {"params": model.head.parameters(), "lr": 1e-3}],
            weight_decay=0.01,
        )
        steps = args.epochs * math.ceil(len(tr_idx) / args.batch_size)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(SEED + fold)

        for epoch in range(args.epochs):
            model.train()
            perm = rng.permutation(tr_idx)
            losses = []
            for i in range(0, len(perm), args.batch_size):
                b = perm[i : i + args.batch_size]
                ids, mask = make_batch(tok, enc, b, device)
                target = torch.tensor(y[b], dtype=torch.float32, device=device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(ids, mask)
                loss = F.mse_loss(pred.float(), target)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            va_pred = predict(model, tok, enc, va_idx, device)
            va_rmse = np.sqrt(np.mean((va_pred - y[va_idx]) ** 2))
            print(f"fold {fold} epoch {epoch + 1}/{args.epochs}  train_mse {np.mean(losses):.3f}  "
                  f"val_rmse {va_rmse:.4f}")

        # fixed number of epochs (no best-epoch picking) keeps the OOF estimate honest
        oof[va_idx] = va_pred
        test_pred += predict(model, tok, enc, test_idx, device) / N_FOLDS
        insample += predict(model, tok, enc, train_idx_all, device) / N_FOLDS
        del model, opt
        torch.cuda.empty_cache()

    cv = np.sqrt(np.mean((oof[train_idx_all] - y[train_idx_all]) ** 2))
    print(f"DeBERTa OOF RMSE: {cv:.4f}")

    out = pd.DataFrame({"key": meta["key"], "deberta": oof, "deberta_insample": np.nan})
    out.loc[test_idx, "deberta"] = test_pred
    out.loc[train_idx_all, "deberta_insample"] = insample
    out.to_csv(FEAT_DIR / "deberta_preds.csv", index=False)
    print("saved deberta_preds.csv")


if __name__ == "__main__":
    main()
