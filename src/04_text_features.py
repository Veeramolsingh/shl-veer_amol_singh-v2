"""Step 4 - Grammar-oriented features from the Whisper transcripts.

Four stages, each cached to its own CSV so a crash never repeats finished work:
  stats : fluency & lexical statistics - speech rate, pauses (from word timestamps), fillers,
          repetitions, vocabulary richness / rarity, sentence length, Whisper confidence,
          language-ID probability.
  cola  : grammatical acceptability of every sentence (RoBERTa fine-tuned on CoLA), plus a
          768-d transcript embedding from the same encoder (features/cola_emb.npy).
  gec   : grammatical error correction (CoEdIT-large). Word-level edits between a sentence and
          its corrected version ~ number of grammar errors -> the most direct grammar signal.
  lm    : language-model surprisal (Qwen2.5-0.5B) - ungrammatical word sequences are improbable.

Output: features/text_features.csv (+ cola_emb.npy, gec_corrections.csv)
"""
import argparse
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from common import (FEAT_DIR, FILLERS, get_device, load_meta, load_transcripts, split_sentences, strip_fillers,
                    word_edit_distance, words_of)

COLA_MODEL = "textattack/roberta-base-CoLA"
GEC_MODEL = "grammarly/coedit-large"
GEC_PREFIX = "Fix grammatical errors in this sentence: "
LM_MODEL = "Qwen/Qwen2.5-0.5B"

SUBORDINATORS = {"because", "although", "though", "which", "who", "whom", "whose", "when", "whenever", "while",
                 "whereas", "if", "unless", "since", "until", "after", "before", "that", "where", "whether"}
TIMING_KEYS = ["speak_span_ratio", "wpm_span", "artic_rate", "pauses_03_per_min", "pauses_1_per_min",
               "mean_pause", "max_pause", "word_prob_mean", "word_prob_p10", "low_prob_frac"]

try:
    from wordfreq import zipf_frequency
except ImportError:  # optional dependency
    zipf_frequency = None


# ============================================================================ stage: stats
def timing_stats(words, duration):
    """Speech-rate and pause features from Whisper word timestamps and word confidences."""
    if len(words) < 2:
        return dict.fromkeys(TIMING_KEYS, np.nan)
    start = np.array([w["s"] for w in words])
    end = np.array([w["e"] for w in words])
    prob = np.array([w["p"] for w in words])
    span = max(end[-1] - start[0], 1e-3)
    gaps = np.clip(start[1:] - end[:-1], 0, None)
    phonation = max(np.clip(end - start, 0, None).sum(), 1e-3)
    return {
        "speak_span_ratio": span / duration,
        "wpm_span": len(words) / span * 60,
        "artic_rate": len(words) / phonation * 60,
        "pauses_03_per_min": (gaps > 0.3).sum() / span * 60,
        "pauses_1_per_min": (gaps > 1.0).sum() / span * 60,
        "mean_pause": gaps[gaps > 0.3].mean() if (gaps > 0.3).any() else 0.0,
        "max_pause": gaps.max(),
        "word_prob_mean": prob.mean(),
        "word_prob_p10": np.percentile(prob, 10),
        "low_prob_frac": (prob < 0.5).mean(),
    }


def transcript_stats(rec):
    duration = max(rec["duration"], 1e-3)
    tokens = words_of(rec["text"])
    content = [t for t in tokens if t not in FILLERS]
    n = len(content)
    f = {
        "en_prob": rec["en_prob"],
        "top_lang_prob": rec["top_lang_prob"],
        "is_english": float(rec["top_lang"] == "en"),
        "vad_speech_ratio": rec["speech_duration"] / duration,
        "n_words": n,
        "wpm": n / duration * 60,
        "filler_rate": (len(tokens) - n) / max(len(tokens), 1),
        "repeat_rate": sum(a == b for a, b in zip(content, content[1:])) / max(n, 1),
        "root_ttr": len(set(content)) / np.sqrt(n) if n else np.nan,  # length-robust vocabulary richness
        "mean_word_len": np.mean([len(t) for t in content]) if n else np.nan,
        "long_word_frac": np.mean([len(t) >= 7 for t in content]) if n else np.nan,
        "subord_per_100w": 100 * sum(t in SUBORDINATORS for t in content) / n if n else np.nan,
    }
    sent_lens = [len(words_of(s)) for s in split_sentences(strip_fillers(rec["text"]), max_words=10_000)]
    f["n_sentences"] = len(sent_lens)
    f["sent_len_mean"] = np.mean(sent_lens) if sent_lens else np.nan
    f["sent_len_std"] = np.std(sent_lens) if sent_lens else np.nan
    f["sent_len_max"] = max(sent_lens) if sent_lens else np.nan
    if zipf_frequency is not None:
        zipf = np.array([zipf_frequency(t, "en") for t in content])
        f["zipf_mean"] = zipf.mean() if n else np.nan
        f["rare_word_frac"] = (zipf < 3.5).mean() if n else np.nan

    segs = rec["segments"]
    f["seg_logprob_mean"] = np.mean([s["avg_logprob"] for s in segs]) if segs else np.nan
    f["seg_logprob_min"] = np.min([s["avg_logprob"] for s in segs]) if segs else np.nan
    f["no_speech_mean"] = np.mean([s["no_speech_prob"] for s in segs]) if segs else np.nan
    f["compression_mean"] = np.mean([s["compression_ratio"] for s in segs]) if segs else np.nan
    f.update(timing_stats(rec["words"], duration))
    return f


def stage_stats(keys, recs):
    return pd.DataFrame([{"key": k, **transcript_stats(r)} for k, r in zip(keys, recs)])


# ============================================================================ stage: cola
@torch.inference_mode()
def acceptability(tok, model, sentences, device, batch_size=64):
    """P(label 1 = grammatically acceptable) for each sentence; length-sorted batches."""
    probs = np.zeros(len(sentences))
    order = np.argsort([len(s) for s in sentences])
    for i in tqdm(range(0, len(order), batch_size), desc="cola sentences"):
        idx = order[i : i + batch_size]
        enc = tok([sentences[j] for j in idx], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(device)
        probs[idx] = model(**enc).logits.float().softmax(-1)[:, 1].cpu().numpy()
    return probs


@torch.inference_mode()
def mean_pooled_embeddings(tok, model, texts, device, batch_size=16):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="cola embeddings"):
        enc = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(out)


def stage_cola(keys, texts, device):
    tok = AutoTokenizer.from_pretrained(COLA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL).to(device).eval()

    items = [(k, s) for k, t in zip(keys, texts) for s in split_sentences(strip_fillers(t))
             if len(s.split()) >= 3]
    df = pd.DataFrame(items, columns=["key", "sent"])
    df["p"] = acceptability(tok, model, df["sent"].tolist(), device) if len(df) else []
    df["nw"] = df["sent"].str.split().str.len()
    df["bad"] = (df["p"] < 0.5).astype(float)
    df["pw"] = df["p"] * df["nw"]
    agg = df.groupby("key").agg(cola_mean=("p", "mean"), cola_min=("p", "min"), cola_std=("p", "std"),
                                cola_bad_frac=("bad", "mean"), pw=("pw", "sum"), nw=("nw", "sum"))
    agg["cola_wmean"] = agg["pw"] / agg["nw"]
    feats = pd.DataFrame({"key": keys}).merge(agg.drop(columns=["pw", "nw"]), on="key", how="left")

    np.save(FEAT_DIR / "cola_emb.npy", mean_pooled_embeddings(tok, model, texts, device).astype(np.float32))
    return feats


# ============================================================================ stage: gec
@torch.inference_mode()
def correct(tok, model, sentences, device, batch_size=16):
    corrected = [""] * len(sentences)
    order = np.argsort([len(s) for s in sentences])
    for i in tqdm(range(0, len(order), batch_size), desc="grammar correction"):
        idx = order[i : i + batch_size]
        enc = tok([GEC_PREFIX + sentences[j] for j in idx], padding=True, truncation=True, max_length=192,
                  return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=160, num_beams=1, do_sample=False)
        for j, text in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
            corrected[j] = text
    return corrected


def stage_gec(keys, texts, device):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(GEC_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL).to(device=device, dtype=dtype).eval()

    items = [(k, s) for k, t in zip(keys, texts) for s in split_sentences(strip_fillers(t))
             if len(words_of(s)) >= 2]
    df = pd.DataFrame(items, columns=["key", "sent"])
    df["corrected"] = correct(tok, model, df["sent"].tolist(), device) if len(df) else []
    df["nw"] = [len(words_of(s)) for s in df["sent"]]
    df["edits"] = [word_edit_distance(words_of(a), words_of(b)) for a, b in zip(df["sent"], df["corrected"])]
    df["changed"] = (df["edits"] > 0).astype(float)
    df.to_csv(FEAT_DIR / "gec_corrections.csv", index=False)  # kept for inspection in the notebook

    agg = df.groupby("key").agg(edits=("edits", "sum"), nw=("nw", "sum"), gec_changed_frac=("changed", "mean"),
                                gec_max_edits=("edits", "max"), gec_mean_edits=("edits", "mean"))
    agg["gec_edits_per_100w"] = 100 * agg["edits"] / agg["nw"].clip(lower=1)
    return pd.DataFrame({"key": keys}).merge(agg.drop(columns=["edits", "nw"]), on="key", how="left")


# ============================================================================ stage: lm
@torch.inference_mode()
def stage_lm(keys, texts, device):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device=device, dtype=dtype).eval()

    rows = []
    for k, t in tqdm(list(zip(keys, texts)), desc="lm surprisal"):
        ids = tok(strip_fillers(t), return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        if ids.shape[1] < 4:
            rows.append({"key": k})
            continue
        logits = model(ids).logits[0, :-1].float()
        nll = F.cross_entropy(logits, ids[0, 1:], reduction="none").cpu().numpy()
        rows.append({"key": k, "lm_nll_mean": nll.mean(), "lm_nll_median": np.median(nll),
                     "lm_nll_p90": np.percentile(nll, 90), "lm_surprise_frac": (nll > 6).mean()})
    return pd.DataFrame(rows)


# ============================================================================ main
def cached(name, fn, force, extra_files=()):
    path = FEAT_DIR / f"text_{name}.csv"
    if path.exists() and all((FEAT_DIR / f).exists() for f in extra_files) and name not in force:
        print(f"[{name}] cached -> {path.name}")
        return pd.read_csv(path)
    print(f"[{name}] computing ...")
    df = fn()
    df.to_csv(path, index=False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], help="stages to recompute, e.g. --force gec lm")
    args = ap.parse_args()

    meta = load_meta()
    transcripts = load_transcripts()
    missing = set(meta["key"]) - set(transcripts)
    if missing:
        raise SystemExit(f"{len(missing)} files are not transcribed yet - finish src/02_transcribe.py first.")

    keys = meta["key"].tolist()
    recs = [transcripts[k] for k in keys]
    texts = [r["text"] for r in recs]
    device = get_device()

    stats = cached("stats", lambda: stage_stats(keys, recs), args.force)
    cola = cached("cola", lambda: stage_cola(keys, texts, device), args.force, extra_files=["cola_emb.npy"])
    gec = cached("gec", lambda: stage_gec(keys, texts, device), args.force)
    lm = cached("lm", lambda: stage_lm(keys, texts, device), args.force)

    feats = stats.merge(cola, on="key").merge(gec, on="key").merge(lm, on="key", how="left")
    feats.to_csv(FEAT_DIR / "text_features.csv", index=False)
    print(f"saved text_features.csv {feats.shape}")


if __name__ == "__main__":
    main()
