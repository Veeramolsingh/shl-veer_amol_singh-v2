"""v2 Step 5 - Train the v2 ensemble, evaluate it, and write the submission.

What changed from v1 (src/06_train_ensemble.py) and why:
  * The test set is dominated by a different recording batch (45 s clips: 56% of test, 17% of train) that gets
    lower scores on the same prompts. Every base model gets two batch flags (is_45s, is_other) as extra inputs.
  * The 37 label-0 clips (a separate batch, absent from the test set) distort the text models, so text/tabular
    models and the stacker are trained on label > 0 rows only.
  * Two new base models: ridge on Qwen2.5-1.5B transcript embeddings (step 2) and ridge on Whisper encoder
    embeddings (step 1). DeBERTa is replaced by the label>0 version (step 4). Tabular features gain the
    grammar-error types (step 3).
  * One global non-negative linear stacker fitted on label > 0 rows; predictions clipped to [2, 5]
    (only 4 of 732 real training labels are below 2).

Evaluation: 5-fold CV on the v1 folds, reported as RMSE on label>0 clips and as test-weighted RMSE (clips
re-weighted to the test set's batch mix). --grouped-check additionally compares v1-style vs v2 stacks on 6 CV
splits that hold out whole prompts (about half the test clips come from prompts never seen in training).

Outputs (outputs/v2/): submission_v2.csv, cv_report_v2.json, oof_predictions_v2.csv (contains train labels)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common_v2 import (DEBERTA_FILE, ERRTYPES_FILE, FEAT_DIR, N_FOLDS, OUT_DIR, QWEN_FILE, SEED, V2_OUT,
                       WHISPER_FILE, load_meta, load_transcripts, task_flags, task_of, test_mix_weights)

warnings.filterwarnings("ignore", category=UserWarning)
ALPHAS = np.logspace(-1, 5, 25)
LENGTH_DEPENDENT = ["n_words", "n_sentences", "gec_max_edits"]  # test clips are shorter -> use rates only
WAVLM_LAYERS = [20, 21, 22]    # best WavLM layer (21) and neighbours, chosen by CV in v1
QWEN_LAYERS = [17, 18, 19]     # best Qwen2.5-1.5B layers by CV
WHISPER_LAYERS = [27, 28, 29]  # best Whisper encoder layers by CV
CLIP = (2.0, 5.0)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def svr(c=3.0, eps=0.1):
    return make_pipeline(StandardScaler(), SVR(C=c, epsilon=eps))


def lgbm():
    return LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                         random_state=SEED, verbose=-1, n_jobs=3)


def run_cv(model, X_tr, y, folds, X_te=None, fit_mask=None):
    """OOF predictions for every train row (models fitted only on fit_mask rows); test = mean of fold models."""
    fit_mask = np.ones(len(y), bool) if fit_mask is None else fit_mask

    def one(f):
        m = clone(model).fit(X_tr[(folds != f) & fit_mask], y[(folds != f) & fit_mask])
        return f, m.predict(X_tr[folds == f]), (m.predict(X_te) if X_te is not None else None)

    oof = np.zeros(len(y))
    test = None if X_te is None else np.zeros(len(X_te))
    for f, p_val, p_te in Parallel(n_jobs=N_FOLDS)(delayed(one)(f) for f in np.unique(folds)):
        oof[folds == f] = p_val
        if test is not None:
            test += p_te / len(np.unique(folds))
    return oof, test


def stack_oof(P, y, folds, fit_mask, clip):
    """Nested CV of a non-negative linear stacker -> honest OOF of the whole ensemble."""
    out = np.zeros(len(y))
    for f in np.unique(folds):
        m = LinearRegression(positive=True).fit(P[(folds != f) & fit_mask], y[(folds != f) & fit_mask])
        out[folds == f] = m.predict(P[folds == f])
    return np.clip(out, *clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped-check", action="store_true",
                    help="also compare v1-style vs v2 stacks on 6 prompt-grouped CV splits (~5 extra minutes)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ data
    meta = load_meta()
    is_te = (meta["split"] == "test").to_numpy()
    is_tr = ~is_te
    tr, te = meta[is_tr].reset_index(drop=True), meta[is_te].reset_index(drop=True)
    y, folds = tr["label"].to_numpy(), tr["fold"].to_numpy()
    nz = y > 0
    w_test = test_mix_weights(meta, is_tr, y)
    task_tr = task_of(tr["duration"])

    def rmse(p, w=None):
        e2 = (np.asarray(p) - y) ** 2
        return float(np.sqrt(e2[nz].mean())) if w is None else float(np.sqrt((w * e2).sum() / w.sum()))

    def scores(p):
        return {"rmse_label_gt0": rmse(p), "rmse_test_weighted": rmse(p, w_test)}

    # ------------------------------------------------------------------ features (rows aligned to meta order)
    for f in (WHISPER_FILE, QWEN_FILE, ERRTYPES_FILE, DEBERTA_FILE):
        if not f.exists():
            raise SystemExit(f"{f} not found - run steps 1-4 of src_v2 first.")

    def aligned(path):
        return pd.read_csv(path).set_index("key").reindex(meta["key"])

    tab = pd.concat([aligned(FEAT_DIR / "acoustic.csv"), aligned(FEAT_DIR / "text_features.csv")], axis=1)
    tab = tab.drop(columns=[c for c in LENGTH_DEPENDENT if c in tab.columns])
    X_tab_v1 = tab.replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
    tab = pd.concat([tab, aligned(ERRTYPES_FILE)], axis=1).replace([np.inf, -np.inf], np.nan)
    X_tab = tab.to_numpy(np.float64)
    wavlm = np.load(FEAT_DIR / "wavlm_layers.npy")
    X_wav = wavlm[:, WAVLM_LAYERS].astype(np.float32).mean(1)
    del wavlm
    X_cola = np.load(FEAT_DIR / "cola_emb.npy")
    X_qwen = np.load(QWEN_FILE)[:, QWEN_LAYERS].astype(np.float32).mean(1)
    X_whisper = np.load(WHISPER_FILE)[:, WHISPER_LAYERS].astype(np.float32).mean(1)
    deberta = aligned(DEBERTA_FILE)["deberta_nz"].to_numpy()
    D = task_flags(meta)

    def flagged(X):
        return np.hstack([X, D])

    # name -> (model, features, train on label>0 only?)
    specs = {
        "wavlm_svr": (svr(3, 0.05), flagged(X_wav), False),
        "wavlm_ridge": (ridge(), flagged(X_wav), True),
        "cola_svr": (svr(3, 0.1), flagged(X_cola), True),
        "tab_ridge": (ridge(), flagged(X_tab), True),
        "tab_lgbm": (lgbm(), flagged(X_tab), True),
        "qwen_ridge": (ridge(), flagged(X_qwen), True),
        "whisper_ridge": (ridge(), flagged(X_whisper), True),
    }
    print(f"train rows {len(tr)} ({nz.sum()} with label > 0) | test rows {len(te)} | tabular features {X_tab.shape[1]}")

    # ------------------------------------------------------------------ base models
    oof, test = {}, {}
    for name, (model, X, nz_only) in specs.items():
        oof[name], test[name] = run_cv(model, X[is_tr], y, folds, X[is_te], fit_mask=nz if nz_only else None)
        s = scores(oof[name])
        print(f"  {name:14s} OOF RMSE label>0 {s['rmse_label_gt0']:.4f} | test-weighted {s['rmse_test_weighted']:.4f}",
              flush=True)
    oof["deberta_nz"], test["deberta_nz"] = deberta[is_tr], deberta[is_te]
    s = scores(oof["deberta_nz"])
    print(f"  {'deberta_nz':14s} OOF RMSE label>0 {s['rmse_label_gt0']:.4f} | test-weighted {s['rmse_test_weighted']:.4f}")

    # ------------------------------------------------------------------ stacking
    names = list(oof)
    P_oof = np.column_stack([oof[n] for n in names])
    P_test = np.column_stack([test[n] for n in names])
    ens_oof = stack_oof(P_oof, y, folds, nz, CLIP)
    stacker = LinearRegression(positive=True).fit(P_oof[nz], y[nz])
    final_test = np.clip(stacker.predict(P_test), *CLIP)

    ens = scores(ens_oof)
    weights = dict(zip(names, np.round(stacker.coef_, 4).tolist()))
    print(f"\nstacker weights: {weights}  intercept {stacker.intercept_:+.3f}")
    print(f"v2 ENSEMBLE (nested 5-fold CV): RMSE label>0 {ens['rmse_label_gt0']:.4f} | "
          f"test-weighted {ens['rmse_test_weighted']:.4f}")
    by_task = {t: float(np.sqrt(((ens_oof - y)[nz & (task_tr == t)] ** 2).mean())) for t in ("45s", "60s", "other")}
    bias = {t: float((ens_oof - y)[nz & (task_tr == t)].mean()) for t in ("45s", "60s", "other")}
    print("  RMSE by batch:", {k: round(v, 4) for k, v in by_task.items()},
          "| bias by batch:", {k: round(v, 3) for k, v in bias.items()})

    v1 = {}
    v1_oof_path = OUT_DIR / "oof_predictions.csv"
    if v1_oof_path.exists():  # same folds and rows -> directly comparable
        v1_oof = pd.read_csv(v1_oof_path).set_index("key").reindex(tr["key"])["ensemble"].to_numpy()
        v1 = scores(v1_oof)
        print(f"v1 ENSEMBLE (same folds)      : RMSE label>0 {v1['rmse_label_gt0']:.4f} | "
              f"test-weighted {v1['rmse_test_weighted']:.4f}")

    report = {"v2_cv": ens, "v1_cv_same_folds": v1, "rmse_by_batch": by_task, "bias_by_batch": bias,
              "stacker_weights": weights, "stacker_intercept": float(stacker.intercept_),
              "base_models": {n: scores(oof[n]) for n in names}, "clip": CLIP,
              "test_pred_summary": pd.Series(final_test).describe().round(4).to_dict()}

    if args.grouped_check:
        report["grouped_check"] = grouped_check(meta, y, nz, w_test, is_tr, X_wav, X_cola, X_tab_v1,
                                                specs, rmse)

    # ------------------------------------------------------------------ files
    sub = pd.DataFrame({"filename": te["filename"], "label": final_test})
    assert len(sub) == 216 and sub["label"].between(0, 5).all() and sub["filename"].is_unique
    sub.to_csv(V2_OUT / "submission_v2.csv", index=False)
    (V2_OUT / "cv_report_v2.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({"key": tr["key"], "label": y, "ensemble": ens_oof, **oof}).to_csv(
        V2_OUT / "oof_predictions_v2.csv", index=False)
    print(f"\ntest predictions: mean {final_test.mean():.3f}, sd {final_test.std():.3f}, "
          f"min {final_test.min():.3f}, max {final_test.max():.3f}")
    print(f"saved {V2_OUT / 'submission_v2.csv'} + cv_report_v2.json + oof_predictions_v2.csv")


def grouped_check(meta, y, nz, w_test, is_tr, X_wav, X_cola, X_tab_v1, specs, rmse):
    """v1-style stack vs v2 stack (both without DeBERTa) on 6 CV splits that hold out whole prompts.

    Prompts are approximated by clustering the transcripts (TF-IDF + k-means, 12/20/30 clusters); whole clusters
    are assigned to folds. DeBERTa is left out of both stacks because re-training it per split is slow.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    transcripts = load_transcripts()
    texts = [transcripts[k]["text"] for k in meta["key"]]
    X_txt = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.3, sublinear_tf=True).fit_transform(texts)
    splits = []
    for n_clusters, seed in [(12, 1), (12, 2), (20, 3), (20, 4), (30, 5), (30, 6)]:
        cl = KMeans(n_clusters, n_init=5, random_state=seed).fit(X_txt).labels_[is_tr]
        splits.append(np.random.default_rng(seed).permutation(n_clusters)[cl] % 5)

    v1_specs = {"wavlm_svr": (svr(3, 0.1), X_wav), "wavlm_ridge": (ridge(), X_wav), "cola_svr": (svr(3, 0.1), X_cola),
                "tab_ridge": (ridge(), X_tab_v1), "tab_lgbm": (lgbm(), X_tab_v1)}
    rows = []
    print("\nprompt-grouped check (test-weighted RMSE, no DeBERTa in either stack):")
    for i, folds in enumerate(splits):
        P1 = np.column_stack([run_cv(m, X[is_tr], y, folds)[0] for m, X in v1_specs.values()])
        p1 = np.zeros(len(y))
        for f in np.unique(folds):  # v1 stacker: all rows (incl. label 0), clip [0, 5]
            p1[folds == f] = LinearRegression(positive=True).fit(P1[folds != f], y[folds != f]).predict(P1[folds == f])
        p1 = np.clip(p1, 0, 5)
        P2 = np.column_stack([run_cv(m, X[is_tr], y, folds, fit_mask=nz if nzo else None)[0]
                              for m, X, nzo in specs.values()])
        p2 = stack_oof(P2, y, folds, nz, CLIP)
        rows.append({"split": i, "v1": rmse(p1, w_test), "v2": rmse(p2, w_test)})
        print(f"  split {i}: v1-style {rows[-1]['v1']:.4f} | v2 {rows[-1]['v2']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    wins = int((df["v2"] < df["v1"]).sum())
    print(f"  mean: v1-style {df['v1'].mean():.4f} | v2 {df['v2'].mean():.4f} | v2 better on {wins}/6 splits")
    return {"per_split": rows, "v1_mean": float(df["v1"].mean()), "v2_mean": float(df["v2"].mean()), "v2_wins": wins}


if __name__ == "__main__":
    main()
