"""Step 6 - Train the ensemble, evaluate it with 5-fold CV and write the submission.

Base models (each produces out-of-fold predictions on the same 5 folds):
  tab_lgbm    LightGBM on the hand-crafted acoustic + text features
  tab_ridge   Ridge on the same features (standardised, median-imputed)
  wavlm_ridge Ridge on the best WavLM layer(s)       - chosen by CV over all 25 layers
  wavlm_svr   RBF-SVR on the same WavLM features
  cola_svr    RBF-SVR on the CoLA-RoBERTa transcript embedding
  deberta     fine-tuned DeBERTa-v3 (step 5, optional)

A non-negative linear stacker combines the OOF predictions. Its own CV score is computed
with a second (nested) pass over the same folds, so every reported CV number is honest.

Outputs (outputs/): submission.csv, cv_report.json, oof_predictions.csv, *.png plots
"""
import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import FEAT_DIR, N_FOLDS, OUT_DIR, SEED, load_meta

warnings.filterwarnings("ignore", category=UserWarning)
ALPHAS = np.logspace(-1, 5, 25)
LENGTH_DEPENDENT = ["n_words", "n_sentences", "gec_max_edits"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pearson(a, b):
    return float(pearsonr(a, b)[0])


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def svr(c):
    return make_pipeline(StandardScaler(), SVR(C=c, epsilon=0.1))


def lgbm():
    return LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                         random_state=SEED, verbose=-1)


class CV:
    """Runs every model on the same folds; keeps OOF, test and in-sample predictions."""

    def __init__(self, y, folds):
        self.y, self.folds = y, folds

    def oof(self, model, X):
        pred = np.zeros(len(self.y))
        for f in range(N_FOLDS):
            m = clone(model).fit(X[self.folds != f], self.y[self.folds != f])
            pred[self.folds == f] = m.predict(X[self.folds == f])
        return pred

    def run(self, model, X_tr, X_te):
        """-> (OOF predictions, test predictions = mean of fold models, in-sample predictions of a full fit)."""
        oof, test = np.zeros(len(self.y)), np.zeros(len(X_te))
        for f in range(N_FOLDS):
            m = clone(model).fit(X_tr[self.folds != f], self.y[self.folds != f])
            oof[self.folds == f] = m.predict(X_tr[self.folds == f])
            test += m.predict(X_te) / N_FOLDS
        full = clone(model).fit(X_tr, self.y)
        return oof, test, full.predict(X_tr), full


def pick_svr_c(cv, X, grid=(1, 3, 10)):
    scores = {c: rmse(cv.oof(svr(c), X), cv.y) for c in grid}
    return min(scores, key=scores.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-zero", action="store_true",
                    help="exclude the label-0 clips (a separate batch, files 5037-5073) from training")
    args = ap.parse_args()

    meta = load_meta()
    is_te = (meta["split"] == "test").to_numpy()
    is_tr = ~is_te
    if args.drop_zero:
        is_tr &= (meta["label"] > 0).to_numpy()
    tr, te = meta[is_tr].reset_index(drop=True), meta[is_te].reset_index(drop=True)
    y, folds = tr["label"].to_numpy(), tr["fold"].to_numpy()
    cv = CV(y, folds)
    print(f"training rows: {len(tr)} ({'label-0 batch dropped' if args.drop_zero else 'all labels'})")

    # ------------------------------------------------------------------ load features (aligned to meta order)
    def aligned(name):
        return pd.read_csv(FEAT_DIR / name).set_index("key").reindex(meta["key"])

    tab = pd.concat([aligned("acoustic.csv"), aligned("text_features.csv")], axis=1)
    # Test clips are shorter (median 45 s vs 60 s in train): drop totals that scale with clip length
    # and keep only their per-minute / per-word versions.
    tab = tab.drop(columns=[c for c in LENGTH_DEPENDENT if c in tab.columns])
    tab = tab.replace([np.inf, -np.inf], np.nan)
    tab_cols = tab.columns.tolist()
    X_tab = tab.to_numpy(dtype=np.float64)
    wavlm = np.load(FEAT_DIR / "wavlm_layers.npy").astype(np.float32)
    cola_emb = np.load(FEAT_DIR / "cola_emb.npy")
    print(f"tabular features: {len(tab_cols)} | WavLM {wavlm.shape} | CoLA embedding {cola_emb.shape}")

    results = {}  # name -> (oof, test, insample)

    # ------------------------------------------------------------------ tabular models
    oof, test, ins, lgbm_full = cv.run(lgbm(), X_tab[is_tr], X_tab[is_te])
    results["tab_lgbm"] = (oof, test, ins)
    oof, test, ins, _ = cv.run(ridge(), X_tab[is_tr], X_tab[is_te])
    results["tab_ridge"] = (oof, test, ins)

    # ------------------------------------------------------------------ WavLM: choose layers by CV
    layer_rmse = [rmse(cv.oof(ridge(), wavlm[is_tr, L]), y) for L in range(wavlm.shape[1])]
    best = int(np.argmin(layer_rmse))
    layers = [L for L in (best - 1, best, best + 1) if 0 <= L < wavlm.shape[1]]
    X_wav = wavlm[:, layers].mean(axis=1)  # average of the best layer and its neighbours
    print(f"best WavLM layer {best} (ridge CV RMSE {layer_rmse[best]:.4f}) -> using layers {layers}")
    oof, test, ins, _ = cv.run(ridge(), X_wav[is_tr], X_wav[is_te])
    results["wavlm_ridge"] = (oof, test, ins)
    c = pick_svr_c(cv, X_wav[is_tr])
    oof, test, ins, _ = cv.run(svr(c), X_wav[is_tr], X_wav[is_te])
    results["wavlm_svr"] = (oof, test, ins)

    # ------------------------------------------------------------------ transcript embedding
    c = pick_svr_c(cv, cola_emb[is_tr])
    oof, test, ins, _ = cv.run(svr(c), cola_emb[is_tr], cola_emb[is_te])
    results["cola_svr"] = (oof, test, ins)

    # ------------------------------------------------------------------ fine-tuned DeBERTa (optional)
    if (FEAT_DIR / "deberta_preds.csv").exists():
        d = aligned("deberta_preds.csv")
        results["deberta"] = (d["deberta"].to_numpy()[is_tr], d["deberta"].to_numpy()[is_te],
                              d["deberta_insample"].to_numpy()[is_tr])
    else:
        print("deberta_preds.csv not found - ensemble runs without DeBERTa")

    # ------------------------------------------------------------------ stacking
    names = list(results)
    P_oof = np.column_stack([results[n][0] for n in names])
    P_test = np.column_stack([results[n][1] for n in names])
    P_ins = np.column_stack([results[n][2] for n in names])
    stacker = LinearRegression(positive=True)
    stack_oof = np.clip(cv.oof(stacker, P_oof), 0, 5)  # nested CV of the stacker itself
    stacker.fit(P_oof, y)
    final_test = np.clip(stacker.predict(P_test), 0, 5)
    final_ins = np.clip(stacker.predict(P_ins), 0, 5)

    # ------------------------------------------------------------------ report
    rows = [{"model": n, "cv_rmse": rmse(results[n][0], y), "cv_pearson": pearson(results[n][0], y),
             "stack_weight": float(w)} for n, w in zip(names, stacker.coef_)]
    rows.append({"model": "ENSEMBLE", "cv_rmse": rmse(stack_oof, y), "cv_pearson": pearson(stack_oof, y),
                 "stack_weight": float(stacker.intercept_)})
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("(stack_weight of ENSEMBLE row = stacker intercept)")

    nz = y > 0  # same subset in both variants -> makes --drop-zero runs comparable
    report = {
        "train_rmse_insample": rmse(final_ins, y), "train_pearson_insample": pearson(final_ins, y),
        "cv_rmse": rmse(stack_oof, y), "cv_pearson": pearson(stack_oof, y),
        "cv_rmse_nonzero_labels": rmse(stack_oof[nz], y[nz]),
        "baseline_rmse_predict_mean": rmse(np.full(len(y), y.mean()), y),
        "drop_zero": args.drop_zero, "wavlm_layers": layers, "models": rows,
    }
    print(f"\nTRAINING RMSE (fit on all train, predict train): {report['train_rmse_insample']:.4f}  "
          f"Pearson {report['train_pearson_insample']:.4f}")
    print(f"5-FOLD CV RMSE (honest estimate for test)      : {report['cv_rmse']:.4f}  "
          f"Pearson {report['cv_pearson']:.4f}")
    print(f"CV RMSE on clips with label > 0 (comparable)  : {report['cv_rmse_nonzero_labels']:.4f}")
    print(f"baseline (always predict mean) RMSE           : {report['baseline_rmse_predict_mean']:.4f}")

    # ------------------------------------------------------------------ files
    tag = "_dropzero" if args.drop_zero else ""
    (OUT_DIR / f"cv_report{tag}.json").write_text(json.dumps(report, indent=2))
    sub = pd.DataFrame({"filename": te["filename"], "label": final_test})
    assert len(sub) == 216 and sub["label"].between(0, 5).all()
    sub.to_csv(OUT_DIR / f"submission{tag}.csv", index=False)
    oof_df = pd.DataFrame({"key": tr["key"], "label": y, "ensemble": stack_oof,
                           **{n: results[n][0] for n in names}})
    oof_df.to_csv(OUT_DIR / f"oof_predictions{tag}.csv", index=False)

    make_plots(y, stack_oof, final_test, table, layer_rmse, layers, lgbm_full, tab_cols)
    print(f"\nsaved outputs/submission{tag}.csv ({len(sub)} rows) + report and plots in outputs/")


def make_plots(y, oof, test_pred, table, layer_rmse, layers, lgbm_full, tab_cols):
    # 1. label distribution vs prediction distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.25, 5.5, 0.5)
    ax.hist(y, bins=bins, alpha=0.6, label="train labels")
    ax.hist(test_pred, bins=bins, alpha=0.6, label="test predictions")
    ax.set(xlabel="grammar score", ylabel="count", title="Label vs prediction distribution")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "distribution.png", dpi=120); plt.close(fig)

    # 2. OOF predicted vs actual
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(y))
    ax.scatter(y + jitter, oof, s=10, alpha=0.5)
    ax.plot([0, 5], [0, 5], "k--", lw=1)
    ax.set(xlabel="true score", ylabel="predicted (out-of-fold)", xlim=(-0.3, 5.3), ylim=(-0.3, 5.3),
           title=f"OOF: RMSE {rmse(oof, y):.3f}, r {pearson(oof, y):.3f}")
    fig.tight_layout(); fig.savefig(OUT_DIR / "pred_vs_actual.png", dpi=120); plt.close(fig)

    # 3. residuals per true label
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = np.sort(np.unique(y))
    ax.boxplot([oof[y == v] - v for v in levels])
    ax.set_xticks(range(1, len(levels) + 1), [f"{v:g}" for v in levels])
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="true score", ylabel="prediction - truth", title="OOF residuals by true score")
    fig.tight_layout(); fig.savefig(OUT_DIR / "residuals_by_label.png", dpi=120); plt.close(fig)

    # 4. model comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table["model"], table["cv_rmse"])
    for i, v in enumerate(table["cv_rmse"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set(xlabel="5-fold CV RMSE (lower is better)", title="Base models vs ensemble")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT_DIR / "model_scores.png", dpi=120); plt.close(fig)

    # 5. WavLM layer-wise CV RMSE
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(layer_rmse)), layer_rmse, marker="o")
    ax.axvspan(min(layers) - 0.4, max(layers) + 0.4, alpha=0.15)
    ax.set(xlabel="WavLM layer", ylabel="ridge CV RMSE", title="Which WavLM layer predicts grammar best")
    fig.tight_layout(); fig.savefig(OUT_DIR / "wavlm_layers.png", dpi=120); plt.close(fig)

    # 6. LightGBM feature importance
    imp = pd.Series(lgbm_full.booster_.feature_importance("gain"), index=tab_cols).nlargest(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index, imp.values)
    ax.set(xlabel="total gain", title="Top-20 hand-crafted features (LightGBM)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "feature_importance.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
