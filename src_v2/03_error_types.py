"""v2 Step 3 - Grammar-error types from the v1 grammar-correction output (features/gec_corrections.csv).

Every word-level edit between a sentence and its CoEdIT correction is classified as a disfluency repair
(false start "the-", immediate repeat) or as a grammar fix by type (article, preposition, auxiliary,
inflection, pronoun, conjunction, other content word). Rates are per 100 words. CPU only, a few seconds.

Output: features/v2/error_types.csv
"""
import re
from difflib import SequenceMatcher

import pandas as pd

from common_v2 import ERRTYPES_FILE, FEAT_DIR, load_meta

TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?-?")  # keeps the trailing hyphen of false starts ("the-")
ARTICLES = {"a", "an", "the"}
PREPS = {"in", "on", "at", "to", "for", "of", "with", "from", "by", "about", "into", "onto", "over", "under", "through",
         "during", "since", "until", "towards", "toward", "around", "between", "among", "across", "after", "before",
         "upon", "within", "without", "like"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had", "do", "does", "did", "will",
       "would", "can", "could", "shall", "should", "may", "might", "must"}
PRON = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
        "our", "their", "this", "that", "these", "those", "there", "which", "who", "what"}
CONJ = {"and", "but", "or", "so", "because", "although", "though", "if", "when", "while", "that"}
TYPES = ["disfl", "article", "prep", "aux", "infl", "pron", "conj", "content_ins", "content_del", "content_rep"]
GRAMMAR = TYPES[1:]


def same_lemma(a, b):
    """Different inflections of the same word (play / plays / played / playing)."""
    if a == b:
        return False
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k >= max(3, min(len(a), len(b)) - 2)


def word_class(w, fallback):
    for name, vocab in (("article", ARTICLES), ("prep", PREPS), ("aux", AUX), ("pron", PRON), ("conj", CONJ)):
        if w in vocab:
            return name
    return fallback


def classify(src, tgt):
    raw = TOK.findall(src.lower())
    s = [w.rstrip("-") for w in raw]
    t = [w.rstrip("-") for w in TOK.findall(tgt.lower())]
    c = dict.fromkeys(TYPES + ["n_ops", "n_tok_changed"], 0)
    for op, i1, i2, j1, j2 in SequenceMatcher(a=s, b=t, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        c["n_ops"] += 1
        c["n_tok_changed"] += max(i2 - i1, j2 - j1)
        if op == "delete":
            for k in range(i1, i2):
                w = s[k]
                nxt = s[i2] if i2 < len(s) else ""
                prv = s[i1 - 1] if i1 > 0 else ""
                repeated = w in (nxt, prv) or (k + 1 < i2 and s[k + 1] == w)
                c["disfl" if raw[k].endswith("-") or repeated else word_class(w, "content_del")] += 1
        elif op == "insert":
            for w in t[j1:j2]:
                c[word_class(w, "content_ins")] += 1
        else:  # replace
            for a, b in zip(s[i1:i2], t[j1:j2]):
                if a in ARTICLES and b in ARTICLES:
                    c["article"] += 1
                elif a in PREPS and b in PREPS:
                    c["prep"] += 1
                elif a in AUX or b in AUX:
                    c["aux"] += 1
                elif same_lemma(a, b):
                    c["infl"] += 1
                elif a in PRON or b in PRON:
                    c["pron"] += 1
                else:
                    c["content_rep"] += 1
            c["content_rep"] += abs((i2 - i1) - (j2 - j1))
    return c


def main():
    meta = load_meta()
    g = pd.read_csv(FEAT_DIR / "gec_corrections.csv")
    e = pd.concat([g[["key", "nw"]].reset_index(drop=True),
                   pd.DataFrame([classify(a, str(b)) for a, b in zip(g["sent"], g["corrected"])])], axis=1)
    agg = e.groupby("key").sum(numeric_only=True)
    nw = agg.pop("nw").clip(lower=1)
    feats = agg.div(nw, axis=0) * 100
    feats.columns = [f"et_{c}_p100" for c in feats.columns]
    feats["et_grammar_p100"] = agg[GRAMMAR].sum(axis=1) / nw * 100
    feats["et_core_p100"] = agg[["article", "prep", "aux", "infl"]].sum(axis=1) / nw * 100
    feats = feats.reindex(meta["key"])  # clips without any sentence stay NaN (imputed later)
    feats.index.name = "key"
    feats.reset_index().to_csv(ERRTYPES_FILE, index=False)
    print(f"saved {ERRTYPES_FILE} {feats.shape}")


if __name__ == "__main__":
    main()
