#!/usr/bin/env python
"""R1.5 baseline: a TRAINED query-type classifier (type-less vs genuine-product-type), the comparison
the reviewer asked for against the paper's deliberately-weak rule-based head-noun detector
(recall 1.00 on concrete queries, precision 0.56). If a small learned gate beats 0.56 precision, a
deployable query-type gate is feasible, strengthening the mitigation story.

Setup ($0, local): embed queries with a sentence-transformers bi-encoder, LogisticRegression.
 - TRAIN on labeled REAL queries: WANDS auto_reference (typed vs type-less) + ESCI queries
   (gpt-4o-mini labels from typeless_prevalence; ambiguous dropped).
 - CROSS-VALIDATE (stratified 5-fold) on the training pool.
 - HELD-OUT TEST on the human-adjudicated 150 probe motifs (ground truth), binary type-less vs not.
Report precision/recall/F1 for BOTH the type-less class and the concrete/typed class (the latter is
the paper's 0.56-precision comparison). Seed 13.

Run: ./.venv/bin/python scripts/query_type_classifier.py
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")


def load_training():
    X, y = [], []  # y=1 type-less, y=0 typed
    for r in map(json.loads, open(f"{DATA}/gold/auto_reference.jsonl")):
        pt = (r.get("explicit") or {}).get("product_type")
        X.append(r["query"]); y.append(0 if pt else 1)
    prevf = f"{RES}/prevalence/esci_query_labels.jsonl"
    if os.path.exists(prevf):
        for r in map(json.loads, open(prevf)):
            lab = r.get("label")
            if lab == "type-less":
                X.append(r["query"]); y.append(1)
            elif lab == "genuine-product-type":
                X.append(r["query"]); y.append(0)
    return X, np.array(y)


def load_heldout():
    """human-adjudicated 150 motifs -> binary type-less(1) vs not(0); drop ambiguous for a clean test."""
    X, y = [], []
    for r in csv.DictReader(open(f"{RES}/r1_1/ADJUDICATION_SAMPLE.csv")):
        h = r["human_label"].strip().lower()
        if h == "type-less":
            X.append(r["motif"]); y.append(1)
        elif h == "genuine-product-type":
            X.append(r["motif"]); y.append(0)
    return X, np.array(y)


def prf(y_true, y_pred, pos):
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    Xtr_q, ytr = load_training()
    Xte_q, yte = load_heldout()
    print(f"train: {len(ytr)} real queries ({int(ytr.sum())} type-less / {int((1-ytr).sum())} typed); "
          f"held-out: {len(yte)} human-labeled motifs ({int(yte.sum())} type-less)", flush=True)
    Xtr = enc.encode(Xtr_q, show_progress_bar=False, normalize_embeddings=True)
    Xte = enc.encode(Xte_q, show_progress_bar=False, normalize_embeddings=True)

    # 5-fold CV on the training pool
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    cv_pred = np.zeros_like(ytr)
    for tr, va in skf.split(Xtr, ytr):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr[tr], ytr[tr])
        cv_pred[va] = clf.predict(Xtr[va])
    # held-out: train on all training data
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Xtr, ytr)
    te_pred = clf.predict(Xte)

    out = {"design": "trained query-type classifier (MiniLM embeddings + LogisticRegression, "
           "class_weight balanced, seed 13); vs the paper's head-noun rule (concrete recall 1.00, "
           "precision 0.56).",
           "n_train": len(ytr), "n_heldout": len(yte),
           "cv_5fold": {
               "typeless_detection": dict(zip(["precision", "recall", "f1"], prf(ytr, cv_pred, 1))),
               "concrete_detection": dict(zip(["precision", "recall", "f1"], prf(ytr, cv_pred, 0)))},
           "heldout_human_labels": {
               "typeless_detection": dict(zip(["precision", "recall", "f1"], prf(yte, te_pred, 1))),
               "concrete_detection": dict(zip(["precision", "recall", "f1"], prf(yte, te_pred, 0)))},
           "paper_headnoun_baseline": {"concrete_detection": {"recall": 1.00, "precision": 0.56}}}
    json.dump(out, open(f"{RES}/query_type_classifier.json", "w"), indent=2)
    print("\n=== trained query-type classifier ===")
    for split in ["cv_5fold", "heldout_human_labels"]:
        d = out[split]
        print(f"  {split}:")
        for cls in ["typeless_detection", "concrete_detection"]:
            m = d[cls]; print(f"    {cls:<20} P={m['precision']} R={m['recall']} F1={m['f1']}")
    print(f"  paper head-noun rule: concrete P=0.56 R=1.00")
    ci = out["cv_5fold"]["concrete_detection"]
    print(f"\n  => learned gate concrete-precision {ci['precision']} vs rule 0.56 "
          f"({'beats' if ci['precision'] > 0.56 else 'does not beat'} the weak baseline)")
    print("wrote", f"{RES}/query_type_classifier.json")


if __name__ == "__main__":
    main()
