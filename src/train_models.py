"""Week 3: train + save the cuisine classifiers and the Word2Vec ingredient embeddings.

Run: python src/train_models.py       (about a few minutes on CPU)
Saves into models/: tfidf_logreg.joblib, linear_svc.joblib, xgboost.joblib, label_encoder.joblib,
                    w2v.model, split.joblib (ids per split so evaluation is reproducible)
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42


def to_doc(ings):
    """Each ingredient becomes ONE token (spaces -> underscores) so 'soy sauce' stays together;
    unigrams + bigrams of tokens are NOT used: ingredient sets have no order."""
    return " ".join(i.replace(" ", "_") for i in ings)


def make_vectorizer():
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=2)


def load_split():
    rec = pd.read_csv(ROOT / "data" / "processed" / "recipes.csv")
    rec["ings"] = rec["ingredients"].str.split("|")
    rec["doc"] = rec["ings"].apply(to_doc)
    le = LabelEncoder().fit(rec["cuisine"])
    rec["y"] = le.transform(rec["cuisine"])
    # stratified 70 / 15 / 15
    tr, tmp = train_test_split(rec.index, test_size=0.30, stratify=rec["y"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=rec.loc[tmp, "y"], random_state=SEED)
    return rec, le, tr, va, te


def main():
    rec, le, tr, va, te = load_split()
    joblib.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}, MODELS / "split.joblib")
    joblib.dump(le, MODELS / "label_encoder.joblib")
    Xtr, ytr = rec.loc[tr, "doc"], rec.loc[tr, "y"]
    Xva, yva = rec.loc[va, "doc"], rec.loc[va, "y"]

    # 1) baseline: TF-IDF + logistic regression
    t = time.time()
    lr = make_pipeline(make_vectorizer(), LogisticRegression(C=10, max_iter=2000))
    lr.fit(Xtr, ytr)
    print("logreg   val acc %.4f  (%.0fs)" % (lr.score(Xva, yva), time.time() - t))
    joblib.dump(lr, MODELS / "tfidf_logreg.joblib")

    # 2) LinearSVC
    t = time.time()
    svc = make_pipeline(make_vectorizer(), LinearSVC(C=0.5))
    svc.fit(Xtr, ytr)
    print("linsvc   val acc %.4f  (%.0fs)" % (svc.score(Xva, yva), time.time() - t))
    joblib.dump(svc, MODELS / "linear_svc.joblib")

    # 3) XGBoost on the TF-IDF matrix
    from xgboost import XGBClassifier
    t = time.time()
    xgb = make_pipeline(make_vectorizer(),
                        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.15, subsample=0.8,
                                      colsample_bytree=0.5, tree_method="hist", n_jobs=-1, random_state=SEED))
    xgb.fit(Xtr, ytr)
    print("xgboost  val acc %.4f  (%.0fs)" % (xgb.score(Xva, yva), time.time() - t))
    joblib.dump(xgb, MODELS / "xgboost.joblib")

    # 4) Word2Vec: each recipe is a "sentence"; order is meaningless, so use a window larger than any recipe
    from gensim.models import Word2Vec
    import random
    rng = random.Random(SEED)
    sents = [list(s) for s in rec.loc[tr, "ings"]]  # train recipes only (no test leakage for the hide-one test)
    class Shuffled:
        def __iter__(self):
            for s in sents:
                s2 = s[:]; rng.shuffle(s2); yield s2
    w2v = Word2Vec(sentences=Shuffled(), vector_size=100, window=60, min_count=5, sg=1, negative=10,
                   epochs=30, workers=4, seed=SEED)
    w2v.save(str(MODELS / "w2v.model"))
    print("word2vec vocab", len(w2v.wv))


if __name__ == "__main__":
    main()
