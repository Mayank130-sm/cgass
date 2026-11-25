"""Week 4: flavor-compatibility score + beam-search ingredient-set generator.

compat(S) = W_PMI * pmi_term + W_FLAVOR * flavor_term           (both in 0..1)
  pmi_term    = mean PMI over ingredient pairs in S (PMI from the co-occurrence graph, 0 if the pair is
                not an edge), divided by PMI_CAP and clipped to [0, 1]
  flavor_term = mean Jaccard overlap of flavor-compound sets over pairs where both ingredients have compounds
The generator adds one ingredient at a time (beam search), maximising
  LAMBDA * compat(S) + (1 - LAMBDA) * P(target cuisine | S)         (classifier = TF-IDF + LogReg)
"""
import itertools
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W_PMI, W_FLAVOR, PMI_CAP = 0.6, 0.4, 2.0


class FlavorModel:
    def __init__(self):
        e = pd.read_csv(ROOT / "data" / "processed" / "cooc_edges.csv")
        self.pmi = {}
        for a, b, w in zip(e["source"], e["target"], e["weight"]):
            self.pmi[(a, b)] = self.pmi[(b, a)] = w
        ic = pd.read_csv(ROOT / "data" / "processed" / "ingredient_compounds.csv")
        self.comp = {i: set(map(int, str(c).split("|"))) for i, c in zip(ic["ingredient"], ic["compounds"])}
        self.category = dict(zip(ic["ingredient"], ic["category"]))
        self.vocab = sorted({n for pair in self.pmi for n in pair})
        self.degree = {n: 0 for n in self.vocab}
        for a, _b in self.pmi:
            self.degree[a] += 1
        self.clf = joblib.load(ROOT / "models" / "tfidf_logreg.joblib")
        self.le = joblib.load(ROOT / "models" / "label_encoder.joblib")
        self.cuisines = list(self.le.classes_)

    # ---------- scoring ----------
    @lru_cache(maxsize=200000)
    def _jac(self, a, b):
        ca, cb = self.comp.get(a), self.comp.get(b)
        if not ca or not cb:
            return None
        return len(ca & cb) / len(ca | cb)

    def pair_terms(self, a, b):
        return self.pmi.get((a, b), 0.0), self._jac(*sorted((a, b)))

    def compat(self, ings):
        ings = list(ings)
        if len(ings) < 2:
            return 0.0, 0.0, 0.0
        pm, fl = [], []
        for a, b in itertools.combinations(ings, 2):
            p, j = self.pair_terms(a, b)
            pm.append(p)
            if j is not None:
                fl.append(j)
        pmi_term = float(np.clip(np.mean(pm) / PMI_CAP, 0, 1))
        flavor_term = float(np.mean(fl)) if fl else 0.0
        return W_PMI * pmi_term + W_FLAVOR * flavor_term, pmi_term, flavor_term

    # ---------- classifier ----------
    @staticmethod
    def doc(ings):
        return " ".join(i.replace(" ", "_") for i in ings)

    def cuisine_proba(self, ings):
        p = self.clf.predict_proba([self.doc(ings)])[0]
        return dict(zip(self.cuisines, p))

    def cuisine_proba_batch(self, sets, cuisine):
        j = self.cuisines.index(cuisine)
        return self.clf.predict_proba([self.doc(s) for s in sets])[:, j]


def generate(fm, seeds, cuisine, n_add=4, beam=4, top_n=8, lam=0.5, pool_size=350, exclude_cat=()):
    """Beam search. Returns list of dicts sorted by final score."""
    seeds = [s for s in seeds if s in fm.pmi or s in fm.vocab]
    # candidate pool: ingredients that co-occur with at least one seed, plus the globally frequent ones
    cnt = fm.degree
    pool = sorted(fm.vocab, key=lambda n: -cnt[n])[:pool_size]
    pool = [p for p in pool if p not in seeds and fm.category.get(p) not in exclude_cat]

    def final(s, comp, prob):
        return lam * comp + (1 - lam) * prob

    start = tuple(sorted(seeds))
    beams = [(start, 0.0)]
    seen, results = {start}, []
    for _ in range(n_add):
        cands = []
        for s, _sc in beams:
            for c in pool:
                if c in s:
                    continue
                ns = tuple(sorted(s + (c,)))
                if ns in seen:
                    continue
                seen.add(ns)
                cands.append(ns)
        if not cands:
            break
        probs = fm.cuisine_proba_batch(cands, cuisine)
        scored = []
        for ns, pr in zip(cands, probs):
            comp, pm, fl = fm.compat(ns)
            scored.append((final(ns, comp, pr), ns, comp, pm, fl, pr))
        scored.sort(key=lambda t: -t[0])
        beams = [(s[1], s[0]) for s in scored[:beam]]
        results += [dict(ingredients=list(s[1]), final=s[0], compat=s[2], pmi_term=s[3],
                         flavor_term=s[4], p_cuisine=float(s[5]), added=len(s[1]) - len(start)) for s in scored[:beam]]
    results.sort(key=lambda d: -d["final"])
    return results[:top_n]
