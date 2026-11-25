"""Ingredient substitution: Word2Vec nearest neighbours, filtered by flavor category and re-ranked
with flavor-compound overlap.  Also the 'hide-one-ingredient' automatic evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]


class Substituter:
    def __init__(self, w2v=None, ic=None, alpha=0.7, beta=0.25):
        self.w2v = w2v or Word2Vec.load(str(ROOT / "models" / "w2v.model"))
        if ic is None:
            ic = pd.read_csv(ROOT / "data" / "processed" / "ingredient_compounds.csv")
            ic["compounds"] = ic["compounds"].apply(lambda s: set(map(int, str(s).split("|"))))
        self.cat = dict(zip(ic["ingredient"], ic["category"]))
        self.comp = dict(zip(ic["ingredient"], ic["compounds"]))
        self.freq = dict(zip(ic["ingredient"], ic["recipe_count"]))
        self.alpha = alpha  # weight of embedding similarity vs. compound overlap
        self.beta = beta    # penalty for ingredients that usually appear WITH the query (they complement, not replace)
        e = pd.read_csv(ROOT / "data" / "processed" / "cooc_edges.csv")
        self.pmi = {}
        for a, b, w in zip(e["source"], e["target"], e["weight"]):
            self.pmi[(a, b)] = self.pmi[(b, a)] = w

    def jaccard(self, a, b):
        ca, cb = self.comp.get(a), self.comp.get(b)
        if not ca or not cb:
            return None
        return len(ca & cb) / len(ca | cb)

    @staticmethod
    def is_variant(a, b):
        """'soy sauce' vs 'light soy sauce', 'basil' vs 'basil leaf': same food, different wording."""
        wa, wb = set(a.split()), set(b.split())
        return wa <= wb or wb <= wa

    def suggest(self, ingredient, k=5, same_category=True, rerank=True, exclude=(), pool=150,
                drop_variants=True, min_recipes=40, _relaxed=False):
        """Top-k substitutes. Returns list of (ingredient, final_score, cosine, flavor_overlap).
        same_category: candidate must have the SAME flavor category as the query (unknown category = dropped).
        rerank: score = alpha*cos + (1-alpha)*compound_jaccard - beta*clip(PMI(query, cand)/2, 0, 1)."""
        if ingredient not in self.w2v.wv:
            return []
        cands = self.w2v.wv.most_similar(ingredient, topn=pool)
        q_cat = self.cat.get(ingredient)
        out = []
        for name, cos in cands:
            if name in exclude:
                continue
            if drop_variants and self.is_variant(ingredient, name):
                continue
            if min_recipes and self.freq.get(name, 0) < min_recipes:
                continue  # rare ingredients have noisy vectors
            if same_category and q_cat is not None and self.cat.get(name) != q_cat:
                continue
            ov = self.jaccard(ingredient, name)
            score = cos
            if rerank:
                pmi = min(max(self.pmi.get((ingredient, name), 0.0) / 2, 0), 1)
                score = self.alpha * cos + (1 - self.alpha) * (ov or 0.0) - self.beta * pmi
            out.append((name, float(score), float(cos), ov))
        out.sort(key=lambda t: -t[1])
        if len(out) < min(3, k) and same_category and not _relaxed:  # category too strict -> relax once
            return self.suggest(ingredient, k, False, rerank, exclude, pool, drop_variants, min_recipes, True)
        return out[:k]

    def hide_one_test(self, recipes, k=10, n=2000, seed=0, **kw):
        """Hide one random ingredient from a recipe; score the model on how well it recovers it using the
        other ingredients as context (mean context vector).  Returns hit@1/5/10 and MRR."""
        rng = np.random.default_rng(seed)
        wv = self.w2v.wv
        hits = {1: 0, 5: 0, 10: 0}; rr = []; done = 0
        for ings in recipes:
            ings = [i for i in ings if i in wv]
            if len(ings) < 3:
                continue
            target = ings[rng.integers(len(ings))]
            ctx = [i for i in ings if i != target]
            res = wv.most_similar(positive=ctx, topn=50)
            ranked = [r[0] for r in res]
            rank = ranked.index(target) + 1 if target in ranked else None
            for kk in hits:
                hits[kk] += int(rank is not None and rank <= kk)
            rr.append(1 / rank if rank else 0)
            done += 1
            if done >= n:
                break
        return {f"hit@{kk}": v / done for kk, v in hits.items()} | {"MRR": float(np.mean(rr)), "n": done}
