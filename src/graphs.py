"""Week 2: build the two ingredient graphs and the analysis helpers.

Reusable by the notebook, the models and the app.
  - load_data()            recipes + ingredient->compound table
  - build_cooc_graph()     co-occurrence graph, edge weight = PPMI
  - build_flavor_graph()   flavor-sharing graph, edge weight = # shared compounds
  - food_pairing_test()    real vs randomised recipes, per cuisine
  - communities(), cuisine_similarity(), centrality()
"""
import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def load_data():
    rec = pd.read_csv(PROC / "recipes.csv")
    rec["ingredients"] = rec["ingredients"].str.split("|")
    ic = pd.read_csv(PROC / "ingredient_compounds.csv")
    ic["compounds"] = ic["compounds"].apply(lambda s: set(map(int, str(s).split("|"))))
    return rec, ic


def build_cooc_graph(rec, min_recipes=25, min_pair=5):
    """Nodes: ingredients in >= min_recipes recipes. Edge weight = PMI (kept if > 0 and pair seen >= min_pair)."""
    n = len(rec)
    cnt = Counter(i for ings in rec["ingredients"] for i in ings)
    keep = {i for i, c in cnt.items() if c >= min_recipes}
    pair = Counter()
    for ings in rec["ingredients"]:
        ings = sorted(i for i in ings if i in keep)
        pair.update(itertools.combinations(ings, 2))
    G = nx.Graph()
    for i in keep:
        G.add_node(i, count=cnt[i])
    for (a, b), c in pair.items():
        if c < min_pair:
            continue
        pmi = np.log((c / n) / ((cnt[a] / n) * (cnt[b] / n)))
        if pmi > 0:
            G.add_edge(a, b, weight=float(pmi), count=c)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def build_flavor_graph(ic, min_shared=1):
    """Nodes: ingredients matched to compounds. Edge weight = number of shared flavor compounds."""
    # several recipe names can map to the same flavor entry -- that's fine, they share all compounds
    d = dict(zip(ic["ingredient"], ic["compounds"]))
    G = nx.Graph()
    G.add_nodes_from(d)
    names = list(d)
    # invert: compound -> ingredients, then count co-membership (fast vs. all pairs of sets)
    inv = defaultdict(list)
    for ing, cs in d.items():
        for c in cs:
            inv[c].append(ing)
    shared = Counter()
    for members in inv.values():
        for a, b in itertools.combinations(sorted(members), 2):
            shared[(a, b)] += 1
    for (a, b), w in shared.items():
        if w >= min_shared:
            G.add_edge(a, b, weight=w)
    return G


def mean_shared(ings, comp):
    """Average number of shared compounds over all ingredient pairs in one recipe (Ahn et al. N_s)."""
    ings = [i for i in ings if i in comp]
    if len(ings) < 2:
        return None
    pairs = list(itertools.combinations(ings, 2))
    return sum(len(comp[a] & comp[b]) for a, b in pairs) / len(pairs)


def food_pairing_test(rec, ic, n_sample=1500, n_random=1, seed=0):
    """Per cuisine: mean Ns of real recipes vs. recipes whose ingredients are replaced by random ones
    drawn from the global matched-ingredient frequency distribution (same recipe size).
    delta = real - random.  delta > 0 -> uniform food pairing (similar flavors); < 0 -> contrasting."""
    rng = random.Random(seed)
    comp = dict(zip(ic["ingredient"], ic["compounds"]))
    pool = [i for ings in rec["ingredients"] for i in ings if i in comp]
    rows = []
    for cuisine, g in rec.groupby("cuisine"):
        g = g.sample(min(n_sample, len(g)), random_state=seed)
        real, rand = [], []
        for ings in g["ingredients"]:
            m = [i for i in ings if i in comp]
            r = mean_shared(m, comp)
            if r is None:
                continue
            real.append(r)
            for _ in range(n_random):
                rand.append(mean_shared(rng.sample(pool, len(m)), comp) or 0)
        real, rand = np.array(real), np.array(rand)
        # simple bootstrap CI on the difference
        d = []
        for _ in range(200):
            d.append(rng.choice(real.tolist()) * 0 + np.mean(rng.choices(real.tolist(), k=len(real))) -
                     np.mean(rng.choices(rand.tolist(), k=len(rand))))
        rows.append({"cuisine": cuisine, "n": len(real), "real": real.mean(), "random": rand.mean(),
                     "delta": real.mean() - rand.mean(),
                     "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)})
    return pd.DataFrame(rows).sort_values("delta")


def communities(G, seed=42):
    import community as community_louvain  # python-louvain
    part = community_louvain.best_partition(G, weight="weight", random_state=seed)
    return part


def cuisine_vectors(rec, top_k=None):
    """Cuisine x ingredient matrix of the share of each cuisine's recipes containing the ingredient."""
    cnt = defaultdict(Counter)
    size = Counter()
    for c, ings in zip(rec["cuisine"], rec["ingredients"]):
        size[c] += 1
        cnt[c].update(ings)
    df = pd.DataFrame(cnt).fillna(0).T
    df = df.div(pd.Series(size), axis=0)
    return df


def cuisine_similarity(rec):
    from sklearn.metrics.pairwise import cosine_similarity
    M = cuisine_vectors(rec)
    S = pd.DataFrame(cosine_similarity(M.values), index=M.index, columns=M.index)
    return M, S


def centrality(G_cooc, G_flavor):
    deg = pd.Series(dict(G_cooc.degree(weight="weight")), name="cooc_weighted_degree")
    k = min(300, G_cooc.number_of_nodes())
    btw = pd.Series(nx.betweenness_centrality(G_cooc, k=k, weight=None, seed=1), name="cooc_betweenness")
    fdeg = pd.Series(dict(G_flavor.degree(weight="weight")), name="flavor_weighted_degree")
    return pd.concat([deg, btw, fdeg], axis=1)
