# flow.md - How the project was built (read top to bottom)

## Status
- [x] Questions answered, scope fixed (see demand.md): you download data, venv, core only, human study skipped
- [x] Folder skeleton + `.venv` + pinned `requirements.txt`
- [x] Week 1 - cleaning (Deliverable 1, part 1)
- [x] Week 2 - graphs + analysis (Deliverable 1 done)
- [x] Week 3 - models (Deliverable 2 done, minus human study)
- [x] Week 4 - generator + Streamlit app (code done and smoke-tested; **live URL is still for you to deploy**)
- [x] Week 5 - README, final report (demo video / slides: yours, out of scope)

## Data flow (one line)
`train.json` + `flavor_network/` -> **src/cleaning.py** -> `recipes.csv`, `ingredient_compounds.csv` -> **src/graphs.py** (notebook 02) -> `cooc_edges.csv`, `communities.csv`, figures -> **src/train_models.py** -> `models/` -> **src/substitute.py**, **src/generate.py** -> **app/streamlit_app.py**

## Step by step

### Week 1 - Cleaning (`src/cleaning.py`, `notebooks/01_cleaning.ipynb`)
1. Clean each ingredient string: drop parentheses/digits/units/prep words/brands, lowercase, lemmatize ("tomatoes" -> "tomato").
2. Fuzzy-merge near duplicates (rapidfuzz >= 94) so typos/variants collapse.
3. Map to flavor compounds: alias table -> exact -> suffix -> contained phrase -> fuzzy.
4. Result: 39,752 recipes, 5,565 ingredients; **76.8% of occurrences matched** (52.6% of unique). Not matched: rare names and staples missing from the data (salt, water, oil). Farnesol removed (README issue).
5. Problem hit: first version of the "contains" matcher was too slow (>10 min); replaced regex over all names by n-gram set lookups (30 s).

### Week 2 - Graphs and analysis (`src/graphs.py`, `notebooks/02_graph_analysis.ipynb`)
- Co-occurrence graph (PMI, 1,227 nodes / 44k edges) and flavor-sharing graph (shared compounds; deduplicated to 394 flavor entries - first version let "ground beef", "beef roast"... all count separately and dominated the hubs).
- Analyses: food-pairing test (real vs. random), Louvain (5 communities), cuisine similarity + dendrogram, centrality.
- Visuals in `reports/`: `ingredient_network.html` (PyVis), `cuisine_ingredient_heatmap.html`, `ingredient_umap.html`, `food_pairing.html`, `cuisine_dendrogram.png`...
- Key result: East Asian cuisines pair *contrasting* flavors, Italian/Mexican/British pair *similar* ones (matches the original paper).

### Week 3 - Models (`src/train_models.py`, `src/substitute.py`, `notebooks/03_models.ipynb`)
- Classifier: TF-IDF + LogReg baseline (0.774 acc), LinearSVC (best, 0.782 acc / 0.701 macro-F1), XGBoost (0.758, slowest). Stratified 70/15/15 split, chosen on validation, reported on test.
- Substitution: Word2Vec on recipes. Plain neighbours returned *companions* (basil -> mozzarella), so I added a category filter, a compound-overlap re-rank, a PMI penalty, variant removal and a rarity cut-off -> sensible swaps (butter -> milk/cream, honey -> sugar/maple syrup).
- Evaluation: hide-one-ingredient test (hit@10 14.9% vs 0.4% random). Human rating study skipped by choice.

### Week 4 - Generator and app (`src/generate.py`, `app/streamlit_app.py`)
- **Compatibility score** = 0.6 x PMI term + 0.4 x shared-compound term.
- **Generator**: beam search; each step adds the ingredient maximising `lambda*compat + (1-lambda)*P(cuisine)`.
- **App tabs**: Predict cuisine (probability bars), Substitute (pick a recipe + ingredient), Generate (cuisine + seeds -> ranked ideas + embedded graph view).
- Swapped deprecated Streamlit calls (`use_container_width`, `components.html`) for the current ones; smoke-tested with `streamlit.testing`.

### Week 5 - Docs
README (setup, sources, run order, deploy steps), `reports/final_report.md`, pinned requirements, notebooks executed top to bottom.

## What is left for you
1. Run the app locally: `.venv\Scripts\activate` then `streamlit run app/streamlit_app.py`.
2. Deploy (steps in README), test with someone new, record the 2-3 min demo.
3. Optional: human rating study, minority-class rebalancing, neural net / Node2Vec.
