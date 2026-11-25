# FlavorGraph: Understanding and Generating Recipes with ML

Knowledge graph of recipes, ingredients, cuisines and flavor compounds -> food-pairing analysis -> cuisine classifier + ingredient substitution -> Streamlit app that predicts cuisine, suggests substitutes and generates ingredient combinations ranked by a flavor-compatibility score.

See `demand.md` (what was asked), `flow.md` (how it was built), `reports/final_report.md` (methods and results).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
Python 3.11 was used.

## Data sources (put in `data/raw/`, not committed)
| Data | Source | Place at |
|---|---|---|
| Recipes (What's Cooking, ~40k, cuisine labels) | https://www.kaggle.com/c/whats-cooking/data | `data/raw/train.json` (unzip) |
| Flavor compounds (Ahn et al. 2011) | https://zenodo.org/doi/10.5281/zenodo.11449657 | `data/raw/flavor_network/` (unzip; contains `ingr_comp/`) |

## Run order
```bash
python src/cleaning.py          # -> data/processed/recipes.csv, ingredient_compounds.csv   (~30 s)
# notebooks/02_graph_analysis.ipynb  -> graphs, analysis, reports/*.html   (~5 min)
python src/train_models.py      # -> models/*.joblib, w2v.model                              (~8 min, XGBoost is slow)
# notebooks/03_models.ipynb     -> evaluation (test set), substitution examples
streamlit run app/streamlit_app.py
```
Notebooks can be re-run top to bottom: `jupyter nbconvert --to notebook --execute --inplace notebooks/0X_*.ipynb` (run from `notebooks/`). Order matters: 02 writes `cooc_edges.csv` and `communities.csv` that the app and generator use.

## Layout
```
data/raw/        original downloads            data/processed/  cleaned CSVs, graph edge lists, analysis tables
notebooks/       01_cleaning, 02_graph_analysis, 03_models
src/             cleaning.py  graphs.py  train_models.py  substitute.py  generate.py
app/             streamlit_app.py
models/          saved classifiers + Word2Vec   reports/  interactive figures, report
```

## Headline results
- Food pairing reproduces Ahn et al.: East Asian cuisines pair *contrasting* flavors; Italian, Mexican, British, Irish pair *similar* ones.
- Cuisine classifier (test): LinearSVC accuracy 0.782, macro-F1 0.701 (baseline TF-IDF+LogReg 0.774 / 0.693; XGBoost 0.758 / 0.676).
- Substitution: Word2Vec + category filter + compound/PMI re-ranking; hide-one-ingredient hit@10 = 14.9% (random 0.4%).
- Human rating study for substitutions was **not** performed.

## Deploy (Streamlit Community Cloud)
1. Push the repo to GitHub **including** `data/processed/`, `models/` (all files are small, ~25 MB total) and `reports/ingredient_network.html`; do not push `data/raw/` or `.venv/`.
2. On https://share.streamlit.io choose the repo, main file `app/streamlit_app.py`, Python 3.11.
3. `requirements.txt` is pinned. If the build is slow, remove `jupyter`, `nbconvert`, `ipykernel`, `seaborn`, `node2vec` from the deployed copy (the app does not need them).
4. Test with someone who has not seen the app before (workflow step).
