# demand.md — What the project asks for

**Project:** FlavorGraph: Understanding and Generating Recipes with ML
(Source: `detail.txt` = deliverables, `workflow.md` = plan. Scope chosen: **core only**; human rating study **skipped**.)

## Idea in one paragraph
Build a knowledge graph of recipes, ingredients, cuisines and flavor compounds; study food-pairing patterns; train a cuisine classifier and an ingredient-substitution model; wrap everything in a Streamlit app that also generates novel ingredient combinations scored by flavor compatibility.

## Deliverables
| # | Deliverable | What "done" means |
|---|-------------|-------------------|
| 1 | Knowledge graph + analysis | Clean `recipes.csv` + `ingredient_compounds.csv`; co-occurrence (PMI) graph and flavor-sharing graph; food-pairing test, Louvain communities, cuisine similarity, centrality; PyVis network, heatmap, UMAP; analysis notebook with findings |
| 2 | ML models | Cuisine classifier (TF-IDF+LogReg baseline, LinearSVC, XGBoost) with accuracy, macro-F1, confusion matrix; Word2Vec substitution (same-category filter + flavor-overlap rerank); saved models |
| 3 | Web app | Streamlit, 3 tabs: Predict cuisine / Substitute / Generate (+ graph view). Generator = greedy/beam search on flavor-compatibility score + classifier confidence |
| – | Polish | README, final report, pinned requirements, notebooks re-run top to bottom |

## Data required
- Recipes: Kaggle "What's Cooking" (`train.json`, ~40k recipes with cuisine).
- Flavor compounds: Ahn et al. 2011 flavor network (Zenodo) — user downloads into `data/raw/`.

## Delivery status
All three deliverables are built (see `flow.md`). Not done: human rating study (skipped by choice), live deployment, demo video (need your accounts/time).

## Out of scope (by user's choice)
Human rating study, Food.com, neural net, Node2Vec comparison, LM title generator, slides, demo video, live deployment (deploy steps will be documented only).

## Tech stack
Python 3.11, pandas, scikit-learn, gensim, NetworkX, PyVis, Plotly, UMAP, rapidfuzz, XGBoost, Streamlit. Folders: `data/ notebooks/ src/ app/`.
