# FlavorGraph - Final Report

## 1. Problem
Can we learn the "grammar" of cuisines from recipes? We build a knowledge graph linking recipes, ingredients, cuisines and flavor compounds, test whether cuisines follow the *food-pairing hypothesis* (ingredients that share flavor compounds taste good together), and use the graph and ML models to (a) predict a recipe's cuisine, (b) suggest ingredient substitutions and (c) generate novel ingredient combinations scored for flavor compatibility.

## 2. Data
- **Recipes:** Kaggle What's Cooking - 39,774 recipes, 20 cuisines, 6,714 raw ingredient strings.
- **Flavor compounds:** Ahn et al. (2011) flavor network - 1,530 ingredients, 1,107 compounds, 36,781 links. (Used instead of FlavorDB: free, bulk-downloadable.)
- **Cleaning** (`src/cleaning.py`): remove parentheses, digits, units, prep words and brand names; lowercase; lemmatize (WordNet); merge near-duplicates with rapidfuzz (ratio >= 94, guarded so "red onion" and "green onion" are not merged). Result: 5,565 distinct ingredients, 39,752 recipes (22 dropped for < 2 ingredients).
- **Mapping to compounds:** alias table -> exact -> suffix ("black olive" -> "olive") -> contained phrase -> fuzzy (>= 90). Match rate **52.6% of unique names, 76.8% of occurrences**. Farnesol removed per dataset README.

## 3. Methods
**Graphs.** (1) Co-occurrence graph: ingredients in >= 25 recipes (1,227 nodes), edge weight PMI, kept if PMI > 0 and pair seen >= 5 times (44,078 edges). (2) Flavor-sharing graph: one node per flavor entry (394), edge weight = number of shared compounds.
**Analysis.** Food-pairing test (mean shared compounds per recipe vs. random recipes of equal size, ingredients drawn by global frequency, 1,500 recipes per cuisine, bootstrap CI); Louvain communities; cuisine similarity (ingredient-frequency vectors, cosine, average-linkage clustering); betweenness / degree centrality; PyVis network, cuisine x ingredient heatmap, UMAP of ingredients.
**Cuisine classifier.** Each ingredient is one token; TF-IDF (sublinear, min_df 2) + Logistic Regression (baseline), LinearSVC, XGBoost. Stratified 70/15/15 split, seed 42; model choice on validation, results reported on test.
**Substitution.** Skip-gram Word2Vec (100-d, window 60 because ingredient sets are unordered, 30 epochs, recipes shuffled each epoch; train split only). Suggestions = nearest neighbours, then: same flavor category, drop name variants and rare ingredients, re-rank by 0.7 cos + 0.3 compound-Jaccard - 0.25 PMI(query, candidate).
**Compatibility score and generator** (`src/generate.py`). compat(S) = 0.6 x mean-PMI term (PMI/2 clipped to [0,1]) + 0.4 x mean compound Jaccard. Beam search (width 4) adds ingredients to the seed set from the 350 best-connected ingredients, maximising lambda x compat + (1 - lambda) x P(target cuisine) with the LogReg classifier. The weights (0.6/0.4, PMI cap 2, lambda 0.5 default) are hand-chosen, not tuned.

## 4. Results
**Food pairing.** Contrasting (real < random, CI excludes 0): Korean -3.7, Japanese -3.3, Chinese -3.2, Moroccan -2.9, Vietnamese -2.5, Indian -2.4, Greek -1.7, Thai -1.1. Uniform: Brazilian +5.7, Italian +3.3, Mexican +2.8, Irish +2.8, British +2.6, Southern US +1.8, French +0.9, Cajun/Creole +0.6. This matches the qualitative finding of Ahn et al.
**Communities.** 5 Louvain communities: South Asian, East/Southeast Asian, Latin/Tex-Mex, Mediterranean/European savoury, baking/dessert.
**Cuisine similarity.** Closest: British-Irish 0.94, Southern US-British 0.94, Thai-Vietnamese 0.89. Dendrogram: East/Southeast Asian; Moroccan-Indian-Mexican; Mediterranean; Anglo/Northern European.
**Classifier (test, n = 5,963).**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LogReg (baseline) | 0.774 | 0.693 |
| TF-IDF + LinearSVC | **0.782** | **0.701** |
| TF-IDF + XGBoost | 0.758 | 0.676 |

Strong: Mexican (F1 0.90), Indian (0.87), Italian (0.85). Weak: Irish (0.48), Russian (0.50), Spanish (0.49), British (0.52) - small classes (recall only 0.39-0.43). Main confusions: Vietnamese->Thai, Spanish/Greek/French->Italian, British/Irish->Southern US/French.
**Substitution.** Same-category rate of top-5: 0.46 plain -> 0.85 with filter. Examples: butter -> milk / sour cream / heavy cream; honey -> sugar / maple syrup; shrimp -> prawn / clam; basil -> rosemary / parsley. Hide-one test (3,000 test recipes): hit@1/5/10 = 2.6% / 10.1% / 14.9%, MRR 0.066 (random: 0.04% / 0.21% / 0.42%).
**Generator.** Seeds chicken + garlic: Italian -> lasagna noodle, mozzarella, ricotta, parmesan (compat 0.49, P(Italian) ~1.0); Indian -> garam masala, garlic paste, red chili powder, yoghurt (0.61, 0.98). Pork + ginger, Chinese -> light/dark soy sauce, Shaoxing wine, five-spice (0.66).
**App.** Streamlit, three tabs (Predict / Substitute / Generate with embedded graph view). Smoke-tested headlessly with `streamlit.testing`; **not yet deployed or user-tested** (needs the user's GitHub / Streamlit account).

## 5. Limitations
- Compound data covers only ~77% of ingredient occurrences; head-noun matches are approximate ("fish sauce" -> "fish"); several recipe names share one flavor entry, which inflates some overlaps. Cuisine ranking in the pairing test is robust, absolute values less so.
- Salt, water, oil and similar staples have no compounds and so carry no flavor signal.
- Class imbalance (16.8x) lowers macro-F1; no resampling/class weights tried. No neural network or Node2Vec comparison (out of agreed scope).
- Substitution quality has **no human evaluation** (study skipped by decision); evidence is examples plus automatic proxies. Word2Vec still mixes substitutes with companions despite the PMI penalty; flavor categories are coarse.
- Generator: beam search returns near-duplicate sets and variants of one ingredient (light + dark soy sauce); it optimises a hand-made score, not real taste, and the compatibility weights are untuned. Generated "recipes" are ingredient sets, not instructions.
- Cuisine labels are one per recipe and Yummly-biased toward Western home cooking.
