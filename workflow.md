Tech stack
Language and tools: Python, Jupyter, pandas, scikit-learn, gensim
Graph work: NetworkX, node2vec, PyVis
Visualisation: Plotly or matplotlib, UMAP
App: Streamlit, deployed on Streamlit Community Cloud or Hugging Face Spaces (both free)
Collaboration: one GitHub repo with folders data/, notebooks/, src/, app/
Week 1: Data and cleaning (Deliverable 1, part 1)
Collect data.
Recipes: Kaggle “What’s Cooking” (Yummly, about 40k recipes with cuisine labels). Optionally add Food.com for more variety.
Flavor compounds: FlavorDB, or the flavor network dataset from Ahn et al. (2011).
Clean ingredient names.
Strip quantities, units and prep words (“chopped”, “fresh”).
Lowercase, then lemmatize.
Merge near-duplicates with fuzzy matching (rapidfuzz).
Map ingredients to compounds. Match your cleaned names to FlavorDB entries and report the match rate. Drop or manually fix the unmatched ones.
Exploratory analysis. Look at recipes per cuisine, the most common ingredients, and class imbalance.
Milestone: a clean recipes.csv and ingredient_compounds.csv.
Week 2: Graph and analysis (Deliverable 1, part 2)
Build two weighted graphs.
Co-occurrence graph: ingredients are linked by how often they appear in the same recipe, normalized with PMI.
Flavor-sharing graph: ingredients are linked by the number of shared compounds.
Run the analyses.
Food-pairing test: compare the average shared compounds in real recipes against randomized recipes, per cuisine.
Community detection with Louvain.
Cuisine similarity: ingredient-frequency vectors, cosine similarity, hierarchical clustering.
Centrality: find the “hub” ingredients.
Visualisations:
Interactive PyVis network
Cuisine-by-ingredient heatmap
UMAP map of ingredients coloured by community
Milestone: an analysis notebook with a short written findings section. Deliverable 1 is done.
Week 3: Models (Deliverable 2)
Cuisine classifier.
Baseline: TF-IDF + logistic regression.
Then try LinearSVC and XGBoost/LightGBM.
Optionally add a small neural net or fine-tuned embedding model.
Use a stratified train/val/test split and report accuracy, macro-F1 and a confusion matrix.
Ingredient embeddings for substitution.
Train Word2Vec on recipes, treating each recipe as a sentence. Optionally compare with Node2Vec on your graph.
Substitution = nearest neighbours by cosine similarity.
Improve it by filtering to the same ingredient category (protein, dairy, spice) and by re-ranking with flavor-compound overlap.
Evaluate substitution. There’s no ground truth, so do this:
Run a small human study: 20 queries, top-3 suggestions each, rated 1-5 by 5+ classmates.
Optionally add an automatic check by hiding one ingredient from a recipe and measuring how often the model recovers it.
Milestone: saved models (joblib/gensim) plus an evaluation notebook. Deliverable 2 is done.
Week 4: Recipe generation and the app (Deliverable 3)
Flavor-compatibility score. Define a function that scores an ingredient set using average PMI between pairs plus shared-compound overlap. You built both ingredients in Weeks 2-3.
Generator (keep it simple and reliable).
Start from a seed ingredient set and a target cuisine.
Greedily or with beam search, add the ingredient that most improves the compatibility score and the classifier’s confidence for that cuisine.
Optionally compare against a recipe-title generator from a small language model.
Build the Streamlit app with three tabs:
Predict cuisine: enter ingredients and see the cuisine with confidence bars.
Substitute: pick a recipe and an ingredient and get ranked alternatives with similarity scores.
Generate: choose a cuisine and seed ingredients and see ideas with their compatibility score. Add an embedded graph view from Deliverable 1.
Deploy and test it with someone who hasn’t seen it.
Milestone: a live URL. Deliverable 3 is done.
Week 5: Polish and submission
Write the README (setup, data sources, how to run).
Write the final report: problem, methods, results, limitations.
Record a 2-3 minute demo video.
Clean up the repo, pin requirements.txt and re-run every notebook top to bottom.
Prepare slides if needed.
Keep 2-3 days of buffer before the deadline.