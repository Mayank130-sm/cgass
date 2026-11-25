"""FlavorGraph - Streamlit app.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate import FlavorModel, generate  # noqa: E402
from substitute import Substituter  # noqa: E402

st.set_page_config(page_title="FlavorGraph", page_icon="🍲", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load():
    fm = FlavorModel()
    sub = Substituter()
    rec = pd.read_csv(ROOT / "data" / "processed" / "recipes.csv")
    edges = pd.read_csv(ROOT / "data" / "processed" / "cooc_edges.csv")
    comm = pd.read_csv(ROOT / "data" / "processed" / "communities.csv", index_col=0).iloc[:, 0].to_dict()
    return fm, sub, rec, edges, comm


fm, sub, rec, edges, comm = load()
vocab = sorted(set(fm.vocab))
pretty = lambda c: c.replace("_", " ").title()

st.title("🍲 FlavorGraph")
st.caption("Cuisine prediction, ingredient substitution and recipe-idea generation from a recipe / flavor knowledge graph.")

tab1, tab2, tab3 = st.tabs(["Predict cuisine", "Substitute", "Generate"])

# ------------------------------------------------------------------ 1. predict
with tab1:
    st.subheader("Which cuisine is this?")
    ings = st.multiselect("Choose ingredients", vocab, default=["soy sauce", "ginger", "garlic", "sesame oil"],
                          key="pred_ings")
    if len(ings) >= 2:
        probs = pd.Series(fm.cuisine_proba(ings)).sort_values(ascending=False).head(8)
        top = probs.index[0]
        st.metric("Predicted cuisine", pretty(top), f"{probs.iloc[0]:.0%} confidence")
        fig = px.bar(probs[::-1], orientation="h", labels={"value": "probability", "index": "cuisine"})
        fig.update_layout(showlegend=False, height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        c, pm, fl = fm.compat(ings)
        st.caption(f"Flavor-compatibility of this set: **{c:.2f}** (PMI part {pm:.2f}, shared-compound part {fl:.2f})")
    else:
        st.info("Pick at least two ingredients.")

# ------------------------------------------------------------------ 2. substitute
with tab2:
    st.subheader("Swap an ingredient")
    c1, c2 = st.columns(2)
    with c1:
        sample = rec.sample(200, random_state=7).reset_index(drop=True)
        labels = [f"#{r.id} - {r.cuisine} ({r.ingredients.count('|') + 1} ingredients)" for r in sample.itertuples()]
        choice = st.selectbox("Pick a recipe", range(len(sample)), format_func=lambda i: labels[i])
        recipe_ings = sample.loc[choice, "ingredients"].split("|")
        st.write(", ".join(recipe_ings))
    with c2:
        in_vocab = [i for i in recipe_ings if i in sub.w2v.wv]
        target = st.selectbox("Ingredient to replace", in_vocab) if in_vocab else None
        same_cat = st.checkbox("Same flavor category only", True)
        rerank = st.checkbox("Re-rank with flavor-compound overlap", True)
        k = st.slider("How many suggestions", 3, 10, 5)
    if target:
        res = sub.suggest(target, k=k, same_category=same_cat, rerank=rerank, exclude=set(recipe_ings))
        if res:
            df = pd.DataFrame(res, columns=["substitute", "score", "embedding similarity", "flavor overlap"])
            df["flavor overlap"] = df["flavor overlap"].round(2)
            st.dataframe(df.round(3), hide_index=True, width="stretch")
        else:
            st.warning("No suggestions found for this ingredient.")

# ------------------------------------------------------------------ 3. generate
with tab3:
    st.subheader("Generate ingredient combinations")
    c1, c2 = st.columns(2)
    with c1:
        cuisine = st.selectbox("Target cuisine", fm.cuisines, index=fm.cuisines.index("italian"),
                               format_func=pretty)
        seeds = st.multiselect("Seed ingredients", vocab, default=["chicken", "garlic"], key="seeds")
    with c2:
        n_add = st.slider("Ingredients to add", 1, 6, 4)
        lam = st.slider("Weight: flavor compatibility vs. cuisine fit", 0.0, 1.0, 0.5,
                        help="1 = only flavor compatibility, 0 = only cuisine confidence")
        beam = st.slider("Beam width", 1, 8, 4)
    if st.button("Generate ideas", type="primary") and seeds:
        with st.spinner("Searching..."):
            ideas = generate(fm, seeds, cuisine, n_add=n_add, beam=beam, top_n=8, lam=lam)
        st.session_state["ideas"] = (ideas, seeds, cuisine)
    if "ideas" in st.session_state:
        ideas, seeds_used, cuis_used = st.session_state["ideas"]
        df = pd.DataFrame([{"ingredients": ", ".join(d["ingredients"]), "final score": d["final"],
                            "flavor compatibility": d["compat"], f"P({cuis_used})": d["p_cuisine"]} for d in ideas])
        st.dataframe(df.round(3), hide_index=True, width="stretch")
        best = ideas[0]
        st.success(f"Top idea: {', '.join(best['ingredients'])}  -  flavor compatibility {best['compat']:.2f}, "
                   f"{best['p_cuisine']:.0%} {pretty(cuis_used)}")

        st.markdown("#### Graph view")
        st.caption("Co-occurrence neighbourhood (PMI) of the top idea. Big nodes = ingredients of the idea; "
                   "colours = Louvain communities from the Deliverable 1 analysis.")
        from pyvis.network import Network
        focus = set(best["ingredients"])
        sub_e = edges[edges["source"].isin(focus) | edges["target"].isin(focus)]
        sub_e = sub_e.sort_values("weight", ascending=False).groupby("source").head(6)  # keep readable
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe
        net = Network(height="520px", width="100%", cdn_resources="in_line", bgcolor="#ffffff")
        nodes = focus | set(sub_e["source"]) | set(sub_e["target"])
        for n in nodes:
            net.add_node(n, label=n, size=26 if n in focus else 10,
                         color=palette[comm.get(n, 0) % len(palette)], borderWidth=3 if n in focus else 1)
        for r in sub_e.itertuples():
            net.add_edge(r.source, r.target, value=float(r.weight))
        st.iframe(net.generate_html(), height=540)
        with st.expander("Full ingredient network (top 150 hubs, from the analysis notebook)"):
            p = ROOT / "reports" / "ingredient_network.html"
            if p.exists():
                st.iframe(p.read_text(encoding="utf-8"), height=780)

st.divider()
st.caption("Data: Kaggle What's Cooking (recipes) + Ahn et al. 2011 flavor network (compounds). "
           "Models: TF-IDF/LogReg, Word2Vec, PMI + shared-compound compatibility score.")
