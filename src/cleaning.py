"""Week 1: clean ingredient names and map them to flavor compounds.

Run:  python src/cleaning.py
Outputs (data/processed/):
  recipes.csv                 id, cuisine, ingredients ("|" separated, cleaned)
  ingredient_compounds.csv    ingredient, category, n_compounds, compounds ("|" separated ids)
  ingredient_map.csv          raw name -> cleaned name (audit trail)
  cleaning_report.txt         match rates and merge counts
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# words that describe preparation / brand / quantity, not the ingredient itself
STOP = set("""
fresh freshly chopped minced diced sliced grated shredded crushed ground peeled
cooked uncooked raw large small medium extra virgin finely coarsely roughly thinly
boneless skinless seedless whole halved quartered cubed melted softened packed
frozen thawed canned drained rinsed dried dry low fat reduced sodium lowfat nonfat
fat free unsalted salted sweetened unsweetened organic lean plain hot cold warm
ripe firm soft hard boiled toasted roasted optional divided taste to inch piece pieces
oz ounce ounces lb lbs pound pounds cup cups tablespoon tablespoons teaspoon teaspoons
tbsp tsp g kg ml l can cans package packages jar bottle container a an of the and or
for with without in into plus about more less if needed
""".split())
# brand-ish tokens that are unique to What's Cooking
BRANDS = set("kraft hellmann hellmanns oscar mayer knorr bertolli land lakes pillsbury "
             "frys crystal farms johnsonville breyers hidden valley old el paso "
             "country crock progresso campbell campbells".split())
# words that must stay even though they look like stop words (they change the food)
KEEP = {"ground beef", "ground pork", "ground turkey", "ground lamb", "ground chicken",
        "dry white wine", "dry red wine", "dry sherry", "low sodium soy sauce"}

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _lem = WordNetLemmatizer()
    _lem.lemmatize("tomatoes")
    def lemma(w):
        return _lem.lemmatize(w)
except Exception:  # offline fallback: tiny rule-based singulariser
    def lemma(w):
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("oes"):
            return w[:-2]
        if w.endswith("s") and not w.endswith(("ss", "us")) and len(w) > 3:
            return w[:-1]
        return w

# words the lemmatizer would wrongly singularise
NO_LEMMA = {"molasses", "hummus", "couscous", "asparagus", "swiss", "bass", "citrus", "watercress"}


def clean_name(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"\(.*?\)", " ", s)              # (14.5 oz) etc.
    s = re.sub(r"[^a-z\s'-]", " ", s)           # digits, %, punctuation, accents
    s = s.replace("'", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in KEEP:
        return s
    toks = [t for t in s.split() if t not in STOP and t not in BRANDS]
    toks = [t if t in NO_LEMMA else lemma(t) for t in toks]
    return " ".join(toks).strip()


def fuzzy_merge(counts: Counter, threshold: int = 94):
    """Map rare names onto very similar frequent names (typos / plurals the lemmatizer missed)."""
    names = [n for n, _ in counts.most_common()]
    canon, mapping = [], {}
    for n in names:
        if len(n) < 5 or not canon:
            canon.append(n); mapping[n] = n; continue
        best = process.extractOne(n, canon, scorer=fuzz.ratio, score_cutoff=threshold)
        # never merge when the only difference is a distinguishing word (e.g. "red onion" vs "green onion")
        if best and set(n.split()) ^ set(best[0].split()) and len(n.split()) == len(best[0].split()) \
                and fuzz.ratio(n.split()[-1], best[0].split()[-1]) == 100 and n.split()[0] != best[0].split()[0] \
                and fuzz.ratio(n.split()[0], best[0].split()[0]) < threshold:
            best = None
        if best:
            mapping[n] = best[0]
        else:
            canon.append(n); mapping[n] = n
    return mapping


# manual fixes for frequent ingredients the automatic matcher misses (target must exist in flavor data)
ALIAS = {
    "spinach": "dried spinach", "flour": "whole grain wheat flour",
    "all purpose flour": "whole grain wheat flour", "flour tortilla": "wheat bread",
    "sugar": "sugarcane", "brown sugar": "sugarcane", "white sugar": "sugarcane",
    "granulated sugar": "sugarcane", "powdered sugar": "sugarcane", "light brown sugar": "sugarcane",
    "chili powder": "cayenne", "paprika": "bell pepper", "ketchup": "tomato",
}
# non-flavor staples with no entry in the flavor data (documented, left unmatched):
# salt, water, oil, cooking spray, baking powder, baking soda


def match_compounds(names, flavor_names):
    """Return {clean_name: (flavor_name, how)} using alias -> exact -> suffix -> contains -> fuzzy."""
    fset = set(flavor_names)
    res = {}
    for n in names:
        if n in ALIAS and ALIAS[n] in fset:
            res[n] = (ALIAS[n], "alias"); continue
        if n in fset:
            res[n] = (n, "exact"); continue
        toks = n.split()
        # try dropping leading adjectives: "black olive" -> "olive"
        hit = None
        for i in range(1, len(toks)):
            cand = " ".join(toks[i:])
            if cand in fset:
                hit = cand; break
        if hit:
            res[n] = (hit, "suffix"); continue
        # flavor name contained in ours or vice versa as whole word, longest wins
        # (check every contiguous word sub-phrase, longest first -- fast set lookups)
        cands = [" ".join(toks[i:j]) for i in range(len(toks)) for j in range(i + 1, len(toks) + 1)
                 if " ".join(toks[i:j]) in fset]
        if cands:
            res[n] = (max(cands, key=len), "contains"); continue
        best = process.extractOne(n, flavor_names, scorer=fuzz.ratio, score_cutoff=90)
        if best:
            res[n] = (best[0], "fuzzy")
    return res


def main():
    recipes = json.load(open(RAW / "train.json"))
    raw_counts = Counter(i for r in recipes for i in r["ingredients"])
    print(f"{len(recipes)} recipes, {len(raw_counts)} raw ingredient strings")

    # 1) rule-based clean
    step1 = {raw: clean_name(raw) for raw in raw_counts}
    c1 = Counter()
    for raw, n in raw_counts.items():
        if step1[raw]:
            c1[step1[raw]] += n
    # 2) fuzzy merge
    fm = fuzzy_merge(c1)
    final = {raw: fm.get(step1[raw], "") for raw in raw_counts}
    pd.DataFrame({"raw": list(final), "clean": list(final.values()),
                  "count": [raw_counts[r] for r in final]}) \
        .sort_values("count", ascending=False).to_csv(OUT / "ingredient_map.csv", index=False)

    rows = []
    for r in recipes:
        ings = sorted({final[i] for i in r["ingredients"] if final[i]})
        if len(ings) >= 2:
            rows.append({"id": r["id"], "cuisine": r["cuisine"], "ingredients": "|".join(ings)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recipes.csv", index=False)
    cnt = Counter(i for s in df["ingredients"] for i in s.split("|"))

    # 3) compounds
    info = pd.read_csv(RAW / "flavor_network/ingr_comp/ingr_info.tsv", sep="\t", comment="#",
                       header=None, names=["id", "name", "category"])
    info = info[info["id"] != 339]  # farnesol: also a compound (README issue)
    info["name"] = info["name"].str.replace("_", " ")
    ic = pd.read_csv(RAW / "flavor_network/ingr_comp/ingr_comp.tsv", sep="\t", comment="#",
                     header=None, names=["ing", "comp"])
    comps = ic.groupby("ing")["comp"].apply(lambda s: sorted(set(s))).to_dict()
    info["compounds"] = info["id"].map(comps)
    info = info.dropna(subset=["compounds"])
    fl = info.set_index("name")
    m = match_compounds(list(cnt), list(fl.index))

    out = []
    for n, (f, how) in m.items():
        out.append({"ingredient": n, "flavor_name": f, "match_type": how,
                    "category": fl.loc[f, "category"],
                    "n_compounds": len(fl.loc[f, "compounds"]),
                    "compounds": "|".join(map(str, fl.loc[f, "compounds"])),
                    "recipe_count": cnt[n]})
    ic_df = pd.DataFrame(out).sort_values("recipe_count", ascending=False)
    ic_df.to_csv(OUT / "ingredient_compounds.csv", index=False)

    total_occ = sum(cnt.values())
    matched_occ = sum(cnt[n] for n in m)
    unmatched = [(n, c) for n, c in cnt.most_common() if n not in m]
    rep = [
        f"recipes kept: {len(df)} / {len(recipes)}",
        f"raw ingredient strings: {len(raw_counts)} -> cleaned unique: {len(cnt)}",
        f"match rate (unique ingredients): {len(m)}/{len(cnt)} = {len(m)/len(cnt):.1%}",
        f"match rate (by occurrence):      {matched_occ}/{total_occ} = {matched_occ/total_occ:.1%}",
        "match types: " + str(ic_df['match_type'].value_counts().to_dict()),
        "top 40 unmatched (name, count): " + str(unmatched[:40]),
    ]
    (OUT / "cleaning_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
