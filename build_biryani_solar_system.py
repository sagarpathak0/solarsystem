"""
Build the Biryani World Tree: one HTML with Three.js.
Loads ALL data from biryani_similarity_data.xlsx and biryani_clusters_and_similarity.xlsx.
Generates biryani_solar_system.html with:
  - Ingredient-driven world tree layout (rice base, common trunk, ingredient branches, biryani leaves)
  - Info panel with cuisine filter, biryani subtabs (ingredients, flavour, steps, compounds)
  - Chef Mode toggle with waitlist email popup
Run: python build_biryani_solar_system.py
Open: python -m http.server 8000  then  http://localhost:8000/biryani_solar_system.html
"""

import json, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE = Path(__file__).resolve().parent
CLUSTER_FILE = BASE / "biryani_clusters_and_similarity.xlsx"
DATA_FILE = BASE / "biryani_similarity_data.xlsx"
OUTPUT_HTML = BASE / "biryani_solar_system.html"

WORLD_SCALE = 18
TOP_PAIRINGS = 150

COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A06CD5",
    "#FF9F43", "#2ED573", "#1E90FF", "#FF6348",
    "#00CEC9", "#D63031", "#6C5CE7", "#FDCB6E",
]


def load(name, file=DATA_FILE):
    try:
        return pd.read_excel(file, sheet_name=name)
    except Exception:
        return pd.DataFrame()


def merge_flavour_profiles(t1, t2, dish_name):
    """Merge taste profiles 1 & 2: average shared descriptors, keep unique ones."""
    rows1 = t1[t1["dish_name"] == dish_name] if not t1.empty else pd.DataFrame()
    rows2 = t2[t2["dish_name"] == dish_name] if not t2.empty else pd.DataFrame()
    merged = {}
    for _, r in rows1.iterrows():
        d = str(r.get("descriptor_name", ""))
        cat = str(r.get("descriptor_category", ""))
        i = float(r.get("intensity", 0))
        if d:
            merged[d] = {"descriptor": d, "category": cat, "intensity": i, "count": 1}
    for _, r in rows2.iterrows():
        d = str(r.get("descriptor_name", ""))
        cat = str(r.get("descriptor_category", ""))
        i = float(r.get("intensity", 0))
        if not d:
            continue
        if d in merged:
            merged[d]["intensity"] = (merged[d]["intensity"] + i) / 2
            merged[d]["count"] = 2
            if cat and cat != "nan":
                merged[d]["category"] = cat
        else:
            merged[d] = {"descriptor": d, "category": cat, "intensity": i, "count": 1}
    result = sorted(merged.values(), key=lambda x: -x["intensity"])
    return [{"descriptor": r["descriptor"], "category": r["category"], "intensity": round(r["intensity"], 3)} for r in result]



def build_hierarchy(member_indices, S, dish_info_list, max_children=4):
    """Recursively build a balanced tree where each internal node has at most max_children children."""
    members = list(member_indices)
    n = len(members)

    if n <= max_children:
        return {
            "t": "g",
            "lb": _subgroup_label(members, dish_info_list),
            "c": [{"t": "l", "id": int(m)} for m in members],
        }

    # Number of sub-groups: ideally ceil(n/max_children) but capped at max_children
    k = min(max_children, max(2, math.ceil(n / max_children)))

    # Sub-cluster using similarity matrix
    idx_arr = np.array(members)
    sub_S = S[np.ix_(idx_arr, idx_arr)]
    D = 1 - np.clip(sub_S, 0, 1)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2  # ensure symmetry
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    labels = fcluster(Z, k, criterion="maxclust")

    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(members[i])

    children = []
    for g in sorted(groups.values(), key=len, reverse=True):
        if len(g) == 1:
            children.append({"t": "l", "id": int(g[0])})
        else:
            children.append(build_hierarchy(g, S, dish_info_list, max_children))

    return {"t": "g", "lb": _subgroup_label(members, dish_info_list), "c": children}


def _subgroup_label(members, dish_info_list):
    """Short descriptive label for a sub-group based on dominant cuisine or key ingredient."""
    cuisines = [dish_info_list[i].get("cuisine", "") for i in members]
    cuisines = [c for c in cuisines if c]
    if cuisines:
        top_c, count = Counter(cuisines).most_common(1)[0]
        if count >= len(members) * 0.5 and top_c:
            return top_c

    # Try finding a shared non-generic ingredient
    generic = {
        "salt", "oil", "water", "ghee", "onions", "ginger", "garlic",
        "rice", "tomatoes", "yoghurt", "mint leaves", "coriander leaves",
        "green chilli", "red chilli powder", "chilli powder",
    }
    ingredient_counts = Counter()
    for i in members:
        seen = set()
        for ing_str in dish_info_list[i].get("ingredients", []):
            parts = ing_str.lower().split()
            name = " ".join(
                p for p in parts
                if not p.replace(".", "").replace(",", "").isdigit()
                and p not in ("tsp", "tbs", "cup", "gm", "number", "as", "required", "to", "taste")
            ).strip()
            if name and name not in generic and name not in seen and len(name) > 2:
                ingredient_counts[name] += 1
                seen.add(name)
    if ingredient_counts:
        top_ing, count = ingredient_counts.most_common(1)[0]
        if count >= len(members) * 0.4:
            return top_ing.title()[:22]
    return ""


def main():
    print("Loading data...")
    clusters = load("Cluster_Labels", CLUSTER_FILE)
    pairs = load("Pairwise_Final_Similarity", CLUSTER_FILE)
    dish_list = load("Dish_List")
    ingredients = load("Ingredients_By_Dish")
    taste1 = load("Taste_Profile_1")
    taste2 = load("Taste_Profile_2")
    cooking_steps = load("Cooking_Steps")
    metadata = load("Dish_Metadata")
    dish_compounds = load("Dish_Compounds")
    cooking_style_master = load("Cooking_Style_Master")
    step_ingredient = load("Step_Ingredient")
    ingredient_master = load("Ingredient_Master")
    ingredient_scientific = load("Ingredient_Scientific")
    ingredient_compound = load("Ingredient_Chemical_Compound")
    compound_synergy_rule = load("Compound_Synergy_Rule")
    compound_synergy_member = load("Compound_Synergy_Member")
    ingredient_taste = load("Ingredient_Taste")
    cluster_meta_df = load("Cluster_Meta", CLUSTER_FILE)

    dish_ids = clusters["dish_id"].astype(int).tolist()
    id_to_idx = {d: i for i, d in enumerate(dish_ids)}
    n = len(dish_ids)

    # --- Similarity matrix ---
    S = np.eye(n)
    for _, row in pairs.iterrows():
        i = id_to_idx.get(int(row["dish_id_a"]))
        j = id_to_idx.get(int(row["dish_id_b"]))
        if i is not None and j is not None:
            S[i, j] = S[j, i] = float(row["final_similarity"])

    # --- MDS 3D embedding ---
    D = 1 - np.clip(S, 0, 1)
    np.fill_diagonal(D, 0)
    print("MDS embedding...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mds = MDS(n_components=3, metric=True, init="random", dissimilarity="precomputed", random_state=42, n_init=8, max_iter=800)
        except TypeError:
            mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(D)

    coords -= coords.mean(axis=0)
    s = np.abs(coords).max()
    if s > 1e-6:
        coords = coords / s * WORLD_SCALE

    # --- Colors by overall weighted cluster (meaningful grouping) ---
    cluster_col = "cluster_overall" if "cluster_overall" in clusters.columns else "cluster_ingredient"
    vals = clusters[cluster_col].astype(int).tolist()
    uniq_clusters = sorted(set(vals))
    cluster_cmap = {c: COLORS[i % len(COLORS)] for i, c in enumerate(uniq_clusters)}

    # Load cluster labels from Cluster_Meta sheet
    cluster_meta = load("Cluster_Meta", CLUSTER_FILE)
    cluster_label_map = {}
    if not cluster_meta.empty and "cluster_id" in cluster_meta.columns:
        for _, r in cluster_meta.iterrows():
            cluster_label_map[int(r["cluster_id"])] = str(r["label"])

    cuisines = sorted(metadata["cuisine_type"].dropna().unique().tolist()) if "cuisine_type" in metadata.columns else []

    nodes = []
    dish_info = []
    for idx in range(n):
        did = dish_ids[idx]
        name = str(clusters["dish_name"].iloc[idx])
        cluster = int(vals[idx])

        color = cluster_cmap.get(cluster, "#FFFFFF")
        cluster_label = cluster_label_map.get(cluster, f"Group {cluster}")

        # Metadata
        meta_row = metadata[metadata["dish_id"] == did]
        cuisine = str(meta_row["cuisine_type"].iloc[0]) if len(meta_row) and pd.notna(meta_row["cuisine_type"].iloc[0]) else ""
        category = str(meta_row["dish_category"].iloc[0]) if len(meta_row) and pd.notna(meta_row["dish_category"].iloc[0]) else ""
        min_t = meta_row["min_total_cooking_time_minutes"].iloc[0] if len(meta_row) and pd.notna(meta_row["min_total_cooking_time_minutes"].iloc[0]) else None
        max_t = meta_row["max_total_cooking_time_minutes"].iloc[0] if len(meta_row) and pd.notna(meta_row["max_total_cooking_time_minutes"].iloc[0]) else None
        serves = str(meta_row["serves_count"].iloc[0]) if len(meta_row) and pd.notna(meta_row["serves_count"].iloc[0]) else ""
        total_time = ""
        if min_t is not None:
            total_time = f"{int(min_t)}" + (f"–{int(max_t)}" if max_t is not None and max_t != min_t else "") + " min"

        # Ingredients
        ing_df = ingredients[ingredients["dish_id"] == did] if not ingredients.empty else pd.DataFrame()
        ing_list = []
        import re
        
        # A comprehensive regex to match numbers (including fractions and hyphens) optionally followed by measurement units.
        # This matches patterns like "1", "0.5", "1/2", "2 1-inch sticks", "Tbs", "1 cup", etc. at the START of the string.
        # We explicitly wrap the units with word boundaries \b to prevent stripping first letters of actual ingredients (e.g. 'G' from Garlic).
        unit_pattern = r'(?i)^(?:[\d\s\./\-]+(?:inch|inch\s+sticks|sticks|inch\s+pieces|pieces|piece)?\s*)?(?:\b(?:to\s+taste|as\s+required|tbs|tbsp|tablespoon|tsp|teaspoon|cup|cups|number|numbers|gm|g|kg|ml|l|sprig|sprigs|clove|cloves|pod|pods|stick|sticks|leaf|leaves|slice|slices)\b)?\s*'

        for _, ir in ing_df.iterrows():
            nm = str(ir.get("ingredient_name", "")).strip()
            prep = str(ir.get("preparation_method", "")).strip() if pd.notna(ir.get("preparation_method")) else ""
            
            # Start with just the ingredient name. We completely ignore the 'quantity' and 'unit_of_measure' columns
            # because they are often incorrectly baked into the 'ingredient_name' string itself in this dataset.
            line = nm
            
            # Aggressively strip out the measurement pattern from the beginning of the string
            line = re.sub(unit_pattern, '', line).strip()
            
            # Catch standalone "Tbs", "tsp", etc., that might not be at the very beginning
            line = re.sub(r'(?i)\b(?:to\s+taste|as\s+required|tbs|tbsp|tablespoon|tsp|teaspoon|cup|cups|number|numbers|gm|g|kg|ml|l|sprig|sprigs|clove|cloves|pod|pods|stick|sticks|leaf|leaves|slice|slices)\b', '', line).strip()
            
            # Catch leftover leading numbers that might have been unattached to a recognized unit
            line = re.sub(r'^[\d\s\./\-]+', '', line).strip()
            
            # Add preparation method if exists
            if prep:
                line += f" ({prep})"
                
            # Clean up extra spaces
            line = re.sub(r'\s+', ' ', line).strip() 
            line = line.capitalize() # capitalize the first char for neatness
            
            
            # Hardcoded fix for the most notorious broken strings that happen after bad regex clipping.
            # E.g. If the string became 'inger' or 'arlic', this puts it back.
            fixes = {
                "inger (paste)": "Ginger (paste)",
                "arlic (paste)": "Garlic (paste)",
                "inger": "Ginger",
                "arlic": "Garlic",
                "hee": "Ghee",
                "reen cardamoms": "Green cardamoms",
                "reen chillies": "Green chillies",
                "s": "Cloves", # Assuming if it just stripped to "s" it was cloves.
                "alt": "Salt"
            }
            clean_lc = line.lower()
            if clean_lc in fixes:
                line = fixes[clean_lc]
                
            if line.strip() != "":
                ing_list.append(line)

        # Flavour profile (merged)
        flavour = merge_flavour_profiles(taste1, taste2, name)

        # Cooking steps
        step_df = cooking_steps[cooking_steps["dish_id"] == did].sort_values("step_number") if not cooking_steps.empty else pd.DataFrame()
        steps = []
        for _, sr in step_df.iterrows():
            steps.append({
                "step": int(sr["step_number"]) if pd.notna(sr.get("step_number")) else 0,
                "desc": str(sr.get("description", "")),
                "style": str(sr.get("cooking_style_name", "")) if pd.notna(sr.get("cooking_style_name")) else "",
                "dur": float(sr["duration_minutes"]) if pd.notna(sr.get("duration_minutes")) else None,
            })

        # Top compounds
        comp_df = dish_compounds[dish_compounds["dish_id"] == did] if not dish_compounds.empty else pd.DataFrame()
        compounds = []
        if not comp_df.empty:
            top = comp_df.nlargest(12, "concentration_percentage") if "concentration_percentage" in comp_df.columns else comp_df.head(12)
            for _, cr in top.iterrows():
                compounds.append({
                    "ingredient": str(cr.get("ingredient_name", "")),
                    "compound": str(cr.get("compound_name", "")),
                    "conc": round(float(cr["concentration_percentage"]), 3) if pd.notna(cr.get("concentration_percentage")) else None,
                })

        # Step ingredients (join via cooking_step_id from cooking_steps for this dish)
        step_ing_list = []
        if not step_ingredient.empty and not step_df.empty:
            if "cooking_step_id" in step_df.columns and "cooking_step_id" in step_ingredient.columns:
                step_ids = step_df["cooking_step_id"].dropna().astype(int).tolist()
                si_df = step_ingredient[step_ingredient["cooking_step_id"].isin(step_ids)]
                si_merged = si_df.merge(
                    step_df[["cooking_step_id", "step_number"]].drop_duplicates(),
                    on="cooking_step_id", how="left"
                ) if not si_df.empty else pd.DataFrame()
                for _, sir in si_merged.iterrows():
                    step_ing_list.append({
                        "step": int(sir.get("step_number", 0)) if pd.notna(sir.get("step_number")) else 0,
                        "ingredient": str(sir.get("ingredient_name", "")),
                        "quantity": str(sir.get("quantity", "")) if pd.notna(sir.get("quantity")) else "",
                        "purpose": str(sir.get("purpose", "")) if pd.notna(sir.get("purpose")) else "",
                    })

        # Cooking style details
        style_details = []
        if not cooking_style_master.empty:
            style_col = "style_name" if "style_name" in cooking_style_master.columns else "cooking_style_name"
            step_styles = step_df["cooking_style_name"].dropna().unique() if not step_df.empty and "cooking_style_name" in step_df.columns else []
            for sname in step_styles:
                cs_row = cooking_style_master[cooking_style_master[style_col] == sname] if style_col in cooking_style_master.columns else pd.DataFrame()
                if not cs_row.empty:
                    r = cs_row.iloc[0]
                    temp_min = r.get("temperature_range_min_c", "")
                    temp_max = r.get("temperature_range_max_c", "")
                    temp_str = ""
                    if pd.notna(temp_min):
                        temp_str = f"{int(temp_min)}"
                        if pd.notna(temp_max) and temp_max != temp_min:
                            temp_str += f"–{int(temp_max)}"
                        temp_str += "°C"
                    style_details.append({
                        "style": str(sname),
                        "temp": temp_str,
                        "browning": str(r.get("browning_potential", "")) if pd.notna(r.get("browning_potential")) else "",
                        "moisture": str(r.get("moisture_change", "")) if pd.notna(r.get("moisture_change")) else "",
                    })

        # --- Ingredient origins (aggregated for this dish) ---
        origins_list = []
        if not ing_df.empty and not ingredient_master.empty:
            for _, ir in ing_df.iterrows():
                iid = ir.get("ingredient_id")
                if pd.notna(iid):
                    im_row = ingredient_master[ingredient_master["ingredient_id"] == int(iid)]
                    if not im_row.empty:
                        orig = im_row.iloc[0].get("origin", "")
                        if pd.notna(orig) and str(orig).strip():
                            origins_list.append({"ingredient": str(ir.get("ingredient_name","")), "origin": str(orig)})

        # --- Ingredient science (aggregated nutrition for this dish) ---
        science_agg = {"pH": [], "moisture": [], "fat": [], "protein": [], "carbs": [], "maillard": 0, "total": 0}
        if not ing_df.empty and not ingredient_scientific.empty:
            for _, ir in ing_df.iterrows():
                iid = ir.get("ingredient_id")
                if pd.notna(iid):
                    sci_row = ingredient_scientific[ingredient_scientific["ingredient_id"] == int(iid)]
                    if not sci_row.empty:
                        sr2 = sci_row.iloc[0]
                        science_agg["total"] += 1
                        for col, key in [("fat_percentage","fat"),("protein_percentage","protein"),("carbs_percentage","carbs"),("moisture_percentage","moisture")]:
                            v = sr2.get(col)
                            if pd.notna(v):
                                try: science_agg[key].append(float(v))
                                except: pass
                        if str(sr2.get("maillard_responsive","")).lower().startswith("yes"):
                            science_agg["maillard"] += 1
        science_profile = {}
        if science_agg["total"] > 0:
            for key in ["fat","protein","carbs","moisture"]:
                vals_list = science_agg[key]
                if vals_list:
                    science_profile[key] = round(sum(vals_list)/len(vals_list), 1)
            science_profile["maillardPct"] = round(100*science_agg["maillard"]/science_agg["total"])

        # --- Taste by category (Flavour, Mouthfeel, Smell, Sound, Visual) ---
        taste_cats = {}
        for f in flavour:
            cat = f.get("category", "Flavour")
            if cat in ("nan", "", None): cat = "Flavour"
            taste_cats.setdefault(cat, []).append({"d": f["descriptor"], "i": round(f["intensity"], 2)})

        # --- Compound synergies active in this dish ---
        dish_synergies = []
        if not comp_df.empty and not compound_synergy_rule.empty and not compound_synergy_member.empty:
            # Get compound IDs present in this dish
            dish_compound_names = set(comp_df["compound_name"].dropna().astype(str).str.lower())
            if "ingredient_compound_id" in compound_synergy_member.columns:
                # Build lookup: compound_id -> compound_name
                cid_to_name = {}
                if "ingredient_compound_id" in ingredient_compound.columns and "compound_name" in ingredient_compound.columns:
                    for _, cr2 in ingredient_compound.iterrows():
                        cid_to_name[cr2["ingredient_compound_id"]] = str(cr2.get("compound_name","")).lower()
                # Check which synergy rules are satisfied
                for _, sr in compound_synergy_rule.iterrows():
                    rule_id = sr.get("synergy_rule_id") or sr.get("synergy_name")
                    members = compound_synergy_member[compound_synergy_member["synergy_rule_name"] == sr.get("synergy_name")]
                    if members.empty:
                        continue
                    member_compounds = set()
                    for _, mr in members.iterrows():
                        cid = mr.get("ingredient_compound_id")
                        cname = cid_to_name.get(cid, "")
                        if cname:
                            member_compounds.add(cname)
                    # Check how many are present in this dish
                    present = member_compounds & dish_compound_names
                    min_req = int(sr.get("minimum_compounds_required", 2)) if pd.notna(sr.get("minimum_compounds_required")) else 2
                    if len(present) >= min_req:
                        dish_synergies.append({
                            "name": str(sr.get("synergy_name","")).replace("_"," "),
                            "type": str(sr.get("synergy_type","")),
                            "factor": round(float(sr.get("amplification_factor",1)),1) if pd.notna(sr.get("amplification_factor")) else 1,
                            "affects": str(sr.get("affected_descriptor_name","")),
                            "mechanism": str(sr.get("amplification_mechanism",""))[:120],
                        })
            dish_synergies = dish_synergies[:8]  # limit

        # --- Prep / marination times ---
        prep_time = ""
        marination_time = ""
        if len(meta_row):
            p_min = meta_row["min_prep_time_minutes"].iloc[0] if pd.notna(meta_row["min_prep_time_minutes"].iloc[0]) else None
            p_max = meta_row["max_prep_time_minutes"].iloc[0] if pd.notna(meta_row["max_prep_time_minutes"].iloc[0]) else None
            if p_min is not None:
                prep_time = f"{int(p_min)}" + (f"–{int(p_max)}" if p_max and p_max != p_min else "") + " min"
            m_min = meta_row["min_marination_time_minutes"].iloc[0] if pd.notna(meta_row["min_marination_time_minutes"].iloc[0]) else None
            m_max = meta_row["max_marination_time_minutes"].iloc[0] if pd.notna(meta_row["max_marination_time_minutes"].iloc[0]) else None
            if m_min is not None:
                marination_time = f"{int(m_min)}" + (f"–{int(m_max)}" if m_max and m_max != m_min else "") + " min"

        nodes.append({
            "id": idx, "name": name,
            "x": round(float(coords[idx, 0]), 3),
            "y": round(float(coords[idx, 1]), 3),
            "z": round(float(coords[idx, 2]), 3),
            "color": color, "cluster": cluster, "cuisine": cuisine,
            "clusterLabel": cluster_label,
        })
        # Top 3 similar biryanis (from similarity matrix, exclude self)
        sim_scores = [(j, float(S[idx, j])) for j in range(n) if j != idx]
        sim_scores.sort(key=lambda x: -x[1])
        top_similar = []
        for j, score in sim_scores[:3]:
            top_similar.append({
                "id": int(j),
                "name": str(clusters["dish_name"].iloc[j]),
                "score": round(score, 3),
            })

        dish_info.append({
            "name": name, "cuisine": cuisine, "category": category,
            "totalTime": total_time, "serves": serves,
            "prepTime": prep_time, "marinationTime": marination_time,
            "ingredients": ing_list, "flavour": flavour,
            "tasteCats": taste_cats,
            "steps": steps, "compounds": compounds,
            "stepIngredients": step_ing_list,
            "styleDetails": style_details,
            "origins": origins_list,
            "science": science_profile,
            "synergies": dish_synergies,
            "topSimilar": top_similar,
            "clusterLabel": cluster_label,
        })

    # --- Cluster centers (per overall cluster for meaningful grouping) ---
    centers = []
    for cid in uniq_clusters:
        mask = np.array(vals) == cid
        if mask.sum() == 0:
            continue
        centers.append({
            "x": round(float(coords[mask, 0].mean()), 3),
            "y": round(float(coords[mask, 1].mean()), 3),
            "z": round(float(coords[mask, 2].mean()), 3),
            "color": cluster_cmap[cid],
            "label": cluster_label_map.get(cid, f"Group {cid}"),
            "size": int(mask.sum()),
        })

    # --- Build hierarchical tree per cluster (max ~4 children per node) ---
    print("Building cluster hierarchies...")
    hierarchy_data = []
    for cid in uniq_clusters:
        members = [i for i in range(n) if vals[i] == cid]
        cl_label = cluster_label_map.get(cid, f"Group {cid}")
        color = cluster_cmap.get(cid, "#FFFFFF")

        if len(members) <= 1:
            tree = {"t": "g", "lb": cl_label, "c": [{"t": "l", "id": int(m)} for m in members]}
        else:
            tree = build_hierarchy(members, S, dish_info, max_children=4)
            tree["lb"] = cl_label  # override root with cluster label

        hierarchy_data.append({"clusterLabel": cl_label, "color": color, "tree": tree})

    # --- Top pairings (with sub-scores) ---
    pair_list = []
    for _, row in pairs.iterrows():
        i = id_to_idx.get(int(row["dish_id_a"]))
        j = id_to_idx.get(int(row["dish_id_b"]))
        if i is not None and j is not None:
            entry = {
                "a": i, "b": j,
                "sim": round(float(row["final_similarity"]), 3),
            }
            for col, key in [("sim_ingredient","sIng"),("sim_flavour1","sFlav1"),
                             ("sim_flavour2","sFlav2"),("sim_cooking_style","sCook"),
                             ("sim_compound","sCmpd"),("sim_cuisine","sCuis"),
                             ("sim_cooking_time","sTime")]:
                if col in row and pd.notna(row[col]):
                    entry[key] = round(float(row[col]), 3)
            pair_list.append(entry)
    pair_list.sort(key=lambda t: -t["sim"])
    pairings = pair_list[:TOP_PAIRINGS]

    # --- Build cluster meta for quiz/trivia ---
    cluster_meta = []
    if not cluster_meta_df.empty:
        for _, cm in cluster_meta_df.iterrows():
            cluster_meta.append({
                "label": str(cm.get("cluster_label","")),
                "size": int(cm["cluster_size"]) if pd.notna(cm.get("cluster_size")) else 0,
                "cuisine": str(cm.get("dominant_cuisine","")),
                "styles": str(cm.get("top_cooking_styles","")),
                "sharedIng": str(cm.get("shared_ingredients","")),
            })

    # --- Unique ingredient origins for map view ---
    all_origins = sorted(ingredient_master["origin"].dropna().unique().tolist()) if not ingredient_master.empty and "origin" in ingredient_master.columns else []

    data = {
        "nodes": nodes, "dishInfo": dish_info,
        "clusterCenters": centers, "pairings": pairings,
        "cuisines": cuisines,
        "hierarchy": hierarchy_data,
        "clusterMeta": cluster_meta,
        "origins": all_origins,
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html = build_html(data_json)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print("Written", OUTPUT_HTML)




def build_html(data_json):
    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Biryani Universe — Interactive Explorer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#060e1e;color:#dff8ff;overflow:hidden;cursor:default}
#bg{position:fixed;inset:0;background:radial-gradient(ellipse at 50% 50%,rgba(20,50,100,.35),transparent 68%),radial-gradient(circle at 80% 20%,rgba(100,60,200,.12),transparent 40%),radial-gradient(circle at 20% 80%,rgba(20,180,200,.10),transparent 38%),#060e1e;z-index:0}
canvas#star-bg{position:fixed;inset:0;z-index:1;pointer-events:none}
#canvas{position:fixed;inset:0;z-index:5}
.glass{background:rgba(8,18,38,.84);border:1px solid rgba(100,200,255,.22);border-radius:14px;backdrop-filter:blur(14px)}

/* HUD */
#hud{position:fixed;left:14px;top:14px;z-index:50;display:flex;gap:8px;align-items:flex-start;flex-direction:column}
#search-wrap{position:relative;padding:8px 10px}
#search{width:250px;padding:8px 30px 8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);color:#e7f9ff;font-size:12px;outline:none;font-family:'Inter',sans-serif}
#search::placeholder{color:#5e8aa5}
#search-clear{position:absolute;right:16px;top:50%;transform:translateY(-50%);background:none;border:none;color:#5e8aa5;cursor:pointer;display:none;font-size:13px}
#search-clear.visible{display:block}
#controls{display:flex;gap:6px;padding:8px;flex-wrap:wrap}
.ctrl{border:none;border-radius:9px;padding:7px 12px;cursor:pointer;background:rgba(255,255,255,.05);color:#8dc5dd;font-size:11px;font-weight:500;letter-spacing:.3px;transition:all .15s;border:1px solid transparent}
.ctrl:hover{background:rgba(100,220,255,.13);color:#c8f7ff}
.ctrl.on{background:rgba(100,220,255,.22);color:#e6fcff;border-color:rgba(100,220,255,.4)}

/* Score HUD */
#score-hud{position:fixed;top:14px;right:14px;z-index:50;padding:12px 16px;min-width:190px;text-align:center}
#score-hud .xp-title{font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:2px;color:#5c94b0;margin-bottom:5px}
#xp-row{display:flex;align-items:center;gap:8px;justify-content:center;margin-bottom:4px}
#xp-num{font-family:'Orbitron',sans-serif;font-size:24px;font-weight:700;color:#6ef3ff;text-shadow:0 0 14px rgba(110,243,255,.4)}
#xp-total-lbl{font-size:10px;color:#7badc4}
#xp-bar-wrap{width:100%;height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;margin-bottom:3px}
#xp-bar{height:100%;width:0%;background:linear-gradient(90deg,#3de8ff,#a78bfa);transition:width .5s}
#xp-msg{font-size:10px;color:#7db3c9;min-height:13px}
.rank-badge{display:inline-block;padding:2px 10px;border-radius:999px;font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:1px;margin-top:5px}
#streak{font-size:10px;color:#ffd166;min-height:13px;margin-top:2px}

/* Tooltip */
#tooltip{position:fixed;z-index:60;padding:10px 14px;border-radius:12px;pointer-events:none;opacity:0;transition:opacity .1s;max-width:240px;font-size:11px;line-height:1.5}
#tooltip .tt-name{font-family:'Orbitron',sans-serif;font-size:11px;color:#8cf7ff;margin-bottom:3px}
#tooltip .tt-row{display:flex;justify-content:space-between;color:#8abed5;margin:1px 0}
#tooltip .tt-hint{margin-top:5px;font-size:9px;color:#5d93aa;font-style:italic}

/* Detail panel */
#info-panel{position:fixed;right:14px;top:150px;bottom:14px;width:375px;z-index:48;padding:16px;overflow:hidden;display:flex;flex-direction:column;opacity:0;transform:translateX(22px);pointer-events:none;transition:all .28s}
#info-panel.visible{opacity:1;transform:translateX(0);pointer-events:auto}
#close-panel{position:absolute;top:10px;right:12px;background:none;border:none;color:#6ea8c0;font-size:18px;cursor:pointer}
.dish-name{font-family:'Orbitron',sans-serif;font-size:16px;color:#7af3ff;padding-right:22px;letter-spacing:.4px;line-height:1.3}
.cuisine-badge{display:inline-block;margin-top:6px;padding:3px 10px;border-radius:999px;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase}
#panel-meta{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0 4px}
.chip{font-size:10px;padding:2px 7px;border-radius:999px;background:rgba(100,200,255,.1);border:1px solid rgba(100,200,255,.22);color:#9dd8ee}
#panel-tabs{display:flex;gap:4px;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,.07);border-bottom:1px solid rgba(255,255,255,.07);padding:6px 0;margin-bottom:6px;flex-shrink:0}
.tab-btn{border:none;background:transparent;color:#7ab0c8;font-size:10px;padding:4px 9px;border-radius:7px;cursor:pointer;transition:all .12s;white-space:nowrap}
.tab-btn:hover{background:rgba(100,220,255,.1)}
.tab-btn.active{background:rgba(110,243,255,.18);color:#d8fcff}
#panel-body{flex:1;overflow-y:auto;overflow-x:hidden}
#panel-body::-webkit-scrollbar{width:4px}
#panel-body::-webkit-scrollbar-thumb{background:rgba(100,200,255,.2);border-radius:2px}
.tab{display:none}
.tab.show{display:block}
.list{list-style:none}
.list li{font-size:11px;line-height:1.5;padding:3px 0;color:#a0cfe0;border-bottom:1px solid rgba(255,255,255,.04)}
.list.bullet li{display:flex;gap:6px;align-items:flex-start}
.list.bullet li::before{content:'◆';font-size:5px;color:#6ef3ff;flex-shrink:0;margin-top:7px}
.muted{font-size:11px;color:#5e8faa;font-style:italic;padding:4px 0}
.section-head{font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:1.2px;color:#5c94b0;margin:8px 0 4px;text-transform:uppercase}

/* Flavour bars */
.flavour{display:flex;gap:7px;align-items:center;margin:4px 0}
.fname{width:100px;font-size:10px;color:#7aacca;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fbar{flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden}
.ffill{height:100%;border-radius:3px}
.fval{width:32px;text-align:right;font-size:10px;color:#8ec0d8}
.cat-label{font-size:9px;color:#a78bfa;letter-spacing:.8px;margin:8px 0 3px;text-transform:uppercase;font-weight:600}

/* Steps */
.step{position:relative;padding-left:18px;margin-bottom:7px;font-size:11px;line-height:1.45;color:#a0d0e4}
.step::before{content:attr(data-step);position:absolute;left:0;top:0;color:#6ef3ff;font-size:10px;font-weight:700;font-family:'Orbitron',sans-serif}
.tiny{font-size:9px;color:#5e8faa;margin-top:1px}

/* Science tab */
.sci-bar-row{display:flex;gap:8px;align-items:center;margin:4px 0;font-size:11px}
.sci-label{width:70px;color:#7aacca;font-size:10px}
.sci-bar{flex:1;height:8px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.sci-fill{height:100%;border-radius:4px;transition:width .6s}
.sci-val{width:38px;text-align:right;font-size:10px;color:#8ec0d8}
.sci-stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.sci-stat{background:rgba(100,200,255,.06);border:1px solid rgba(100,200,255,.14);border-radius:8px;padding:7px 10px;text-align:center}
.sci-stat-val{font-family:'Orbitron',sans-serif;font-size:16px;font-weight:700;color:#6ef3ff}
.sci-stat-lbl{font-size:9px;color:#5e8faa;margin-top:2px}
.origin-row{display:flex;justify-content:space-between;font-size:10px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);color:#8abed5}
.origin-tag{font-size:9px;padding:1px 6px;border-radius:999px;background:rgba(167,139,250,.12);color:#c4b0ff;border:1px solid rgba(167,139,250,.2)}

/* Synergy cards */
.syn-card{background:rgba(100,200,255,.06);border:1px solid rgba(100,200,255,.16);border-radius:10px;padding:9px 11px;margin:5px 0}
.syn-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px}
.syn-name{font-size:11px;color:#8cf7ff;font-weight:600}
.syn-type{font-size:8px;padding:2px 7px;border-radius:999px;font-weight:600;letter-spacing:.5px}
.syn-type.Amplification{background:rgba(61,220,151,.15);color:#3ddc97;border:1px solid rgba(61,220,151,.3)}
.syn-type.Suppression{background:rgba(255,127,145,.15);color:#ff7f91;border:1px solid rgba(255,127,145,.3)}
.syn-type.Modification{background:rgba(255,209,102,.15);color:#ffd166;border:1px solid rgba(255,209,102,.3)}
.syn-type.Creation{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}
.syn-affects{font-size:10px;color:#7ab0c8;margin-bottom:3px}
.syn-mechanism{font-size:9px;color:#5e8faa;line-height:1.4}
.syn-factor{font-family:'Orbitron',sans-serif;font-size:13px;color:#ffd166;font-weight:700}

/* Radar chart */
#radar-wrap{display:flex;justify-content:center;padding:8px 0}
#radar-wrap svg text{font-family:'Inter',sans-serif}

/* Similar tab */
.sim-item{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);cursor:pointer}
.sim-item:hover .sim-name{color:#6ef3ff}
.sim-name{font-size:11px;color:#9ddcf0;transition:color .12s}
.sim-score-pill{font-size:9px;padding:2px 8px;border-radius:999px;background:rgba(110,243,255,.1);color:#6ef3ff;border:1px solid rgba(110,243,255,.2)}
.sub-score-row{display:flex;gap:3px;flex-wrap:wrap;margin-top:3px}
.sub-pill{font-size:8px;padding:1px 5px;border-radius:999px;background:rgba(167,139,250,.08);color:#a78bfa;border:1px solid rgba(167,139,250,.15)}

/* Compare panel */
#compare-panel{position:fixed;left:14px;bottom:70px;z-index:50;padding:14px 16px;width:400px;max-height:72vh;overflow-y:auto;opacity:0;pointer-events:none;transition:all .22s}
#compare-panel::-webkit-scrollbar{width:4px}
#compare-panel::-webkit-scrollbar-thumb{background:rgba(100,200,255,.2);border-radius:2px}
#compare-panel.visible{opacity:1;pointer-events:auto}
.cp-title{font-family:'Orbitron',sans-serif;font-size:10px;color:#5c94b0;letter-spacing:1.4px;margin-bottom:8px}
.cp-vs{display:flex;gap:10px;align-items:center;margin-bottom:8px}
.cp-dish{flex:1;padding:7px 9px;border-radius:9px;background:rgba(100,200,255,.06);border:1px solid rgba(100,200,255,.18);text-align:center;font-size:10px;color:#9ddcf0}
.cp-dish.filled{color:#7af3ff;border-color:rgba(110,243,255,.4)}
.cp-spark{font-family:'Orbitron',sans-serif;font-size:16px;font-weight:700;color:#a78bfa}
.cp-score-bar{width:100%;height:8px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden;margin:4px 0}
.cp-score-fill{height:100%;border-radius:4px;transition:width .5s}
.cp-score-label{font-family:'Orbitron',sans-serif;font-size:20px;font-weight:700;text-align:center}
.cp-score-text{font-size:10px;color:#7db3c9;text-align:center;margin-bottom:8px}
.cp-section{margin-bottom:10px}
.cp-section-title{font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:1.1px;color:#5c94b0;margin-bottom:5px;text-transform:uppercase}
.cp-grid{display:grid;grid-template-columns:1fr auto 1fr;gap:4px 8px;font-size:10px;color:#8abed5}
.cp-grid .lbl{text-align:center;color:#5c94b0;font-size:9px}
.cp-grid .va{text-align:right;padding:2px 5px;border-radius:5px;background:rgba(255,127,145,.08)}
.cp-grid .vb{text-align:left;padding:2px 5px;border-radius:5px;background:rgba(167,139,250,.08)}
.cp-shared{display:flex;flex-wrap:wrap;gap:3px;margin-top:4px}
.cp-tag{font-size:9px;padding:2px 6px;border-radius:999px;background:rgba(110,243,255,.1);color:#8cf7ff;border:1px solid rgba(110,243,255,.18)}
.cp-tag.ua{background:rgba(255,127,145,.1);color:#ff9daa;border-color:rgba(255,127,145,.22)}
.cp-tag.ub{background:rgba(167,139,250,.1);color:#c4b0ff;border-color:rgba(167,139,250,.22)}
.sub-sim-row{display:flex;align-items:center;gap:7px;margin:3px 0;font-size:10px}
.sub-sim-lbl{width:90px;color:#7aacca;font-size:9px}
.sub-sim-bar{flex:1;height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden}
.sub-sim-fill{height:100%;border-radius:3px;transition:width .5s}
.sub-sim-val{width:30px;text-align:right;font-size:9px;color:#8ec0d8}
.cp-frow{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:9px;color:#7aacca}
.cp-frow .fn{width:70px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp-frow .fb{flex:1;height:4px;background:rgba(255,255,255,.04);border-radius:2px;position:relative;overflow:hidden}
.cp-frow .ffA{position:absolute;top:0;left:0;height:100%;border-radius:2px;background:#ff7f91}
.cp-frow .ffB{position:absolute;top:0;right:0;height:100%;border-radius:2px;background:#a78bfa}
.cp-actions{display:flex;gap:8px;justify-content:center;margin-top:8px}
.cp-btn{border:none;border-radius:8px;padding:5px 13px;cursor:pointer;font-size:10px;font-weight:500}
.cp-btn.clear{background:rgba(255,100,100,.1);color:#ff9d9d;border:1px solid rgba(255,100,100,.28)}
.cp-btn.clear:hover{background:rgba(255,100,100,.2)}

/* Quiz panel */
#quiz-panel{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(.92);z-index:80;padding:28px 32px;width:460px;max-width:95vw;opacity:0;pointer-events:none;transition:all .28s}
#quiz-panel.visible{opacity:1;transform:translate(-50%,-50%) scale(1);pointer-events:auto}
.qz-title{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:2px;color:#5c94b0;margin-bottom:14px}
.qz-category{font-size:9px;padding:2px 8px;border-radius:999px;background:rgba(255,209,102,.12);color:#ffd166;border:1px solid rgba(255,209,102,.25);display:inline-block;margin-bottom:10px}
.qz-question{font-size:14px;color:#d8f8ff;line-height:1.5;margin-bottom:18px;font-weight:500}
.qz-options{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.qz-opt{border:1px solid rgba(100,200,255,.2);border-radius:10px;padding:10px 12px;text-align:center;font-size:11px;color:#9ddcf0;cursor:pointer;background:rgba(100,200,255,.04);transition:all .15s}
.qz-opt:hover:not(.locked){background:rgba(100,220,255,.12);border-color:rgba(100,220,255,.4);color:#d8fcff}
.qz-opt.correct{background:rgba(61,220,151,.18);border-color:rgba(61,220,151,.5);color:#3ddc97}
.qz-opt.wrong{background:rgba(255,100,100,.12);border-color:rgba(255,100,100,.35);color:#ff9d9d}
.qz-opt.locked{cursor:default}
.qz-feedback{text-align:center;font-size:11px;min-height:20px;margin-bottom:10px}
.qz-feedback.right{color:#3ddc97}
.qz-feedback.wrong{color:#ff7f91}
.qz-stats{display:flex;justify-content:space-between;align-items:center;border-top:1px solid rgba(255,255,255,.07);padding-top:10px}
.qz-score{font-family:'Orbitron',sans-serif;font-size:13px;color:#ffd166}
#quiz-close{border:none;background:rgba(255,255,255,.06);color:#7aacca;border-radius:8px;padding:5px 12px;cursor:pointer;font-size:10px;border:1px solid rgba(255,255,255,.1)}
#quiz-close:hover{background:rgba(255,255,255,.12)}
#quiz-next{border:none;background:rgba(110,243,255,.14);color:#6ef3ff;border-radius:8px;padding:6px 16px;cursor:pointer;font-size:10px;border:1px solid rgba(110,243,255,.28)}
#quiz-next:hover{background:rgba(110,243,255,.25)}
#quiz-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:78;opacity:0;pointer-events:none;transition:opacity .28s}
#quiz-overlay.visible{opacity:1;pointer-events:auto}

/* Achievement badges */
#ach-log-btn{position:fixed;top:14px;right:216px;z-index:50;padding:6px 11px;font-size:10px;cursor:pointer}
#ach-log{position:fixed;top:50px;right:216px;z-index:55;width:280px;max-height:62vh;overflow-y:auto;padding:14px 14px 10px;opacity:0;pointer-events:none;transition:all .2s;scrollbar-width:thin;scrollbar-color:rgba(100,200,255,.25) transparent}
#ach-log::-webkit-scrollbar{width:4px}
#ach-log::-webkit-scrollbar-track{background:rgba(255,255,255,.03);border-radius:2px}
#ach-log::-webkit-scrollbar-thumb{background:rgba(100,200,255,.25);border-radius:2px}
#ach-log::-webkit-scrollbar-thumb:hover{background:rgba(100,200,255,.45)}
#ach-log.visible{opacity:1;pointer-events:auto}
#ach-log .al-title{font-family:'Orbitron',sans-serif;font-size:9px;color:#5c94b0;letter-spacing:1.8px;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(100,200,255,.12);position:sticky;top:0;background:rgba(8,18,38,.92);z-index:1;backdrop-filter:blur(8px)}
.al-item{display:flex;gap:9px;align-items:center;padding:7px 8px;border-radius:9px;margin-bottom:3px;font-size:10px;color:#8abed5;transition:background .12s}
.al-item:hover{background:rgba(100,200,255,.05)}
.al-item.unlocked{background:rgba(255,209,102,.05);border:1px solid rgba(255,209,102,.12)}
.al-item .al-icon{font-size:18px;flex-shrink:0;width:24px;text-align:center}
.al-info{flex:1;min-width:0}
.al-name{color:#ffd166;font-weight:600;margin-bottom:1px}
.al-item.locked .al-name{color:#4e7a8c}
.al-desc{color:#5a8fa6;font-size:9px;line-height:1.3}
.al-item.locked{opacity:.55}

/* Challenge */
#challenge{position:fixed;bottom:70px;right:14px;z-index:50;padding:10px 13px;width:210px;text-align:center}
.ch-label{font-family:'Orbitron',sans-serif;font-size:8px;letter-spacing:1.4px;color:#5c94b0;margin-bottom:4px;text-transform:uppercase}
.ch-text{font-size:10px;color:#9ddcf0;margin-bottom:6px;line-height:1.4}
.ch-btn{border:none;border-radius:8px;padding:5px 11px;font-size:10px;cursor:pointer;background:rgba(255,209,102,.14);color:#ffd166;border:1px solid rgba(255,209,102,.28)}
.ch-btn:hover{background:rgba(255,209,102,.26)}

/* Legend */
#legend{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);z-index:48;display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:8px 14px;max-width:92vw}
.legend-title{font-size:9px;letter-spacing:1.4px;text-transform:uppercase;color:#5c94b0;font-family:'Orbitron',sans-serif}
.legend-item{display:flex;align-items:center;gap:5px;padding:3px 9px;border-radius:999px;border:1px solid transparent;font-size:10px;color:#7ab0c8;cursor:pointer;user-select:none;transition:all .14s}
.legend-item:hover{background:rgba(255,255,255,.05)}
.legend-item.active{background:rgba(110,243,255,.16);border-color:rgba(110,243,255,.4);color:#d8fcff}
.legend-dot{width:8px;height:8px;border-radius:50%}

/* Toast */
#achievement{position:fixed;top:50%;left:50%;transform:translate(-50%,-54%) scale(.82);z-index:100;padding:20px 32px;border-radius:18px;text-align:center;opacity:0;pointer-events:none;transition:all .38s}
#achievement.show{opacity:1;transform:translate(-50%,-54%) scale(1)}
#achievement .ach-icon{font-size:36px;margin-bottom:5px}
#achievement .ach-title{font-family:'Orbitron',sans-serif;font-size:13px;color:#ffd166;letter-spacing:1px;margin-bottom:3px}
#achievement .ach-desc{font-size:11px;color:#9ddcf0}

/* Nodes/links */
.sim-link{stroke-linecap:round}
.node{cursor:pointer;transition:opacity .18s}
.node circle.core{stroke-width:1.4}
.node circle.aura{stroke:none}
.node.discovered circle.core{stroke-width:2;filter:url(#glow)}
.node.dimmed{opacity:.1}
.node.filtered-out{opacity:.05}
.node.selected circle.core{stroke:#ffd166;stroke-width:2.4;filter:url(#selGlow)}
.node.compare-pick circle.core{stroke:#a78bfa;stroke-width:2.2;filter:url(#selGlow)}
.node.highlight circle.core{stroke:#6ef3ff;stroke-width:2.2}
.node-label{font-size:9px;fill:#6eaac5;pointer-events:none;font-family:'Inter',sans-serif}
@keyframes pulse-ring{0%{r:6;opacity:.5}100%{r:22;opacity:0}}
.pulse-ring{fill:none;stroke:#6ef3ff;stroke-width:1.2;animation:pulse-ring .8s ease-out forwards}
</style>
</head>
<body>
<div id="bg"></div>
<canvas id="star-bg"></canvas>

<!-- HUD -->
<div id="hud">
  <div class="glass" id="search-wrap">
    <input id="search" placeholder="Search dish, ingredient, cuisine…" autocomplete="off">
    <button id="search-clear">✕</button>
  </div>
  <div class="glass" id="controls">
    <button class="ctrl" id="btn-fit">Reset</button>
    <button class="ctrl" id="btn-shake">Shake</button>
    <button class="ctrl" id="btn-labels">Labels</button>
    <button class="ctrl" id="btn-compare">⚖ Compare</button>
    <button class="ctrl" id="btn-quiz">🧠 Quiz</button>
  </div>
</div>

<!-- XP HUD -->
<div class="glass" id="score-hud">
  <div class="xp-title">Explorer Progress</div>
  <div id="xp-row"><span id="xp-num">0</span><span id="xp-total-lbl">/ 0 discovered</span></div>
  <div id="xp-bar-wrap"><div id="xp-bar"></div></div>
  <div id="xp-msg">Click a dish node to explore!</div>
  <div id="xp-rank"></div>
  <div id="streak"></div>
</div>

<!-- Badge log -->
<button class="glass ctrl" id="ach-log-btn">🏅 Badges</button>
<div class="glass" id="ach-log"><div class="al-title">Achievements</div><div id="ach-list"></div></div>

<!-- Tooltip -->
<div id="tooltip" class="glass"></div>

<!-- Canvas -->
<svg id="canvas"></svg>

<!-- Detail panel -->
<div id="info-panel" class="glass">
  <button id="close-panel">✕</button>
  <div class="dish-name" id="dish-name"></div>
  <div class="cuisine-badge" id="dish-cuisine"></div>
  <div id="panel-meta"></div>
  <div id="panel-tabs"></div>
  <div id="panel-body"></div>
</div>

<!-- Compare panel -->
<div id="compare-panel" class="glass">
  <div class="cp-title">⚖ Compare Mode</div>
  <div class="cp-vs">
    <div class="cp-dish" id="cp-a">Pick 1st dish</div>
    <span class="cp-spark">VS</span>
    <div class="cp-dish" id="cp-b">Pick 2nd dish</div>
  </div>
  <div id="cp-score-wrap"></div>
  <div id="cp-deep"></div>
  <div class="cp-actions"><button class="cp-btn clear" id="cp-clear">Clear</button></div>
</div>

<!-- Quiz -->
<div id="quiz-overlay"></div>
<div id="quiz-panel" class="glass">
  <div class="qz-title">🧠 Biryani Quiz</div>
  <div id="qz-cat" class="qz-category"></div>
  <div id="qz-q" class="qz-question"></div>
  <div id="qz-opts" class="qz-options"></div>
  <div id="qz-feedback" class="qz-feedback"></div>
  <div class="qz-stats">
    <div class="qz-score">Score: <span id="qz-score">0</span> / <span id="qz-total">0</span></div>
    <div style="display:flex;gap:8px">
      <button id="quiz-next" style="display:none">Next →</button>
      <button id="quiz-close">Close</button>
    </div>
  </div>
</div>

<!-- Challenge -->
<div class="glass" id="challenge">
  <div class="ch-label">Challenge</div>
  <div class="ch-text" id="ch-text">Loading…</div>
  <button class="ch-btn" id="ch-btn" style="display:none">✓ Claim</button>
</div>

<!-- Legend -->
<div id="legend" class="glass"><span class="legend-title">Cuisines</span></div>

<!-- Toast -->
<div id="achievement" class="glass">
  <div class="ach-icon" id="ach-icon"></div>
  <div class="ach-title" id="ach-title"></div>
  <div class="ach-desc" id="ach-desc"></div>
</div>

<script>
const DATA="""
        + data_json
        + """;

/* ── Star-field ── */
(()=>{
  const c=document.getElementById('star-bg'),x=c.getContext('2d');
  c.width=window.innerWidth;c.height=window.innerHeight;
  const S=Array.from({length:200},()=>({x:Math.random()*c.width,y:Math.random()*c.height,r:Math.random()*1.2+.3,a:Math.random()*6}));
  (function draw(){
    x.clearRect(0,0,c.width,c.height);
    S.forEach(s=>{s.a+=.003+Math.random()*.002;const o=.12+Math.abs(Math.sin(s.a))*.42;x.beginPath();x.arc(s.x,s.y,s.r,0,Math.PI*2);x.fillStyle=`rgba(180,220,255,${o})`;x.fill();});
    requestAnimationFrame(draw);
  })();
  window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight;});
})();

const W=window.innerWidth, H=window.innerHeight;
const svg=d3.select('#canvas').attr('width',W).attr('height',H);
const defs=svg.append('defs');
['glow','selGlow'].forEach((id,i)=>{
  const f=defs.append('filter').attr('id',id).attr('x','-60%').attr('y','-60%').attr('width','220%').attr('height','220%');
  f.append('feGaussianBlur').attr('stdDeviation',i?3.5:2.5).attr('result','b');
  const m=f.append('feMerge');m.append('feMergeNode').attr('in','b');m.append('feMergeNode').attr('in','SourceGraphic');
});
const lf=defs.append('filter').attr('id','lGlow');
lf.append('feGaussianBlur').attr('stdDeviation',1.4).attr('result','b');
const lm=lf.append('feMerge');lm.append('feMergeNode').attr('in','b');lm.append('feMergeNode').attr('in','SourceGraphic');

const mainG=svg.append('g'), lkG=mainG.append('g'), ndG=mainG.append('g'), lbG=mainG.append('g'), fxG=mainG.append('g');

const $=id=>document.getElementById(id);
const esc=v=>String(v==null?'':v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

/* colour palette */
const PALETTE=['#ff7f91','#64b5ff','#ffd166','#b388ff','#3ddc97','#ff9f40','#36d6d0','#8ee24a','#f06292','#29b6f6','#ab47bc','#fdd835'];
const C={};
(DATA.cuisines||[]).forEach((c,i)=>C[c]=PALETTE[i%PALETTE.length]);
const catGrad=['#3de8ff','#a78bfa','#ffd166','#3ddc97','#ff9f40'];

/* nodes */
const nodes=(DATA.nodes||[]).map((n,i)=>({
  ...n,index:i,dish:DATA.dishInfo[i]||{},
  x:W/2+(n.x||0)*14+(Math.random()-.5)*30,
  y:H/2+(n.y||0)*14+(Math.random()-.5)*30,
  fx:null,fy:null
}));

/* links */
const lkSet=new Set(), simLinks=[];
function pushLink(a,b,sim,extra){
  if(a===b)return;const k=a<b?a+'-'+b:b+'-'+a;
  if(lkSet.has(k))return;lkSet.add(k);
  simLinks.push({source:a,target:b,sim:+sim||0,...(extra||{})});
}
const pairingMap={};
(DATA.pairings||[]).forEach(p=>{
  pushLink(p.a,p.b,p.sim,{sIng:p.sIng,sFlav1:p.sFlav1,sFlav2:p.sFlav2,sCook:p.sCook,sCmpd:p.sCmpd,sCuis:p.sCuis,sTime:p.sTime});
  const k=p.a<p.b?p.a+'-'+p.b:p.b+'-'+p.a;pairingMap[k]=p;
});
nodes.forEach((n,i)=>(n.dish.topSimilar||[]).forEach(s=>pushLink(i,+s.id,+s.score)));

const simExt=d3.extent(simLinks,d=>d.sim);
const opSc=d3.scaleLinear().domain(simExt).range([.008,.045]);
const wSc=d3.scaleLinear().domain(simExt).range([.3,1.1]);

/* render links */
const lkSel=lkG.selectAll('line').data(simLinks).enter().append('line')
  .attr('class','sim-link').attr('stroke','#4a7a9e')
  .attr('stroke-opacity',d=>opSc(d.sim)).attr('stroke-width',d=>wSc(d.sim))
  .attr('filter','url(#lGlow)');

/* render nodes */
const ndSel=ndG.selectAll('g.node').data(nodes).enter().append('g').attr('class','node');
ndSel.append('circle').attr('class','aura').attr('r',12).attr('fill',d=>C[d.cuisine]||'#9ca3af').attr('fill-opacity',.08);
ndSel.append('circle').attr('class','core').attr('r',d=>4.5+((d.cluster||0)%3)*.6)
  .attr('fill',d=>C[d.cuisine]||'#9ca3af').attr('fill-opacity',.92).attr('stroke','rgba(255,255,255,.2)');

let showLbls=false;
const lbSel=lbG.selectAll('text').data(nodes).enter().append('text')
  .attr('class','node-label').style('display','none').text(d=>d.name.length>20?d.name.slice(0,18)+'…':d.name);

/* zoom */
const zoom=d3.zoom().scaleExtent([.2,6]).on('zoom',ev=>mainG.attr('transform',ev.transform));
svg.call(zoom).on('dblclick.zoom',null);

/* simulation */
const sim=d3.forceSimulation(nodes)
  .force('link',d3.forceLink(simLinks).id(d=>d.index).distance(d=>Math.max(28,130-d.sim*90)).strength(d=>.12+d.sim*.45))
  .force('charge',d3.forceManyBody().strength(-55))
  .force('collide',d3.forceCollide().radius(14).iterations(2))
  .force('center',d3.forceCenter(W/2,H/2).strength(.03))
  .alpha(.85).alphaDecay(.022).on('tick',tick);

function tick(){
  lkSel.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  ndSel.attr('transform',d=>`translate(${d.x},${d.y})`);
  lbSel.attr('x',d=>d.x+9).attr('y',d=>d.y-9).style('display',showLbls?'block':'none');
}

ndSel.call(d3.drag()
  .on('start',(ev,d)=>{if(!ev.active)sim.alphaTarget(.15).restart();d.fx=d.x;d.fy=d.y;})
  .on('drag',(ev,d)=>{d.fx=ev.x;d.fy=ev.y;})
  .on('end',(ev,d)=>{if(!ev.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));

/* ── Game state ── */
const discovered=new Set(),clusterDone={};
(DATA.hierarchy||[]).forEach(h=>{clusterDone[h.clusterLabel||'']=new Set();});
let selectedNode=null,searchQ='';
const cuisineFilters=new Set();
let compareMode=false,cmpA=null,cmpB=null;
let cuisinesExp=new Set(),compareCnt=0,streakCnt=0,lastDisc=0;
const achUnlocked=new Set();
let tabCounts={Ingredients:0,Flavour:0,Science:0,Synergies:0,Steps:0,Compounds:0};

/* ranks */
const RANKS=[
  {min:0,name:'Novice',col:'#5e8faa',bg:'rgba(94,143,170,.2)'},
  {min:5,name:'Taster',col:'#3ddc97',bg:'rgba(61,220,151,.15)'},
  {min:15,name:'Apprentice',col:'#64b5ff',bg:'rgba(100,181,255,.15)'},
  {min:30,name:'Explorer',col:'#a78bfa',bg:'rgba(167,139,250,.15)'},
  {min:50,name:'Connoisseur',col:'#ffd166',bg:'rgba(255,209,102,.15)'},
  {min:75,name:'Master Chef',col:'#ff7f91',bg:'rgba(255,127,145,.15)'},
  {min:90,name:'Biryani Legend',col:'#6ef3ff',bg:'rgba(110,243,255,.2)'}
];
function gRank(n){let r=RANKS[0];for(const rk of RANKS)if(n>=rk.min)r=rk;return r;}

/* achievements */
const ACHS=[
  {id:'first',icon:'🌱',name:'First Bite',desc:'Discover your first dish',check:()=>discovered.size>=1},
  {id:'five',icon:'⭐',name:'Curious Taster',desc:'Discover 5 dishes',check:()=>discovered.size>=5},
  {id:'ten',icon:'🌟',name:'Explorer',desc:'Discover 10 dishes',check:()=>discovered.size>=10},
  {id:'q25',icon:'🔥',name:'Quarter Way',desc:'25% of all dishes found',check:()=>discovered.size>=Math.ceil(nodes.length*.25)},
  {id:'half',icon:'💫',name:'Halfway',desc:'50% of all dishes found',check:()=>discovered.size>=Math.ceil(nodes.length*.5)},
  {id:'q75',icon:'🚀',name:'Almost Legend',desc:'75% of all dishes found',check:()=>discovered.size>=Math.ceil(nodes.length*.75)},
  {id:'all',icon:'🏆',name:'Biryani Legend',desc:'Discovered every dish!',check:()=>discovered.size>=nodes.length},
  {id:'c3',icon:'🌍',name:'World Traveler',desc:'Explore 3+ cuisines',check:()=>cuisinesExp.size>=3},
  {id:'call',icon:'🗺️',name:'Global Palate',desc:'Explore every cuisine',check:()=>cuisinesExp.size>=(DATA.cuisines||[]).length},
  {id:'cl1',icon:'🎯',name:'Cluster Cleared',desc:'Complete one cluster',check:()=>{for(const cl in clusterDone){const t=nodes.filter(n=>(n.clusterLabel||'')===cl).length;if(t>1&&clusterDone[cl].size>=t)return true;}return false;}},
  {id:'cl3',icon:'🎯🎯',name:'Triple Cluster',desc:'Complete 3 clusters',check:()=>{let c=0;for(const cl in clusterDone){const t=nodes.filter(n=>(n.clusterLabel||'')===cl).length;if(t>1&&clusterDone[cl].size>=t)c++;}return c>=3;}},
  {id:'cmp1',icon:'⚖️',name:'First Duel',desc:'Compare two dishes',check:()=>compareCnt>=1},
  {id:'cmp5',icon:'🔬',name:'Compare Guru',desc:'Compare 5 pairs',check:()=>compareCnt>=5},
  {id:'streak',icon:'⚡',name:'Hot Streak',desc:'5 rapid discoveries',check:()=>streakCnt>=5},
  {id:'ing10',icon:'📋',name:'Ingredient Inspector',desc:'View Ingredients tab 10x',check:()=>tabCounts.Ingredients>=10},
  {id:'sci5',icon:'🔭',name:'Food Scientist',desc:'View Science tab 5x',check:()=>tabCounts.Science>=5},
  {id:'syn3',icon:'⚗️',name:'Synergy Hunter',desc:'View Synergies tab 3x',check:()=>tabCounts.Synergies>=3},
  {id:'quiz5',icon:'🧠',name:'Quiz Whiz',desc:'Answer 5 quiz questions correctly',check:()=>quizCorrect>=5},
];

let quizCorrect=0;

/* XP */
const xpN=$('xp-num'),xpMsg=$('xp-msg'),xpBar=$('xp-bar'),xpRank=$('xp-rank'),strEl=$('streak'),xpTl=$('xp-total-lbl');
xpTl.textContent='/ '+nodes.length+' discovered';

function updateXP(){
  const n=discovered.size;
  xpN.textContent=n;
  xpBar.style.width=Math.min(100,(n/nodes.length)*100)+'%';
  const p=Math.round((n/nodes.length)*100);
  xpMsg.textContent=p===0?'Click nodes to explore!':p<25?'Keep exploring!':p<50?'Nice progress!':p<75?'Over halfway!':p<100?'Almost legendary!':'LEGENDARY! 🏆';
  const rk=gRank(n);
  xpRank.innerHTML=`<span class="rank-badge" style="background:${rk.bg};color:${rk.col};border:1px solid ${rk.col}44">${rk.name}</span>`;
  strEl.textContent=streakCnt>=2?`⚡ Streak: ${streakCnt}`:'';
  checkAchs();renderAchLog();
}

function showToast(icon,title,desc){
  const el=$('achievement');
  $('ach-icon').textContent=icon;$('ach-title').textContent=title;$('ach-desc').textContent=desc;
  el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2800);
}
function checkAchs(){ACHS.forEach(a=>{if(!achUnlocked.has(a.id)&&a.check()){achUnlocked.add(a.id);showToast(a.icon,a.name,a.desc);}});}
function renderAchLog(){
  $('ach-list').innerHTML=ACHS.map(a=>{
    const u=achUnlocked.has(a.id);
    return `<div class="al-item ${u?'unlocked':'locked'}"><span class="al-icon">${u?a.icon:'🔒'}</span><div class="al-info"><div class="al-name">${a.name}</div><div class="al-desc">${a.desc}</div></div></div>`;
  }).join('');
}

let achLogOpen=false;
$('ach-log-btn').onclick=()=>{achLogOpen=!achLogOpen;$('ach-log').classList.toggle('visible',achLogOpen);};

function clusterTrack(node){
  const cl=node.clusterLabel||'';
  if(!clusterDone[cl])clusterDone[cl]=new Set();
  clusterDone[cl].add(node.index);
  cuisinesExp.add(node.cuisine||'Unknown');
}
function streakTick(){
  const now=Date.now();
  if(lastDisc&&(now-lastDisc)<3000)streakCnt++;else streakCnt=1;
  lastDisc=now;
}

/* ── Visual state ── */
let hovNode=null;
function applyVis(){
  const focus=selectedNode||hovNode;
  let sSet=null;
  if(focus){sSet=new Set([focus.index]);((focus.dish||{}).topSimilar||[]).forEach(s=>sSet.add(+s.id));}
  ndSel
    .classed('selected',d=>selectedNode&&d.index===selectedNode.index)
    .classed('discovered',d=>discovered.has(d.index))
    .classed('compare-pick',d=>(cmpA&&d.index===cmpA.index)||(cmpB&&d.index===cmpB.index))
    .classed('highlight',d=>searchQ&&matchSearch(d))
    .classed('filtered-out',d=>!matchCuisine(d))
    .classed('dimmed',d=>{
      if(!matchCuisine(d))return true;
      if(searchQ&&!matchSearch(d))return true;
      if(sSet&&!sSet.has(d.index))return true;
      return false;
    });
  lkSel
    .attr('stroke-opacity',l=>{
      const a=l.source,b=l.target;
      if(!matchCuisine(a)||!matchCuisine(b))return 0;
      if(searchQ&&(!matchSearch(a)||!matchSearch(b)))return 0;
      if(focus){
        if(a.index===focus.index||b.index===focus.index)return Math.max(.35,opSc(l.sim)*8);
        return .008;
      }
      return opSc(l.sim);
    })
    .attr('stroke-width',l=>{
      if(focus&&(l.source.index===focus.index||l.target.index===focus.index))return Math.max(1.2,wSc(l.sim)*1.5);
      return wSc(l.sim);
    })
    .attr('stroke',l=>{
      if(focus&&(l.source.index===focus.index||l.target.index===focus.index))return '#6ef3ff';
      return '#4a7a9e';
    });
}
function matchSearch(n){
  if(!searchQ)return true;
  const q=searchQ.toLowerCase(),d=n.dish||{};
  return n.name.toLowerCase().includes(q)||(n.cuisine||'').toLowerCase().includes(q)||(d.ingredients||[]).some(i=>String(i).toLowerCase().includes(q));
}
function matchCuisine(n){if(!cuisineFilters.size)return true;return cuisineFilters.has(n.cuisine||'Unknown');}

/* ── Tooltip ── */
const tip=$('tooltip');
ndSel
  .on('mouseenter',(ev,d)=>{
    hovNode=d;applyVis();
    const di=d.dish||{};
    let h=`<div class="tt-name">${esc(d.name)}</div><div class="tt-row"><span>${esc(d.cuisine||'Unknown')}</span><span>${di.category||''}</span></div>`;
    if(di.totalTime) h+=`<div class="tt-row"><span>Time</span><span>${esc(di.totalTime)}</span></div>`;
    if(di.serves) h+=`<div class="tt-row"><span>Serves</span><span>${esc(di.serves)}</span></div>`;
    const ic=(di.ingredients||[]).length;
    if(ic) h+=`<div class="tt-row"><span>Ingredients</span><span>${ic}</span></div>`;
    const sc=di.science||{};
    if(sc.maillardPct!=null) h+=`<div class="tt-row"><span>Maillard react.</span><span>${sc.maillardPct}%</span></div>`;
    h+=`<div class="tt-hint">${discovered.has(d.index)?'✓ Discovered':'Click to discover'}</div>`;
    tip.innerHTML=h;tip.style.opacity='1';
  })
  .on('mousemove',ev=>{tip.style.left=Math.min(ev.clientX+14,W-260)+'px';tip.style.top=Math.min(ev.clientY+14,H-150)+'px';})
  .on('mouseleave',()=>{hovNode=null;tip.style.opacity='0';applyVis();});

/* ── Radar/Spider chart (canvas-less, pure SVG) ── */
function buildRadar(tasteCats){
  const cats=['Flavour','Mouthfeel','Smell','Sound','Visual'];
  const vals=cats.map(cat=>{
    const items=(tasteCats||{})[cat]||[];
    if(!items.length)return 0;
    return Math.min(1,(items.reduce((s,i)=>s+i.i,0)/items.length)/10);
  });
  const R=70, cx=90, cy=90, N=cats.length;
  const ang=i=>(Math.PI*2*i/N)-Math.PI/2;
  const pt=(r,i)=>[cx+r*Math.cos(ang(i)),cy+r*Math.sin(ang(i))];
  const rings=[.25,.5,.75,1];
  let s=`<svg width="180" height="180" viewBox="0 0 180 180">`;
  // rings
  rings.forEach(r=>{
    const pts=cats.map((_,i)=>pt(R*r,i).join(',')).join(' ');
    s+=`<polygon points="${pts}" fill="none" stroke="rgba(100,200,255,.12)" stroke-width="1"/>`;
  });
  // axes
  cats.forEach((_,i)=>{
    const [x,y]=pt(R,i);s+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(100,200,255,.18)" stroke-width="1"/>`;
  });
  // data
  const dPts=cats.map((_,i)=>pt(R*vals[i],i).join(',')).join(' ');
  s+=`<polygon points="${dPts}" fill="rgba(110,243,255,.15)" stroke="#6ef3ff" stroke-width="1.5"/>`;
  // labels
  cats.forEach((c,i)=>{
    const [x,y]=pt(R+14,i);
    s+=`<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="middle" fill="${catGrad[i]}" font-size="8" font-family="Inter,sans-serif">${c}</text>`;
  });
  // dots
  cats.forEach((_,i)=>{
    const [x,y]=pt(R*vals[i],i);
    s+=`<circle cx="${x}" cy="${y}" r="3" fill="#6ef3ff"/>`;
  });
  s+='</svg>';
  return s;
}

/* ── Panel tabs ── */
const TABS=['Ingredients','Flavour','Science','Synergies','Steps','Compounds','Style','Similar'];

function renderTab(dish,tab){
  if(tab==='Ingredients'){
    const list=(dish.ingredients||[]);
    if(!list.length)return '<p class="muted">No ingredients listed.</p>';
    // group by origin if available
    const orMap={};(dish.origins||[]).forEach(o=>orMap[o.ingredient.toLowerCase()]=o.origin);
    let h='<ul class="list bullet">';
    list.forEach(i=>{
      const nm=String(i).toLowerCase().trim().split(' ').slice(0,3).join(' ');
      const org=orMap[nm]||Object.keys(orMap).find(k=>nm.startsWith(k))||'';
      h+=`<li>${esc(i)}${org?`<span style="margin-left:6px;font-size:9px;color:#a78bfa;opacity:.7">${esc(org)}</span>`:''}</li>`;
    });
    return h+'</ul>';
  }
  if(tab==='Flavour'){
    const tc=dish.tasteCats||{};
    const cats=['Flavour','Mouthfeel','Smell','Sound','Visual'];
    const hasData=cats.some(c=>(tc[c]||[]).length>0);
    if(!hasData){
      const list=(dish.flavour||[]);
      if(!list.length)return '<p class="muted">No flavour data.</p>';
      const mx=Math.max(1,...list.map(f=>+f.intensity||0));
      return list.map(f=>{const v=+f.intensity||0,p=(v/mx*100).toFixed(0);
        return `<div class="flavour"><div class="fname">${esc(f.descriptor)}</div><div class="fbar"><div class="ffill" style="width:${p}%;background:linear-gradient(90deg,#3de8ff,#a78bfa)"></div></div><div class="fval">${v.toFixed(2)}</div></div>`;
      }).join('');
    }
    let h=`<div id="radar-wrap">${buildRadar(tc)}</div>`;
    cats.forEach(cat=>{
      const items=tc[cat]||[];
      if(!items.length)return;
      const mx=Math.max(1,...items.map(f=>f.i||0));
      h+=`<div class="cat-label">${cat}</div>`;
      items.slice(0,8).forEach(f=>{
        const p=(f.i/mx*100).toFixed(0);
        h+=`<div class="flavour"><div class="fname">${esc(f.d)}</div><div class="fbar"><div class="ffill" style="width:${p}%;background:linear-gradient(90deg,#3de8ff,#a78bfa)"></div></div><div class="fval">${(f.i||0).toFixed(2)}</div></div>`;
      });
    });
    return h;
  }
  if(tab==='Science'){
    const sc=dish.science||{};
    const hasNutr=sc.fat!=null||sc.protein!=null||sc.carbs!=null||sc.moisture!=null;
    let h='';
    if(hasNutr){
      h+=`<div class="section-head">Nutritional Profile (avg per ingredient)</div>`;
      [['fat','Fat','#ffd166'],['protein','Protein','#3ddc97'],['carbs','Carbs','#64b5ff'],['moisture','Moisture','#36d6d0']].forEach(([k,lbl,col])=>{
        if(sc[k]==null)return;
        h+=`<div class="sci-bar-row"><div class="sci-label">${lbl}</div><div class="sci-bar"><div class="sci-fill" style="width:${Math.min(100,sc[k])}%;background:${col}55;border-right:2px solid ${col}"></div></div><div class="sci-val">${sc[k]}%</div></div>`;
      });
    }
    if(sc.maillardPct!=null){
      h+=`<div class="sci-stat-grid"><div class="sci-stat"><div class="sci-stat-val">${sc.maillardPct}%</div><div class="sci-stat-lbl">Maillard Reactive</div></div>${(dish.compounds||[]).length?`<div class="sci-stat"><div class="sci-stat-val">${(dish.compounds||[]).length}</div><div class="sci-stat-lbl">Compounds</div></div>`:''}</div>`;
    }
    const origs=dish.origins||[];
    if(origs.length){
      h+=`<div class="section-head" style="margin-top:10px">Ingredient Origins</div>`;
      origs.slice(0,16).forEach(o=>{
        h+=`<div class="origin-row"><span>${esc(o.ingredient)}</span><span class="origin-tag">${esc(o.origin)}</span></div>`;
      });
    }
    if(!h)h='<p class="muted">No science data available.</p>';
    return h;
  }
  if(tab==='Synergies'){
    const syns=dish.synergies||[];
    if(!syns.length)return '<p class="muted">No active synergies detected.</p>';
    return syns.map(s=>{
      const tc=s.type||'';
      return `<div class="syn-card"><div class="syn-header"><div class="syn-name">${esc(s.name)}</div><span class="syn-factor">${s.factor>0?'×'+s.factor:s.factor}</span></div><div style="margin-bottom:4px"><span class="syn-type ${tc}">${tc}</span></div><div class="syn-affects">Affects: ${esc(s.affects||'—')}</div><div class="syn-mechanism">${esc(s.mechanism||'')}</div></div>`;
    }).join('');
  }
  if(tab==='Steps'){
    const list=dish.steps||[];
    if(!list.length)return '<p class="muted">No cooking steps listed.</p>';
    return list.map(s=>`<div class="step" data-step="${esc(s.step||'•')}">${esc(s.desc||'')}${s.style?`<div class="tiny">${esc(s.style)}${s.dur?' · '+esc(s.dur)+' min':''}</div>`:''}</div>`).join('');
  }
  if(tab==='Compounds'){
    const list=(dish.compounds||[]).slice(0,20);
    if(!list.length)return '<p class="muted">No compounds found.</p>';
    return '<ul class="list">'+list.map(c=>`<li>${esc(c.compound||'Unknown')}${c.ingredient?` <span class="tiny">(${esc(c.ingredient)})</span>`:''}${c.conc!=null?` <span class="tiny">${esc(c.conc)}%</span>`:''}</li>`).join('')+'</ul>';
  }
  if(tab==='Style'){
    const styles=dish.styleDetails||[];
    const si=dish.stepIngredients||[];
    let h='';
    if(dish.prepTime||dish.marinationTime){
      h+=`<div class="section-head">Timings</div>`;
      if(dish.prepTime) h+=`<div class="origin-row"><span>Prep</span><span class="chip">${esc(dish.prepTime)}</span></div>`;
      if(dish.marinationTime) h+=`<div class="origin-row"><span>Marination</span><span class="chip">${esc(dish.marinationTime)}</span></div>`;
    }
    if(styles.length){
      h+=`<div class="section-head" style="margin-top:8px">Cooking Methods</div><ul class="list">`;
      styles.forEach(s=>{h+=`<li><strong>${esc(s.style||'')}</strong>${s.temp?`<div class="tiny">Temp: ${esc(s.temp)}</div>`:''} ${s.browning?`<div class="tiny">Browning: ${esc(s.browning)}</div>`:''}</li>`;});
      h+='</ul>';
    }
    if(si.length){
      const grp={};si.forEach(r=>{const k=String(r.step||'0');if(!grp[k])grp[k]=[];grp[k].push(r);});
      h+=`<div class="section-head" style="margin-top:8px">Step Ingredients</div>`;
      Object.keys(grp).sort((a,b)=>+a - +b).forEach(step=>{
        const ln=grp[step].map(r=>{let t=esc(r.ingredient||'');if(r.quantity)t+=` (${esc(r.quantity)})`;if(r.purpose)t+=' – '+esc(r.purpose);return t;}).join(', ');
        h+=`<div class="step" data-step="${esc(step)}">${ln}</div>`;
      });
    }
    return h||'<p class="muted">No style data.</p>';
  }
  // Similar
  const sim=(dish.topSimilar||[]).slice(0,8);
  if(!sim.length)return '<p class="muted">No similarity data.</p>';
  return sim.map(s=>{
    const k=Math.min(s.id,dish._idx)<Math.max(s.id,dish._idx)?Math.min(s.id,dish._idx)+'-'+Math.max(s.id,dish._idx):Math.max(s.id,dish._idx)+'-'+Math.min(s.id,dish._idx);
    const pm=pairingMap[k]||{};
    const subPills=[['Ingredients','sIng'],['Flavour','sFlav1'],['Cooking','sCook'],['Compounds','sCmpd']].filter(([,kk])=>pm[kk]!=null).map(([lbl,kk])=>`<span class="sub-pill">${lbl} ${(pm[kk]*100).toFixed(0)}%</span>`).join('');
    return `<div class="sim-item" onclick="jumpTo(${s.id})"><div><div class="sim-name">${esc(s.name)}</div>${subPills?`<div class="sub-score-row">${subPills}</div>`:''}</div><span class="sim-score-pill">${(+s.score*100).toFixed(1)}%</span></div>`;
  }).join('');
}

window.jumpTo=function(idx){
  const n=nodes[idx];if(n)selectNode(n);
};

function openPanel(node){
  const d=node.dish||{};d._idx=node.index;
  const col=C[node.cuisine]||'#9ca3af';
  $('dish-name').textContent=node.name;
  const cb=$('dish-cuisine');cb.textContent=node.cuisine||'Unknown';
  cb.style.cssText=`background:${col}22;color:${col};border:1px solid ${col}55`;
  const chips=[];
  if(d.category)chips.push(`<span class="chip">${esc(d.category)}</span>`);
  if(d.totalTime)chips.push(`<span class="chip">⏱ ${esc(d.totalTime)}</span>`);
  if(d.serves)chips.push(`<span class="chip">👥 ${esc(d.serves)}</span>`);
  if(node.clusterLabel)chips.push(`<span class="chip">🔵 ${esc(node.clusterLabel)}</span>`);
  if(d.prepTime)chips.push(`<span class="chip">🔪 ${esc(d.prepTime)}</span>`);
  if(d.marinationTime)chips.push(`<span class="chip">🫙 ${esc(d.marinationTime)}</span>`);
  $('panel-meta').innerHTML=chips.join('');
  $('panel-tabs').innerHTML=TABS.map((t,i)=>`<button class="tab-btn${i===0?' active':''}" data-tab="${t}">${t}</button>`).join('');
  $('panel-body').innerHTML=TABS.map((t,i)=>`<div class="tab${i===0?' show':''}" data-tab="${t}">${renderTab(d,t)}</div>`).join('');
  tabCounts.Ingredients++;
  $('panel-tabs').querySelectorAll('.tab-btn').forEach(btn=>{
    btn.onclick=()=>{
      $('panel-tabs').querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b===btn));
      $('panel-body').querySelectorAll('.tab').forEach(p=>p.classList.toggle('show',p.dataset.tab===btn.dataset.tab));
      if(tabCounts[btn.dataset.tab]!=null)tabCounts[btn.dataset.tab]++;
      checkAchs();
    };
  });
  $('info-panel').classList.add('visible');
}

/* pulse FX */
function pulse(x,y,col){
  const c=fxG.append('circle').attr('class','pulse-ring').attr('cx',x).attr('cy',y).attr('r',6).attr('stroke',col||'#6ef3ff');
  setTimeout(()=>c.remove(),900);
}

/* select node */
function selectNode(node){
  if(compareMode)return handleCmp(node);
  selectedNode=node;
  const wasNew=!discovered.has(node.index);
  discovered.add(node.index);clusterTrack(node);
  if(wasNew){streakTick();pulse(node.x,node.y,C[node.cuisine]||'#6ef3ff');updateXP();}
  openPanel(node);applyVis();
  checkChallenge(node);
}
ndSel.on('click',(ev,d)=>selectNode(d));
$('close-panel').onclick=()=>{$('info-panel').classList.remove('visible');selectedNode=null;applyVis();};

/* ── Compare ── */
$('btn-compare').onclick=()=>{
  compareMode=!compareMode;
  $('btn-compare').classList.toggle('on',compareMode);
  $('compare-panel').classList.toggle('visible',compareMode);
  if(!compareMode)clearCmp();
};
$('cp-clear').onclick=clearCmp;

function clearCmp(){cmpA=null;cmpB=null;$('cp-a').textContent='Pick 1st dish';$('cp-a').classList.remove('filled');$('cp-b').textContent='Pick 2nd dish';$('cp-b').classList.remove('filled');$('cp-score-wrap').innerHTML='';$('cp-deep').innerHTML='';applyVis();}

function handleCmp(node){
  discovered.add(node.index);clusterTrack(node);streakTick();updateXP();
  pulse(node.x,node.y,C[node.cuisine]||'#a78bfa');
  if(!cmpA){cmpA=node;$('cp-a').textContent=node.name;$('cp-a').classList.add('filled');}
  else if(!cmpB&&node.index!==cmpA.index){cmpB=node;$('cp-b').textContent=node.name;$('cp-b').classList.add('filled');compareCnt++;buildDeepCmp();checkAchs();}
  applyVis();
}

function buildDeepCmp(){
  if(!cmpA||!cmpB)return;
  const dA=cmpA.dish||{},dB=cmpB.dish||{};
  const cA=C[cmpA.cuisine]||'#ff7f91',cB=C[cmpB.cuisine]||'#a78bfa';
  const k=Math.min(cmpA.index,cmpB.index)+'-'+Math.max(cmpA.index,cmpB.index);
  const pm=pairingMap[k]||{};
  const sv=pm.sim||0,sp=(sv*100).toFixed(1);
  const barCol=sv>.6?'linear-gradient(90deg,#3ddc97,#6ef3ff)':sv>.3?'linear-gradient(90deg,#ffd166,#ff9f40)':'linear-gradient(90deg,#ff7f91,#ff5555)';
  $('cp-score-wrap').innerHTML=`<div class="cp-score-label" style="color:${sv>.5?'#3ddc97':'#ffd166'}">${sp}%</div><div class="cp-score-bar"><div class="cp-score-fill" style="width:${sp}%;background:${barCol}"></div></div><div class="cp-score-text">${sv>.7?'Very similar!':sv>.4?'Moderately similar':sv>.1?'Quite different':'Distant cousins'}</div>`;

  let h='';

  // sub-similarity bars
  const subKeys=[['Ingredient','sIng','#ffd166'],['Flavour','sFlav1','#3ddc97'],['Flavour2','sFlav2','#64b5ff'],['Cooking','sCook','#ff9f40'],['Compounds','sCmpd','#a78bfa'],['Cuisine','sCuis','#ff7f91'],['Cook Time','sTime','#36d6d0']];
  const hasSub=subKeys.some(([,k2])=>pm[k2]!=null);
  if(hasSub){
    h+=`<div class="cp-section"><div class="cp-section-title">Similarity Breakdown</div>`;
    subKeys.forEach(([lbl,k2,col])=>{
      if(pm[k2]==null)return;
      const v=(pm[k2]*100).toFixed(0);
      h+=`<div class="sub-sim-row"><div class="sub-sim-lbl">${lbl}</div><div class="sub-sim-bar"><div class="sub-sim-fill" style="width:${v}%;background:${col}88;border-right:2px solid ${col}"></div></div><div class="sub-sim-val">${v}%</div></div>`;
    });
    h+='</div>';
  }

  // Quick stats
  h+=`<div class="cp-section"><div class="cp-section-title">Stats</div><div class="cp-grid">`;
  h+=`<div class="va" style="color:${cA}">${esc(cmpA.name.split(' ').slice(0,2).join(' '))}</div><div class="lbl">Name</div><div class="vb" style="color:${cB}">${esc(cmpB.name.split(' ').slice(0,2).join(' '))}</div>`;
  [['Cuisine','cuisine'],['Time','totalTime'],['Serves','serves']].forEach(([lbl,k2])=>{h+=`<div class="va">${esc(dA[k2]||'?')}</div><div class="lbl">${lbl}</div><div class="vb">${esc(dB[k2]||'?')}</div>`;});
  h+=`<div class="va">${(dA.ingredients||[]).length}</div><div class="lbl">Ingrts</div><div class="vb">${(dB.ingredients||[]).length}</div>`;
  h+=`<div class="va">${esc(cmpA.clusterLabel||'?')}</div><div class="lbl">Cluster</div><div class="vb">${esc(cmpB.clusterLabel||'?')}</div>`;
  h+='</div></div>';

  // Shared ingredients
  const iA=new Set((dA.ingredients||[]).map(i=>String(i).toLowerCase().trim()));
  const iB=new Set((dB.ingredients||[]).map(i=>String(i).toLowerCase().trim()));
  const shared=[...iA].filter(i=>iB.has(i)),onlyA=[...iA].filter(i=>!iB.has(i)),onlyB=[...iB].filter(i=>!iA.has(i));
  h+=`<div class="cp-section"><div class="cp-section-title">Shared Ingredients (${shared.length})</div><div class="cp-shared">${shared.slice(0,10).map(i=>`<span class="cp-tag">${esc(i)}</span>`).join('')}${!shared.length?'<span style="font-size:10px;color:#5e8faa">None</span>':''}</div></div>`;
  h+=`<div class="cp-section"><div class="cp-section-title" style="color:#ff9daa">Only in ${esc(cmpA.name.split(' ')[0])} (${onlyA.length})</div><div class="cp-shared">${onlyA.slice(0,7).map(i=>`<span class="cp-tag ua">${esc(i)}</span>`).join('')}</div></div>`;
  h+=`<div class="cp-section"><div class="cp-section-title" style="color:#c4b0ff">Only in ${esc(cmpB.name.split(' ')[0])} (${onlyB.length})</div><div class="cp-shared">${onlyB.slice(0,7).map(i=>`<span class="cp-tag ub">${esc(i)}</span>`).join('')}</div></div>`;

  // Flavour comparison
  const fA=dA.flavour||[],fB=dB.flavour||[];
  if(fA.length||fB.length){
    const desc=new Set([...fA,...fB].map(f=>f.descriptor));
    const mA={},mB={};fA.forEach(f=>mA[f.descriptor]=+f.intensity||0);fB.forEach(f=>mB[f.descriptor]=+f.intensity||0);
    const mx=Math.max(1,...Object.values(mA),...Object.values(mB));
    h+=`<div class="cp-section"><div class="cp-section-title">Flavour</div><div style="display:flex;justify-content:space-between;font-size:9px;color:#5e8faa;margin-bottom:4px"><span style="color:${cA}">◆ ${esc(cmpA.name.split(' ')[0])}</span><span style="color:${cB}">◆ ${esc(cmpB.name.split(' ')[0])}</span></div>`;
    [...desc].slice(0,10).forEach(d=>{
      const vA=((mA[d]||0)/mx*100).toFixed(0),vB=((mB[d]||0)/mx*100).toFixed(0);
      h+=`<div class="cp-frow"><div class="fn">${esc(d)}</div><div class="fb"><div class="ffA" style="width:${vA}%"></div><div class="ffB" style="width:${vB}%"></div></div></div>`;
    });
    h+='</div>';
  }

  // Science compare
  const scA=dA.science||{},scB=dB.science||{};
  if(Object.keys(scA).length||Object.keys(scB).length){
    h+=`<div class="cp-section"><div class="cp-section-title">Science</div><div class="cp-grid">`;
    [['Fat%','fat'],['Protein%','protein'],['Carbs%','carbs'],['Maillard','maillardPct']].forEach(([lbl,k2])=>{
      h+=`<div class="va">${scA[k2]!=null?scA[k2]+'%':'—'}</div><div class="lbl">${lbl}</div><div class="vb">${scB[k2]!=null?scB[k2]+'%':'—'}</div>`;
    });
    h+='</div></div>';
  }

  // Shared synergies
  const synA=new Set((dA.synergies||[]).map(s=>s.name.toLowerCase()));
  const synB=new Set((dB.synergies||[]).map(s=>s.name.toLowerCase()));
  const synShared=[...synA].filter(s=>synB.has(s));
  if(synShared.length){
    h+=`<div class="cp-section"><div class="cp-section-title">Shared Synergies</div><div class="cp-shared">${synShared.map(s=>`<span class="cp-tag">${esc(s)}</span>`).join('')}</div></div>`;
  }

  $('cp-deep').innerHTML=h;
}

/* ── Quiz ── */
const QZ_CATS=['Ingredients','Science','Geography','Flavour','Compounds'];
let qzQ=null,qzAnswered=false,qzScore=0,qzTotal=0;

function buildQuizQuestion(){
  const n=nodes.length;
  if(!n)return null;
  const cat=QZ_CATS[Math.floor(Math.random()*QZ_CATS.length)];
  const pick=()=>nodes[Math.floor(Math.random()*n)];
  let q=null;

  if(cat==='Ingredients'){
    const node=pick();const d=node.dish||{};const ings=(d.ingredients||[]);
    if(ings.length<2)return buildQuizQuestion();
    const real=ings[Math.floor(Math.random()*ings.length)];
    const wrongs=[];
    while(wrongs.length<3){
      const o=pick();const oi=(o.dish||{}).ingredients||[];
      const ci=oi[Math.floor(Math.random()*oi.length)];
      if(ci&&ci!==real&&!wrongs.includes(ci))wrongs.push(ci);
    }
    const opts=shuffle([real,...wrongs]);
    return{cat,question:`Which ingredient is used in ${node.name}?`,answer:real,options:opts};
  }
  if(cat==='Science'){
    const node=pick();const sc=(node.dish||{}).science||{};
    const keys=Object.keys(sc).filter(k=>sc[k]!=null);
    if(!keys.length)return buildQuizQuestion();
    const k=keys[Math.floor(Math.random()*keys.length)];
    const lbl={fat:'fat%',protein:'protein%',carbs:'carbs%',moisture:'moisture%',maillardPct:'Maillard reactivity %'}[k]||k;
    const correctVal=sc[k];
    // wrong answers: use other dishes
    const wrongs=[];
    while(wrongs.length<3){
      const o=pick();const os=(o.dish||{}).science||{};
      if(os[k]!=null&&os[k]!==correctVal)wrongs.push(os[k]);
    }
    const opts=shuffle([correctVal,...wrongs.slice(0,3)]);
    return{cat,question:`What is the average ${lbl} of ingredients in ${node.name}?`,answer:correctVal,options:opts.map(v=>v+'%')};
  }
  if(cat==='Geography'){
    const node=pick();
    if(!node.cuisine)return buildQuizQuestion();
    const real=node.cuisine;
    const wrongs=[...new Set((DATA.cuisines||[]).filter(c=>c!==real))].sort(()=>Math.random()-.5).slice(0,3);
    if(wrongs.length<3)return buildQuizQuestion();
    return{cat,question:`Which cuisine does ${node.name} belong to?`,answer:real,options:shuffle([real,...wrongs])};
  }
  if(cat==='Flavour'){
    const node=pick();const d=node.dish||{};
    const flav=(d.flavour||[]);
    if(flav.length<1)return buildQuizQuestion();
    const best=flav[0].descriptor;
    const wrongs=[];
    while(wrongs.length<3){
      const o=pick();const of2=((o.dish||{}).flavour||[]);
      if(of2.length>0&&of2[0].descriptor!==best&&!wrongs.includes(of2[0].descriptor))wrongs.push(of2[0].descriptor);
    }
    return{cat,question:`What is the dominant flavour descriptor in ${node.name}?`,answer:best,options:shuffle([best,...wrongs])};
  }
  // Compounds
  const node=pick();const d=node.dish||{};
  const comps=(d.compounds||[]);
  if(!comps.length)return buildQuizQuestion();
  const real=comps[Math.floor(Math.random()*comps.length)].compound;
  const wrongs=[];
  while(wrongs.length<3){
    const o=pick();const oc=((o.dish||{}).compounds||[]);
    const ci=oc[Math.floor(Math.random()*oc.length)];
    if(ci&&ci.compound&&ci.compound!==real&&!wrongs.includes(ci.compound))wrongs.push(ci.compound);
  }
  if(wrongs.length<3)return buildQuizQuestion();
  return{cat,question:`Which chemical compound is found in ${node.name}?`,answer:real,options:shuffle([real,...wrongs])};
}

function shuffle(a){return a.sort(()=>Math.random()-.5);}

function openQuiz(){
  qzQ=buildQuizQuestion();
  if(!qzQ)return;
  qzAnswered=false;
  $('qz-cat').textContent=qzQ.cat;
  $('qz-q').textContent=qzQ.question;
  $('qz-opts').innerHTML=qzQ.options.map(o=>`<button class="qz-opt" onclick="answerQuiz(this,'${esc(String(o))}')">${esc(String(o))}</button>`).join('');
  $('qz-feedback').textContent='';$('qz-feedback').className='qz-feedback';
  $('quiz-next').style.display='none';
  $('quiz-panel').classList.add('visible');
  $('quiz-overlay').classList.add('visible');
  $('qz-score').textContent=qzScore;$('qz-total').textContent=qzTotal;
}

window.answerQuiz=function(btn,ans){
  if(qzAnswered)return;qzAnswered=true;qzTotal++;
  const correct=String(qzQ.answer)===ans||(qzQ.answer+'%')===ans;
  if(correct){qzScore++;quizCorrect++;$('qz-feedback').textContent='✓ Correct!';$('qz-feedback').className='qz-feedback right';showToast('🎓','Correct!',qzQ.question);}
  else{$('qz-feedback').textContent=`✗ Wrong. Answer: ${qzQ.answer}`;$('qz-feedback').className='qz-feedback wrong';}
  $('qz-opts').querySelectorAll('.qz-opt').forEach(b=>{b.classList.add('locked');if(b.textContent===String(qzQ.answer)||(b.textContent===qzQ.answer+'%'))b.classList.add('correct');else if(b===btn)b.classList.add('wrong');});
  $('qz-score').textContent=qzScore;$('qz-total').textContent=qzTotal;
  $('quiz-next').style.display='inline-block';
  checkAchs();
};

$('quiz-next').onclick=openQuiz;
$('quiz-close').onclick=()=>{$('quiz-panel').classList.remove('visible');$('quiz-overlay').classList.remove('visible');};
$('btn-quiz').onclick=openQuiz;
$('quiz-overlay').onclick=()=>{$('quiz-panel').classList.remove('visible');$('quiz-overlay').classList.remove('visible');};

/* ── Challenges ── */
const CHALL=[
  {t:'Find a dish with 12+ ingredients',c:n=>((n.dish||{}).ingredients||[]).length>=12},
  {t:'Discover a South Indian biryani',c:n=>(n.cuisine||'').toLowerCase().includes('south')},
  {t:'Find a dish serving 6+ people',c:n=>{const s=parseInt((n.dish||{}).serves);return!isNaN(s)&&s>=6;}},
  {t:'Discover a Dum-cooked biryani',c:n=>((n.dish||{}).styleDetails||[]).some(s=>(s.style||'').toLowerCase().includes('dum'))||(n.name||'').toLowerCase().includes('dum')},
  {t:'Compare two dishes from different cuisines',c:()=>cmpA&&cmpB&&cmpA.cuisine!==cmpB.cuisine,g:true},
  {t:'Explore 3 dishes from the same cluster',c:()=>{for(const cl in clusterDone)if(clusterDone[cl].size>=3)return true;return false;},g:true},
  {t:'Find a dish with Maillard reactivity > 60%',c:n=>{const sc=(n.dish||{}).science||{};return(sc.maillardPct||0)>60;}},
];
let challIdx=0,challDone=false;
const chText=$('ch-text'),chBtn=$('ch-btn');
function loadChall(){if(challIdx>=CHALL.length){chText.textContent='All challenges done! 🎉';chBtn.style.display='none';return;}challDone=false;chText.textContent=CHALL[challIdx].t;chBtn.style.display='none';}
function checkChallenge(node){if(challIdx>=CHALL.length||challDone)return;const ch=CHALL[challIdx];const pass=ch.g?ch.c():ch.c(node);if(pass){challDone=true;chBtn.style.display='inline-block';showToast('🏅','Challenge Complete!',ch.t);}}
chBtn.onclick=()=>{challIdx++;loadChall();};
loadChall();

/* ── Controls ── */
$('search').addEventListener('input',()=>{searchQ=$('search').value.trim();$('search-clear').classList.toggle('visible',!!searchQ);applyVis();});
$('search-clear').onclick=()=>{$('search').value='';searchQ='';$('search-clear').classList.remove('visible');applyVis();};
$('btn-fit').onclick=()=>svg.transition().duration(450).call(zoom.transform,d3.zoomIdentity);
$('btn-shake').onclick=()=>sim.alpha(.85).restart();
$('btn-labels').onclick=()=>{showLbls=!showLbls;tick();$('btn-labels').classList.toggle('on',showLbls);};

/* ── Legend ── */
const leg=$('legend');
(DATA.cuisines||[]).forEach(c=>{
  const col=C[c]||'#9ca3af';
  const div=document.createElement('div');div.className='legend-item';
  div.innerHTML=`<span class="legend-dot" style="background:${col};box-shadow:0 0 6px ${col}88"></span><span>${c}</span>`;
  div.onclick=()=>{if(cuisineFilters.has(c))cuisineFilters.delete(c);else cuisineFilters.add(c);div.classList.toggle('active',cuisineFilters.has(c));applyVis();};
  leg.appendChild(div);
});

applyVis();updateXP();renderAchLog();
window.addEventListener('resize',()=>location.reload());
</script>
</body>
</html>"""
    )


if __name__ == "__main__":
    main()
