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
TOP_PAIRINGS = 60

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
            "ingredients": ing_list, "flavour": flavour,
            "steps": steps, "compounds": compounds,
            "stepIngredients": step_ing_list,
            "styleDetails": style_details,
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

    # --- Top pairings ---
    pair_list = []
    for _, row in pairs.iterrows():
        i = id_to_idx.get(int(row["dish_id_a"]))
        j = id_to_idx.get(int(row["dish_id_b"]))
        if i is not None and j is not None:
            pair_list.append((i, j, float(row["final_similarity"])))
    pair_list.sort(key=lambda t: -t[2])
    pairings = [{"a": a, "b": b, "sim": round(sim, 3)} for a, b, sim in pair_list[:TOP_PAIRINGS]]

    data = {
        "nodes": nodes, "dishInfo": dish_info,
        "clusterCenters": centers, "pairings": pairings,
        "cuisines": cuisines,
        "hierarchy": hierarchy_data,
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
<title>Biryani Ingredient Tree</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#080f1e;font-family:'Inter',sans-serif;overflow:hidden;width:100vw;height:100vh;color:#e8e0d0}
  #canvas{display:block}
  #header{position:fixed;top:18px;left:50%;transform:translateX(-50%);text-align:center;z-index:60;pointer-events:none}
  #header h1{font-family:'Playfair Display',serif;font-size:24px;color:#e8c97a;letter-spacing:3px;text-shadow:0 0 30px rgba(232,201,122,.45)}
  #header p{font-size:10px;color:#4a5e80;margin-top:5px;letter-spacing:2.5px;text-transform:uppercase}

  #info-panel{position:fixed;right:18px;top:50%;transform:translateY(-50%) translateX(20px);width:280px;
    background:rgba(6,14,28,.93);border:1px solid rgba(200,168,76,.22);border-radius:18px;padding:20px;
    backdrop-filter:blur(14px);z-index:60;opacity:0;pointer-events:none;
    transition:opacity .35s ease,transform .35s ease;max-height:76vh;overflow:auto}
  #info-panel.visible{opacity:1;transform:translateY(-50%) translateX(0);pointer-events:auto}
  .panel-emoji{font-size:34px;margin-bottom:9px;line-height:1}
  .panel-name{font-family:'Playfair Display',serif;font-size:15px;color:#e8c97a;line-height:1.35;margin-bottom:8px}
  .cuisine-badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:9px;font-weight:700;
    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:13px}
  .section-title{font-size:8px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#2e4055;margin:10px 0 7px}
  #panel-ingredients{list-style:none}
  #panel-ingredients li{font-size:11.5px;color:#7a93ae;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);display:flex;align-items:center;gap:7px}
  #panel-ingredients li::before{content:"◆";font-size:5px;color:#e8c97a;flex-shrink:0}
  #panel-similar{list-style:none}
  #panel-similar li{font-size:10.5px;color:#8ea3bc;line-height:1.5;margin:2px 0}
  #panel-path{margin-top:11px;font-size:9.5px;color:#2e4055;font-style:italic;line-height:1.7;border-top:1px solid rgba(255,255,255,.04);padding-top:9px}
  #close-panel{position:absolute;top:12px;right:12px;background:none;border:none;color:#2e4055;cursor:pointer;font-size:14px;line-height:1;padding:2px}
  #close-panel:hover{color:#e8c97a}

  #legend{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
    background:rgba(6,14,28,.88);border:1px solid rgba(200,168,76,.13);border-radius:40px;
    padding:9px 22px;display:flex;align-items:center;gap:5px 14px;flex-wrap:wrap;
    justify-content:center;z-index:60;backdrop-filter:blur(10px);max-width:92vw}
  .leg-title{font-size:8px;color:#2e4055;text-transform:uppercase;letter-spacing:1.5px}
  .leg-item{display:flex;align-items:center;gap:5px;font-size:10.5px;color:#6a7e92}
  .leg-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
  .leg-sep{width:1px;height:16px;background:rgba(255,255,255,.07);margin:0 3px}

  #controls{position:fixed;bottom:70px;right:18px;display:flex;flex-direction:column;gap:7px;z-index:60}
  .ctrl-btn{width:36px;height:36px;border-radius:50%;background:rgba(6,14,28,.92);
    border:1px solid rgba(200,168,76,.28);color:#a07a30;font-size:18px;cursor:pointer;
    display:flex;align-items:center;justify-content:center;transition:all .2s;line-height:1}
  .ctrl-btn:hover{background:rgba(200,168,76,.14);border-color:#e8c97a;color:#e8c97a}

  #qtip{position:fixed;background:rgba(6,14,28,.96);border:1px solid rgba(200,168,76,.35);
    border-radius:8px;padding:7px 11px;font-size:11px;color:#c8d8e8;pointer-events:none;
    z-index:200;display:none;white-space:nowrap;box-shadow:0 4px 20px rgba(0,0,0,.5)}

  svg{cursor:grab}
  svg.grabbing{cursor:grabbing}
  .nd{cursor:pointer}
  .nd text{font-family:'Inter',sans-serif;user-select:none;pointer-events:none}

  #search-wrap{position:fixed;top:18px;left:18px;z-index:60}
  #search{background:rgba(6,14,28,.88);border:1px solid rgba(200,168,76,.22);border-radius:10px;
    padding:7px 13px;font-size:12px;color:#c8d8e8;width:220px;outline:none;
    font-family:'Inter',sans-serif}
  #search::placeholder{color:#2e4055}
  #search:focus{border-color:rgba(200,168,76,.5)}
  #search-clear{position:absolute;right:9px;top:50%;transform:translateY(-50%);background:none;border:none;color:#2e4055;cursor:pointer;font-size:14px;display:none}
  #search-clear.visible{display:block}

  .nd.highlighted circle.main-circle{stroke:#e8c97a !important;stroke-width:3px !important}
  .nd.dimmed{opacity:.18}
</style>
</head>
<body>
<div id="header">
  <h1>🌿 Biryani Ingredient Tree</h1>
  <p>Click nodes to expand · Hover dishes for details · Scroll to zoom · Drag to pan</p>
</div>

<div id="search-wrap">
  <input id="search" placeholder="🔍  Search biryani or ingredient…" autocomplete="off">
  <button id="search-clear" title="Clear">✕</button>
</div>

<svg id="canvas"></svg>

<div id="info-panel">
  <button id="close-panel">✕</button>
  <div class="panel-emoji" id="panel-emoji">🍽️</div>
  <div class="panel-name" id="panel-name"></div>
  <div class="cuisine-badge" id="panel-cuisine"></div>
  <div class="section-title">Ingredients</div>
  <ul id="panel-ingredients"></ul>
  <div class="section-title">Top Similar</div>
  <ul id="panel-similar"></ul>
  <div id="panel-path"></div>
</div>

<div id="legend">
  <span class="leg-title">Leaf colour = Cuisine</span>
  <div class="leg-sep"></div>
</div>

<div id="controls">
  <button class="ctrl-btn" id="btn-reset" title="Reset view">⊙</button>
  <button class="ctrl-btn" id="btn-expand" title="Expand all">⊕</button>
  <button class="ctrl-btn" id="btn-collapse" title="Collapse to trunks">⊖</button>
</div>

<div id="qtip"></div>

<script>
const DATA="""
        + data_json
        + """;

const CLUSTER_COLOR = {};
DATA.hierarchy.forEach((h, idx) => {
  CLUSTER_COLOR[h.clusterLabel] = h.color || '#888';
  CLUSTER_COLOR['cluster_' + (idx + 1)] = h.color || '#888';
});

const PALETTE = [
  '#e74c3c','#4a90d9','#f39c12','#9b59b6','#27ae60','#e67e22','#1abc9c',
  '#2ecc71','#d35400','#8e9eab','#16a085','#8e44ad','#f1c40f','#3498db'
];
const CUISINE_C = {};
(DATA.cuisines || []).forEach((c, i) => { CUISINE_C[c] = PALETTE[i % PALETTE.length]; });

function dishById(idx){ return (DATA.dishInfo && DATA.dishInfo[idx]) ? DATA.dishInfo[idx] : null; }

function toLeafNode(id){
  const d = dishById(id) || {};
  return {
    name: d.name || ('Dish ' + id),
    type: 'leaf',
    cuisine: d.cuisine || 'Unknown',
    emoji: '🍽️',
    ingredients: (d.ingredients || []).slice(0, 10),
    dishId: id
  };
}

function toBranchNode(node, depth){
  if(!node) return {name:'Group', type:'branch', children:[]};
  if(node.t === 'l') return toLeafNode(node.id);
  return {
    name: node.lb || ('Group ' + depth),
    type: 'branch',
    children: (node.c || []).map(ch => toBranchNode(ch, depth + 1))
  };
}

const FOOD = {
  name:'Biryani Kitchen', type:'root', emoji:'🌿',
  children:(DATA.hierarchy || []).map((h, i) => ({
    name: h.clusterLabel || ('Cluster ' + (i + 1)),
    type:'trunk',
    family:'cluster_' + (i + 1),
    emoji:'🍲',
    children:[toBranchNode(h.tree, 1)]
  }))
};

const FAMILY_C = {};
FOOD.children.forEach((c, i) => { FAMILY_C[c.family] = CLUSTER_COLOR['cluster_' + (i + 1)] || '#888'; });

function nodeColor(d){
  if(d.data.type==='root') return '#e8c97a';
  if(d.data.type==='trunk') return FAMILY_C[d.data.family] || '#888';
  if(d.data.type==='leaf') return CUISINE_C[d.data.cuisine] || '#aaa';
  const t=d.ancestors().find(a=>a.data.type==='trunk');
  const base=t?FAMILY_C[t.data.family]:'#556';
  return d3.interpolateRgb(base,'#2a3a4e')((d.depth-1)/4);
}
function nodeR(d){ return {root:22,trunk:16,leaf:10}[d.data.type] || 8; }
function fontSize(d){ return {root:14,trunk:13,leaf:10}[d.data.type] || 11; }
function linkW(d){ return {root:3,trunk:2.5}[d.source.data.type] || (d.target.data.type==='leaf'?1:1.7); }
function linkAlpha(d){ return {root:.7,trunk:.55}[d.source.data.type] || .3; }
function linkCol(d){
  const t=d.target.ancestors().find(a=>a.data.type==='trunk');
  return t?FAMILY_C[t.data.family]:'#3a4e66';
}

const W=window.innerWidth, H=window.innerHeight;
const R=Math.min(W,H)*0.41;

const svg=d3.select('#canvas').attr('width',W).attr('height',H);
const defs=svg.append('defs');
const bg=defs.append('radialGradient').attr('id','bg').attr('cx','50%').attr('cy','50%').attr('r','70%');
bg.append('stop').attr('offset','0%').attr('stop-color','#0e1d35');
bg.append('stop').attr('offset','100%').attr('stop-color','#050a15');
svg.append('rect').attr('width',W).attr('height',H).attr('fill','url(#bg)');

['glow','sGlow'].forEach((id,i)=>{
  const f=defs.append('filter').attr('id',id);
  f.append('feGaussianBlur').attr('stdDeviation',i===0?5:2.5).attr('result','cb');
  const m=f.append('feMerge');
  m.append('feMergeNode').attr('in','cb');
  m.append('feMergeNode').attr('in','SourceGraphic');
});

const root2=svg.append('g').attr('id','root2').attr('transform',`translate(${W/2},${H/2})`);
const rings=root2.append('g');
[1,2,3,4].forEach(d=>{
  rings.append('circle').attr('r',d*(R/4))
    .attr('fill','none')
    .attr('stroke',`rgba(255,255,255,${d===4?.05:.025})`)
    .attr('stroke-dasharray',d===4?'5,10':'3,9');
});

const gLink=root2.append('g').attr('class','links');
const gNode=root2.append('g').attr('class','nodes');

const treeLayout=d3.tree()
  .size([2*Math.PI,R])
  .separation((a,b)=>(a.parent===b.parent?1:2)/a.depth);

let rootNode=d3.hierarchy(FOOD);
let uid=0; rootNode.each(d=>d.id=uid++);
rootNode.each(d=>{
  d._children=d.children?[...d.children]:null;
  if(d.depth>=3) d.children=null;
  d.x0=0; d.y0=0;
});

const diagonal=d3.linkRadial().angle(d=>d.x).radius(d=>d.y);
const px=(d)=>`translate(${d.y*Math.sin(d.x)},${-d.y*Math.cos(d.x)})`;
const pxSrc=(src)=>`translate(${(src.y0||0)*Math.sin(src.x0||0)},${-(src.y0||0)*Math.cos(src.x0||0)})`;

const panel=document.getElementById('info-panel');
const pEmoji=document.getElementById('panel-emoji');
const pName=document.getElementById('panel-name');
const pCuisine=document.getElementById('panel-cuisine');
const pIngr=document.getElementById('panel-ingredients');
const pSimilar=document.getElementById('panel-similar');
const pPath=document.getElementById('panel-path');
const qtip=document.getElementById('qtip');

function showLeaf(event,d){
  if(d.data.type!=='leaf') return;
  const info = dishById(d.data.dishId) || {};
  pEmoji.textContent='🍽️';
  pName.textContent=info.name || d.data.name;
  const c=CUISINE_C[info.cuisine || d.data.cuisine]||'#888';
  pCuisine.textContent=info.cuisine || d.data.cuisine || 'Unknown';
  pCuisine.style.cssText=`background:${c}20;color:${c};border:1px solid ${c}45`;
  pIngr.innerHTML=(info.ingredients||d.data.ingredients||[]).slice(0,12).map(i=>`<li>${i}</li>`).join('');
  const topSimilar=(info.topSimilar||[]).slice(0,3);
  pSimilar.innerHTML=topSimilar.length
    ? topSimilar.map(s=>`<li>${s.name} (${(s.score*100).toFixed(1)}%)</li>`).join('')
    : '<li>No similarity data</li>';
  pPath.textContent=d.ancestors().reverse().map(a=>a.data.name).join(' → ');
  panel.classList.add('visible');
}
document.getElementById('close-panel').onclick=()=>panel.classList.remove('visible');

function showQtip(event,d){
  if(d.data.type==='leaf'){showLeaf(event,d);return;}
  const state=d.children?'click to collapse':'click to expand';
  qtip.textContent=`${d.data.emoji?d.data.emoji+' ':''}${d.data.name} (${state})`;
  qtip.style.display='block';
  qtip.style.left=(event.clientX+14)+'px';
  qtip.style.top=(event.clientY-8)+'px';
}
function hideQtip(){qtip.style.display='none';}

function update(src){
  const dur=650;
  treeLayout(rootNode);
  const nodes=rootNode.descendants();
  const links=rootNode.links();
  nodes.forEach(d=>{ d.y=d.depth*(R/4); });

  const link=gLink.selectAll('path.lk').data(links,d=>d.target.id);
  link.enter().append('path').attr('class','lk')
    .attr('fill','none')
    .attr('d',()=>{const o={x:src.x0||0,y:src.y0||0};return diagonal({source:o,target:o});})
    .attr('stroke',d=>linkCol(d)).attr('stroke-width',d=>linkW(d)).attr('stroke-opacity',0)
    .merge(link)
    .transition().duration(dur)
    .attr('d',diagonal)
    .attr('stroke',d=>linkCol(d)).attr('stroke-width',d=>linkW(d)).attr('stroke-opacity',d=>linkAlpha(d));
  link.exit().transition().duration(dur)
    .attr('d',()=>{const o={x:src.x,y:src.y};return diagonal({source:o,target:o});})
    .attr('stroke-opacity',0).remove();

  const node=gNode.selectAll('g.nd').data(nodes,d=>d.id);
  const ne=node.enter().append('g').attr('class',d=>`nd nd-${d.data.type}`)
    .attr('transform',()=>pxSrc(src)).style('opacity',0)
    .on('click',(ev,d)=>{
      if(d.children){d._children=d.children;d.children=null;}
      else if(d._children){d.children=d._children;d._children=null;}
      d.x0=d.x; d.y0=d.y;
      update(d);
      clearSearch();
    })
    .on('mouseover',(ev,d)=>{showQtip(ev,d);})
    .on('mouseout',()=>{hideQtip();});

  ne.filter(d=>d.data.type==='leaf').append('circle').attr('class','leaf-ring')
    .attr('r',d=>nodeR(d)+5).attr('fill','none').attr('stroke',d=>nodeColor(d)).attr('stroke-width',1).attr('stroke-opacity',.35);

  ne.append('circle').attr('class','main-circle').attr('r',0)
    .attr('fill',d=>nodeColor(d))
    .attr('stroke','#060e1b').attr('stroke-width',2)
    .attr('filter',d=>d.data.type==='root'?'url(#glow)':d.data.type==='trunk'?'url(#sGlow)':null);

  ne.append('circle').attr('class','cdot').attr('r',2.5).attr('fill','#e8c97a').attr('opacity',0);

  ne.append('text').attr('class','lbl').attr('dy','0.31em').style('opacity',0)
    .attr('font-size',d=>fontSize(d)+'px')
    .attr('fill',d=>d.data.type==='root'?'#e8c97a':d.data.type==='trunk'?nodeColor(d):d.data.type==='leaf'?'#b8cce0':'#6a8090')
    .attr('font-weight',d=>d.data.type==='trunk'?600:400)
    .text(d=>{
      const em=d.data.emoji?d.data.emoji+' ':'';
      return (d.data.type==='trunk'||d.data.type==='root')?em+d.data.name:d.data.name;
    });

  ne.filter(d=>d.data.type==='leaf').append('text').attr('class','leafi')
    .attr('text-anchor','middle').attr('dominant-baseline','central')
    .attr('font-size','9px').attr('pointer-events','none').text('🍽️');

  const nu=ne.merge(node);
  nu.transition().duration(dur).attr('transform',d=>px(d)).style('opacity',1);
  nu.select('circle.main-circle').transition().duration(dur).attr('r',d=>nodeR(d));
  nu.select('circle.cdot').transition().duration(dur).attr('opacity',d=>(!d.children&&d._children)?1:0);
  nu.select('text.lbl').transition().duration(dur).style('opacity',1)
    .attr('text-anchor',d=>d.depth===0?'middle':d.x<Math.PI?'start':'end')
    .attr('x',d=>d.depth===0?0:(d.x<Math.PI?(nodeR(d)+6):-(nodeR(d)+6)))
    .attr('y',d=>d.depth===0?-(nodeR(d)+8):0);

  node.exit().transition().duration(dur)
    .attr('transform',`translate(${src.y*Math.sin(src.x)},${-src.y*Math.cos(src.x)})`)
    .style('opacity',0).remove();

  nodes.forEach(d=>{d.x0=d.x;d.y0=d.y;});
}

const zoom=d3.zoom().scaleExtent([.25,5])
  .on('zoom',ev=>{ root2.attr('transform',`translate(${W/2+ev.transform.x},${H/2+ev.transform.y}) scale(${ev.transform.k})`); });
svg.call(zoom).on('dblclick.zoom',null);
svg.on('mousedown.grab',()=>svg.classed('grabbing',true)).on('mouseup.grab',()=>svg.classed('grabbing',false));

document.getElementById('btn-reset').onclick=()=>{ svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity); };
document.getElementById('btn-expand').onclick=()=>{ rootNode.each(d=>{if(d._children){d.children=d._children;d._children=null;}}); update(rootNode); };
document.getElementById('btn-collapse').onclick=()=>{
  rootNode.each(d=>{ if(d.depth>0&&d.children){d._children=d.children;d.children=null;} });
  update(rootNode);
};

const searchInput=document.getElementById('search');
const searchClear=document.getElementById('search-clear');
function doSearch(term){
  if(!term){clearSearch();return;}
  const lo=term.toLowerCase();
  rootNode.each(d=>{if(d._children){d.children=d._children;d._children=null;}});
  update(rootNode);
  setTimeout(()=>{
    gNode.selectAll('g.nd').each(function(d){
      const info = d.data.type === 'leaf' ? dishById(d.data.dishId) : null;
      const match = d.data.name.toLowerCase().includes(lo)
        || (info && info.ingredients && info.ingredients.some(i=>i.toLowerCase().includes(lo)))
        || (d.data.cuisine && d.data.cuisine.toLowerCase().includes(lo));
      d3.select(this).classed('highlighted',!!match).classed('dimmed',!match);
    });
  },700);
}
function clearSearch(){
  searchInput.value='';
  searchClear.classList.remove('visible');
  gNode.selectAll('g.nd').classed('highlighted',false).classed('dimmed',false);
}
searchInput.addEventListener('input',()=>{
  const v=searchInput.value.trim();
  searchClear.classList.toggle('visible',v.length>0);
  doSearch(v);
});
searchClear.onclick=clearSearch;

const legEl=document.getElementById('legend');
(DATA.cuisines || []).forEach(c=>{
  const col = CUISINE_C[c] || '#777';
  const d=document.createElement('div');
  d.className='leg-item';
  d.innerHTML=`<div class="leg-dot" style="background:${col};box-shadow:0 0 5px ${col}88"></div><span>${c}</span>`;
  legEl.appendChild(d);
});
const s=document.createElement('div');s.className='leg-sep';legEl.appendChild(s);
legEl.innerHTML+=`<span class="leg-title">Node size = Depth</span>`;

update(rootNode);

window.addEventListener('resize',()=>{
  const nw=window.innerWidth,nh=window.innerHeight;
  svg.attr('width',nw).attr('height',nh);
  svg.select('rect').attr('width',nw).attr('height',nh);
  root2.attr('transform',`translate(${nw/2},${nh/2})`);
});
</script>
</body>
</html>"""
    )


if __name__ == "__main__":
    main()
