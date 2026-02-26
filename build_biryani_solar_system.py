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

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS

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
        for _, ir in ing_df.iterrows():
            q = ir.get("quantity")
            u = str(ir.get("unit_of_measure", "")) if pd.notna(ir.get("unit_of_measure")) else ""
            nm = str(ir.get("ingredient_name", ""))
            prep = str(ir.get("preparation_method", "")) if pd.notna(ir.get("preparation_method")) else ""
            line = f"{q} {u} {nm}".strip() if pd.notna(q) else nm
            if prep:
                line += f" ({prep})"
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Biryani World Tree</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#100b06;--panel:#17110b;--border:#2b1f12;--text:#f3eee6;--muted:#b8a98c;--accent:#e89b3a;--leaf:#7aa96b;--rice:#f4e2c3}
body{overflow:hidden;background:radial-gradient(1200px 600px at 10% 10%,#1a140d 0%,#0f0b07 50%,#0b0805 100%);font-family:'Space Grotesk','Segoe UI',sans-serif;color:var(--text)}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(600px 280px at 70% 20%,rgba(232,155,58,0.08),transparent 70%),radial-gradient(500px 320px at 30% 80%,rgba(122,169,107,0.08),transparent 70%);pointer-events:none;z-index:0}
#scene-shell{position:fixed;top:56px;left:14px;right:14px;bottom:14px;z-index:0;border-radius:36px 36px 52px 52px;background:radial-gradient(900px 420px at 55% 35%,rgba(74,108,66,0.12),transparent 65%),radial-gradient(700px 340px at 45% 78%,rgba(232,155,58,0.08),transparent 70%),linear-gradient(180deg,rgba(20,14,10,0.55),rgba(12,9,7,0.78));border:1px solid rgba(255,255,255,0.07);box-shadow:inset 0 0 80px rgba(0,0,0,0.35)}
#canvas{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:1}

#tree-summary{position:fixed;top:72px;left:26px;z-index:70;width:300px;max-height:calc(100vh - 120px);overflow:auto;padding:0.85rem 0.9rem;background:rgba(19,14,10,0.78);border:1px solid rgba(255,255,255,0.1);border-radius:16px 28px 16px 24px;backdrop-filter:blur(8px)}
#tree-summary h3{font-family:'Fraunces',serif;font-size:0.95rem;color:#f8d8a8;margin-bottom:0.45rem}
#tree-summary h4{font-size:0.75rem;letter-spacing:0.04em;text-transform:uppercase;color:#d8be93;margin:0.6rem 0 0.35rem}
#tree-summary ul{list-style:none;padding:0;margin:0}
#tree-summary li{font-size:0.78rem;color:#e8dfd1;line-height:1.45;margin-bottom:0.22rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#tree-summary .muted{color:#b8a98c;font-size:0.75rem}
@media (max-width:1050px){#tree-summary{display:none}}

/* Top bar */
#topbar{position:fixed;top:0;left:0;right:0;height:48px;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:0 1.2rem;background:rgba(5,5,16,0.85);backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
#topbar h1{font-family:'Fraunces',serif;font-size:0.95rem;letter-spacing:0.08em;text-transform:none;color:rgba(255,255,255,0.7);font-weight:600}
#chef-toggle{display:flex;align-items:center;gap:0.5rem;cursor:pointer;font-size:0.85rem;color:var(--muted)}
#chef-toggle .dot{width:36px;height:20px;border-radius:10px;background:#333;position:relative;transition:background 0.3s}
#chef-toggle .dot::after{content:'';position:absolute;width:16px;height:16px;border-radius:50%;background:#888;top:2px;left:2px;transition:left 0.3s,background 0.3s}

/* Hover tooltip */
#tooltip{position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity 0.2s;background:rgba(12,12,26,0.95);border:1px solid var(--border);border-radius:10px;padding:0.6rem 0.8rem;min-width:180px;backdrop-filter:blur(6px)}
#tooltip.show{opacity:1}
#tooltip img{width:60px;height:60px;border-radius:8px;object-fit:cover;float:left;margin-right:0.6rem}
#tooltip .tt-name{font-weight:600;font-size:0.9rem;margin-bottom:2px}
#tooltip .tt-cuisine{font-size:0.75rem;color:var(--muted)}

/* Intro */
#intro-label{position:fixed;left:50%;bottom:3.5rem;transform:translateX(-50%);color:rgba(255,255,255,0.6);font-size:0.95rem;z-index:20;transition:opacity 0.5s}
#intro-label.hide{opacity:0;pointer-events:none}
#skip{position:fixed;bottom:1.5rem;left:50%;transform:translateX(-50%);z-index:20;padding:0.5rem 1.2rem;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);border-radius:999px;color:rgba(255,255,255,0.8);font-size:0.8rem;cursor:pointer}
#skip:hover{background:rgba(255,255,255,0.12)}
#skip.hide{display:none}

/* Info panel */
#panel{position:fixed;top:48px;right:0;width:380px;height:calc(100vh - 48px);z-index:80;background:var(--panel);border-left:1px solid var(--border);transform:translateX(100%);transition:transform 0.35s ease;display:flex;flex-direction:column;overflow:hidden}
#panel.open{transform:translateX(0)}
#panel-close{position:absolute;top:0.6rem;right:0.8rem;background:none;border:none;color:var(--muted);font-size:1.2rem;cursor:pointer;z-index:2}
#cuisine-filter{padding:0.8rem 1rem;border-bottom:1px solid var(--border)}
#cuisine-filter select{width:100%;padding:0.4rem 0.5rem;background:#111;color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:0.8rem}
#dish-list{max-height:200px;overflow-y:auto;border-bottom:1px solid var(--border)}
#dish-list .dish-item{padding:0.5rem 1rem;font-size:0.82rem;cursor:pointer;border-bottom:1px solid rgba(255,255,255,0.04)}
#dish-item:hover,.dish-item:hover{background:rgba(255,255,255,0.04)}
#dish-item.active,.dish-item.active{background:rgba(240,165,0,0.12);color:var(--accent)}
#dish-detail{flex:1;overflow-y:auto;display:none}
#dish-detail.show{display:block}
#dish-detail .detail-header{padding:1rem;border-bottom:1px solid var(--border);display:flex;gap:0.8rem;align-items:flex-start}
#dish-detail .detail-header img{width:72px;height:72px;border-radius:10px;object-fit:cover;flex-shrink:0}
#dish-detail .detail-header h2{font-size:1.05rem;margin-bottom:0.3rem}
#dish-detail .meta-tags{display:flex;gap:0.4rem;flex-wrap:wrap}
#dish-detail .meta-tag{font-size:0.7rem;padding:0.2rem 0.5rem;background:rgba(255,255,255,0.06);border-radius:4px;color:var(--muted)}
.tabs{display:flex;flex-wrap:wrap;border-bottom:1px solid var(--border);padding:0 0.6rem;gap:0.1rem}
.tab-btn{padding:0.4rem 0.55rem;font-size:0.7rem;color:var(--muted);cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;transition:color 0.2s,border-color 0.2s;white-space:nowrap}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-content{display:none;padding:0.8rem 1rem;font-size:0.82rem;line-height:1.6}
.tab-content.show{display:block}
.tab-content ul{margin:0;padding-left:1.2rem}
.tab-content li{margin-bottom:0.3rem}
.flavour-bar{display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem}
.flavour-bar .f-name{width:90px;text-align:right;font-size:0.75rem;color:var(--muted);flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.flavour-bar .f-bar{flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden}
.flavour-bar .f-fill{height:100%;border-radius:4px;transition:width 0.4s}
.flavour-bar .f-val{font-size:0.7rem;color:var(--muted);width:30px}
.read-more{color:var(--accent);cursor:pointer;font-size:0.75rem;margin-top:0.4rem;display:inline-block}
.step-item{margin-bottom:0.6rem;padding-left:1.5rem;position:relative}
.step-item::before{content:attr(data-n);position:absolute;left:0;width:1.1rem;height:1.1rem;border-radius:50%;background:rgba(240,165,0,0.15);color:var(--accent);font-size:0.65rem;display:flex;align-items:center;justify-content:center;top:2px}
.step-style{font-size:0.7rem;color:var(--muted);margin-top:2px}
.compound-row{display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.78rem}
.compound-row span:last-child{color:var(--muted)}
#panel-toggle{position:fixed;top:56px;right:8px;z-index:90;width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.08);border:1px solid var(--border);color:var(--text);cursor:pointer;font-size:0.9rem;display:flex;align-items:center;justify-content:center}

/* Waitlist modal */
#modal-overlay{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);align-items:center;justify-content:center}
#modal-overlay.show{display:flex}
#modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:2rem;width:340px;max-width:90vw;text-align:center}
#modal h3{font-size:1.1rem;margin-bottom:0.5rem}
#modal p{font-size:0.85rem;color:var(--muted);margin-bottom:1.2rem}
#modal input[type=email]{width:100%;padding:0.6rem 0.8rem;background:#111;border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:0.9rem;margin-bottom:0.4rem}
#modal input[type=email]:focus{outline:none;border-color:var(--accent)}
#modal .err{color:#ff6b6b;font-size:0.75rem;min-height:1rem;margin-bottom:0.6rem}
#modal button{padding:0.6rem 1.5rem;background:var(--accent);color:#000;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:0.9rem}
#modal button:hover{opacity:0.9}
#modal .close-modal{position:absolute;top:0.8rem;right:1rem;background:none;border:none;color:var(--muted);font-size:1.2rem;cursor:pointer}
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <h1>Biryani World Tree</h1>
  <div id="chef-toggle" onclick="toggleChef()">
    <span>Chef Mode</span>
    <div class="dot"></div>
  </div>
</div>

<div id="scene-shell"></div>

<!-- 3D canvas -->
<canvas id="canvas"></canvas>

<aside id="tree-summary"></aside>

<!-- Hover tooltip -->
<div id="tooltip">
  <img id="tt-img" src="" alt="">
  <div class="tt-name" id="tt-name"></div>
  <div class="tt-cuisine" id="tt-cuisine"></div>
</div>

<!-- Intro -->
<div id="intro-label"></div>
<button id="skip">Skip intro</button>

<!-- Cuisine legend -->

<!-- Panel toggle -->
<button id="panel-toggle" onclick="togglePanel()">☰</button>

<!-- Info panel -->
<div id="panel">
  <button id="panel-close" onclick="closePanel()">✕</button>
  <div id="cuisine-filter">
    <select id="cuisine-select" onchange="filterCuisine()">
      <option value="">All cuisines</option>
    </select>
  </div>
  <div id="dish-list"></div>
  <div id="dish-detail">
    <div class="detail-header">
      <img id="detail-img" src="ambur_biryani.png" alt="Biryani">
      <div>
        <h2 id="detail-name"></h2>
        <div class="meta-tags" id="detail-meta"></div>
      </div>
    </div>
    <div class="tabs" id="tabs"></div>
    <div id="tab-panels"></div>
  </div>
</div>

<!-- Waitlist modal -->
<div id="modal-overlay">
  <div id="modal" style="position:relative">
    <button class="close-modal" onclick="closeModal()">✕</button>
    <h3>Chef Mode</h3>
    <p>Advanced AI-powered biryani analysis. Join the waitlist to be the first to try it.</p>
    <input type="email" id="waitlist-email" placeholder="Enter your email">
    <div class="err" id="email-err"></div>
    <button onclick="submitWaitlist()">Join Waitlist</button>
  </div>
</div>

<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script>
(function(){
var THREE=window.THREE;
if(!THREE){document.body.innerHTML='<div style="color:#fff;padding:3rem;text-align:center">Three.js did not load. Serve this page: <code>python -m http.server 8000</code></div>';return}

var DATA="""
        + data_json
        + """;

/* ======================== PLACEHOLDERS ======================== */
var PLACEHOLDER_IMG='data:image/svg+xml;base64,'+btoa('<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60"><rect width="60" height="60" rx="8" fill="#1a1a2e"/><circle cx="30" cy="26" r="14" fill="#f0a500" opacity="0.6"/><ellipse cx="30" cy="44" rx="20" ry="6" fill="#f0a500" opacity="0.3"/></svg>');

/* ======================== PANEL LOGIC ======================== */
var panelOpen=false;
window.togglePanel=function(){
  panelOpen=!panelOpen;
  document.getElementById('panel').classList.toggle('open',panelOpen);
};
window.closePanel=function(){
  panelOpen=false;
  document.getElementById('panel').classList.remove('open');
};
// Cuisine filter
var sel=document.getElementById('cuisine-select');
DATA.cuisines.forEach(function(c){var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o)});
window.filterCuisine=function(){
  var c=sel.value;
  renderDishList(c);
};
function renderDishList(cuisineFilter){
  var el=document.getElementById('dish-list');
  el.innerHTML='';
  DATA.dishInfo.forEach(function(d,i){
    if(cuisineFilter&&d.cuisine!==cuisineFilter) return;
    var div=document.createElement('div');
    div.className='dish-item';
    div.textContent=d.name;
    div.onclick=function(){showDish(i)};
    el.appendChild(div);
  });
}
renderDishList('');

function showDish(idx){
  selectInViz(idx);
  var d=DATA.dishInfo[idx];
  document.getElementById('dish-detail').classList.add('show');
  document.getElementById('detail-name').textContent=d.name;
  document.getElementById('detail-img').src='ambur_biryani.png';
  var meta='';
  if(d.cuisine) meta+='<span class="meta-tag">'+d.cuisine+'</span>';
  if(d.category) meta+='<span class="meta-tag">'+d.category+'</span>';
  if(d.totalTime) meta+='<span class="meta-tag">'+d.totalTime+'</span>';
  if(d.serves) meta+='<span class="meta-tag">Serves '+d.serves+'</span>';
  document.getElementById('detail-meta').innerHTML=meta;

  var tabNames=['Ingredients','Flavour','Steps','Compounds','Cooking Style','Similar'];
  var tabsEl=document.getElementById('tabs');
  var panelsEl=document.getElementById('tab-panels');
  tabsEl.innerHTML='';panelsEl.innerHTML='';

  tabNames.forEach(function(t,ti){
    var btn=document.createElement('button');
    btn.className='tab-btn'+(ti===0?' active':'');
    btn.textContent=t;
    btn.onclick=function(){
      tabsEl.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active')});
      btn.classList.add('active');
      panelsEl.querySelectorAll('.tab-content').forEach(function(p){p.classList.remove('show')});
      panelsEl.children[ti].classList.add('show');
    };
    tabsEl.appendChild(btn);

    var panel=document.createElement('div');
    panel.className='tab-content'+(ti===0?' show':'');

    if(t==='Ingredients'){
      if(d.ingredients.length){
        panel.innerHTML='<ul>'+d.ingredients.map(function(ing){return '<li>'+ing+'</li>'}).join('')+'</ul>';
      } else panel.innerHTML='<p style="color:var(--muted)">No ingredients listed.</p>';

    } else if(t==='Flavour'){
      var maxI=Math.max.apply(null,d.flavour.map(function(f){return f.intensity}));
      if(maxI<=0) maxI=1;
      var showCount=7;
      var html='';
      d.flavour.slice(0,showCount).forEach(function(f){
        var pct=Math.round(f.intensity/maxI*100);
        html+='<div class="flavour-bar"><span class="f-name" title="'+f.descriptor+'">'+f.descriptor+'</span><div class="f-bar"><div class="f-fill" style="width:'+pct+'%;background:var(--accent)"></div></div><span class="f-val">'+f.intensity.toFixed(2)+'</span></div>';
      });
      if(d.flavour.length>showCount){
        html+='<span class="read-more" id="rm-'+idx+'">Show all '+d.flavour.length+' descriptors</span>';
        html+='<div id="extra-'+idx+'" style="display:none">';
        d.flavour.slice(showCount).forEach(function(f){
          var pct=Math.round(f.intensity/maxI*100);
          html+='<div class="flavour-bar"><span class="f-name" title="'+f.descriptor+'">'+f.descriptor+'</span><div class="f-bar"><div class="f-fill" style="width:'+pct+'%;background:var(--accent)"></div></div><span class="f-val">'+f.intensity.toFixed(2)+'</span></div>';
        });
        html+='</div>';
      }
      panel.innerHTML=html||'<p style="color:var(--muted)">No flavour data.</p>';
      setTimeout(function(){
        var rm=document.getElementById('rm-'+idx);
        if(rm) rm.onclick=function(){
          var ex=document.getElementById('extra-'+idx);
          ex.style.display=ex.style.display==='none'?'block':'none';
          rm.textContent=ex.style.display==='none'?'Show all '+d.flavour.length+' descriptors':'Show less';
        };
      },0);

    } else if(t==='Steps'){
      if(d.steps.length){
        var h='';
        d.steps.forEach(function(s){
          h+='<div class="step-item" data-n="'+s.step+'">'+s.desc;
          if(s.style) h+='<div class="step-style">'+s.style+(s.dur?' · '+s.dur+' min':'')+'</div>';
          h+='</div>';
        });
        panel.innerHTML=h;
      } else panel.innerHTML='<p style="color:var(--muted)">No steps listed.</p>';

    } else if(t==='Compounds'){
      if(d.compounds.length){
        var h='';
        d.compounds.forEach(function(c){
          h+='<div class="compound-row"><span>'+c.compound+' <small style="color:var(--muted)">(' +c.ingredient+')</small></span><span>'+(c.conc!=null?c.conc+'%':'')+'</span></div>';
        });
        panel.innerHTML=h;
      } else panel.innerHTML='<p style="color:var(--muted)">No compound data.</p>';

    } else if(t==='Cooking Style'){
      var sd=d.styleDetails||[];
      var si=d.stepIngredients||[];
      var h='';
      if(sd.length){
        h+='<div style="margin-bottom:0.7rem"><strong style="font-size:0.85rem;color:var(--accent)">Techniques</strong>';
        sd.forEach(function(s){
          h+='<div class="step-item" style="margin-bottom:0.4rem"><strong>'+s.style+'</strong>';
          if(s.temp) h+='<div style="font-size:0.75rem;color:var(--muted)">Temp: '+s.temp+'</div>';
          if(s.browning) h+='<div style="font-size:0.75rem;color:var(--muted)">Browning: '+s.browning+'</div>';
          if(s.moisture) h+='<div style="font-size:0.75rem;color:var(--muted)">Moisture: '+s.moisture+'</div>';
          h+='</div>';
        });
        h+='</div>';
      }
      if(si.length){
        h+='<strong style="font-size:0.85rem;color:var(--accent)">Step Ingredients</strong>';
        var byStep={};
        si.forEach(function(r){var k=r.step;if(!byStep[k])byStep[k]=[];byStep[k].push(r)});
        Object.keys(byStep).sort(function(a,b){return a-b}).forEach(function(sn){
          h+='<div class="step-item" data-n="'+sn+'">';
          byStep[sn].forEach(function(r){
            h+=r.ingredient+(r.quantity?' ('+r.quantity+')':'')+(r.purpose?' <small style="color:var(--muted)">– '+r.purpose+'</small>':'')+', ';
          });
          h=h.replace(/, $/,'')+'</div>';
        });
      }
      panel.innerHTML=h||'<p style="color:var(--muted)">No cooking style data.</p>';
    } else if(t==='Similar'){
      var ts=d.topSimilar||[];
      if(ts.length){
        var h='<p style="font-size:0.8rem;color:var(--muted);margin-bottom:0.5rem">Top 3 most similar biryanis:</p><ul>';
        ts.forEach(function(s){
          h+='<li><strong>'+s.name+'</strong> <span style="color:var(--muted);font-size:0.8rem">(similarity: '+(s.score*100).toFixed(1)+'%)</span></li>';
        });
        h+='</ul>';
        panel.innerHTML=h;
      } else panel.innerHTML='<p style="color:var(--muted)">No similarity data.</p>';
    }
    panelsEl.appendChild(panel);
  });

  document.querySelectorAll('.dish-item').forEach(function(el){el.classList.remove('active')});
  var items=document.getElementById('dish-list').children;
  for(var k=0;k<items.length;k++) if(items[k].textContent===d.name) items[k].classList.add('active');

  if(!panelOpen) togglePanel();
}

/* ======================== CHEF MODE / WAITLIST ======================== */
window.toggleChef=function(){
  document.getElementById('modal-overlay').classList.add('show');
};
window.closeModal=function(){
  document.getElementById('modal-overlay').classList.remove('show');
  document.getElementById('email-err').textContent='';
};
window.submitWaitlist=function(){
  var email=document.getElementById('waitlist-email').value.trim();
  var err=document.getElementById('email-err');
  if(!email){err.textContent='Please enter an email address.';return}
  if(email.length>254){err.textContent='Email is too long.';return}
  var re=/^[a-zA-Z0-9.!#$%&'*+\\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$/;
  if(!re.test(email)){err.textContent='Please enter a valid email address.';return}
  err.style.color='#2ed573';
  err.textContent='You have been added to the waitlist!';
  document.getElementById('waitlist-email').value='';
  setTimeout(closeModal,2000);
};
document.getElementById('modal-overlay').onclick=function(e){if(e.target===this) closeModal()};

/* ======================== THREE.JS SCENE ======================== */
var canvas=document.getElementById('canvas');
var scene=new THREE.Scene();
var camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,1000);
camera.position.set(0,6,42);
var renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x0b0805,1);

document.getElementById('intro-label').textContent='From rice roots to biryani leaves, follow the ingredient branches.';

var TREE={baseY:-18,trunkTop:4,canopyTop:16,branchRadius:9};

function normalizeIngredient(raw){
  if(!raw) return '';
  var s=String(raw).toLowerCase();
  s=s.replace(/\([^)]*\)/g,'');
  s=s.replace(/[^a-z\s]/g,' ');
  s=s.replace(/\b(as required|to taste|a few|few|some|little|about|approx|approximate|as needed)\b/g,'');
  s=s.replace(/\b(number|numbers|tsp|tbsp|tablespoon|teaspoon|cup|cups|gm|g|kg|ml|l|inch|inches|piece|pieces|sprig|sprigs|clove|cloves|pod|pods|stick|sticks|leaf|leaves|slice|slices)\b/g,'');
  s=s.replace(/\b[0-9]+\b/g,'');
  s=s.replace(/\b(and|or|with|of|for|the|a|an|as|required|taste)\b/g,'');
  s=s.replace(/\s+/g,' ').trim();
  return s;
}

function cleanLabel(text){
  if(!text) return '';
  var s=text.replace(/\s+/g,' ').trim();
  if(!s || s==='other') return '';
  s=s.replace(/\b(powder|paste|chopped|sliced|ground|crushed|fresh|dry|dried)\b/g,'');
  s=s.replace(/\s+/g,' ').trim();
  s=s.split(' ').filter(function(t){return t.length>1;}).slice(0,3).join(' ');
  s=s.replace(/\b([a-z])/g,function(m){return m.toUpperCase();});
  if(s.length>24) s=s.slice(0,22)+'…';
  return s;
}

function titleCase(text){
  return String(text||'').replace(/\b([a-z])/g,function(m){return m.toUpperCase();});
}

function ingredientCategory(ing){
  var s=String(ing||'');
  if(!s) return 'Other';
  if(/mutton|chicken|beef|pork|lamb|keema|mince|gosht|meat/.test(s)) return 'Meat';
  if(/prawn|fish|shrimp|seafood/.test(s)) return 'Seafood';
  if(/rice|wheat|grain|dal|lentil/.test(s)) return 'Grain & Pulse';
  if(/mint|coriander|cilantro|leaf|leaves|herb/.test(s)) return 'Herbs';
  if(/chilli|pepper|cardamom|clove|cinnamon|masala|saffron|nutmeg|mace|bay/.test(s)) return 'Spices';
  if(/yoghurt|curd|milk|cream|ghee|butter|cheese/.test(s)) return 'Dairy';
  if(/potato|onion|tomato|carrot|vegetable|jackfruit|eggplant|brinjal/.test(s)) return 'Vegetables';
  if(/lime|lemon|plum|apricot|orange|fruit/.test(s)) return 'Fruits';
  if(/cashew|almond|nut|raisin/.test(s)) return 'Nuts & Dry Fruit';
  return 'Other';
}

function buildTreeLayout(){
  var dishIngredients=[];
  var ingredientCounts={};
  var total=DATA.dishInfo.length||1;
  var CATEGORIES=['Meat','Seafood','Vegetables','Spices','Herbs','Dairy','Fruits','Nuts & Dry Fruit','Other'];
  var generic={
    'salt':1,'water':1,'oil':1,'ghee':1,'onion':1,'onions':1,'ginger':1,'garlic':1,
    'tomato':1,'tomatoes':1,'yoghurt':1,'milk':1,'cream':1,'coriander leaves':1,
    'mint leaves':1,'green chilli':1,'red chilli powder':1,'chilli powder':1
  };
  function isRice(ing){return ing.indexOf('rice')>=0;}
  function addCount(ing){ingredientCounts[ing]=(ingredientCounts[ing]||0)+1;}

  DATA.dishInfo.forEach(function(d,i){
    var list=(d.ingredients||[]).map(normalizeIngredient).filter(Boolean);
    var uniq=[];
    list.forEach(function(ing){if(uniq.indexOf(ing)===-1) uniq.push(ing);});
    dishIngredients[i]=uniq;
    uniq.forEach(addCount);
  });

  var commonIgnore={'salt':1,'water':1,'oil':1,'ghee':1};
  var commonList=Object.keys(ingredientCounts).filter(function(ing){
    return !isRice(ing) && !commonIgnore[ing];
  }).sort(function(a,b){return ingredientCounts[b]-ingredientCounts[a];})
    .map(cleanLabel).filter(Boolean)
    .filter(function(v,i,a){return a.indexOf(v)===i;})
    .slice(0,8);

  function pickPrimary(ings){
    var filtered=ings.filter(function(ing){return !generic[ing] && !isRice(ing);});
    if(!filtered.length) filtered=ings.filter(function(ing){return !isRice(ing) && ing!=='salt';});
    if(!filtered.length) return 'Other';
    var score={};
    filtered.forEach(function(ing){
      var c=ingredientCategory(ing);
      score[c]=(score[c]||0)+1;
    });
    var best='Other',bestV=-1;
    Object.keys(score).forEach(function(k){if(score[k]>bestV){bestV=score[k];best=k;}});
    return best;
  }

  var branchCounts={};
  var dishBranch=[];
  dishIngredients.forEach(function(ings,i){
    var primary=pickPrimary(ings);
    dishBranch[i]=primary;
    branchCounts[primary]=(branchCounts[primary]||0)+1;
  });

  var branchNames=CATEGORIES.filter(function(c){return (branchCounts[c]||0)>0;});
  if(branchNames.indexOf('Other')===-1 && branchCounts['Other']) branchNames.push('Other');
  if(!branchNames.length) branchNames=['Other'];

  var branchMap={};
  var branchNodes=[];
  var branchAngles={};
  var angleStep=(Math.PI*2)/branchNames.length;
  var branchColors=['#7aa96b','#9ac16e','#b7cf6f','#8fb685','#6fa37f','#a2b98a','#c3d09b','#7f9f6d','#9bb481','#7ea972','#97b590','#b1c48d','#98a96b'];

  branchNames.forEach(function(name,idx){
    var count=branchCounts[name]||1;
    var freq=count/total;
    var y=TREE.baseY+8+(1-freq)*12;
    y=Math.max(TREE.baseY+6,Math.min(TREE.canopyTop-2,y));
    var radius=TREE.branchRadius*(0.6+(1-freq)*0.6);
    var angle=idx*angleStep;
    var x=Math.cos(angle)*radius;
    var z=Math.sin(angle)*radius;
    branchAngles[name]=angle;
    branchMap[name]={x:x,y:y,z:z,angle:angle,color:branchColors[idx%branchColors.length]};
    branchNodes.push({name:name,x:x,y:y,z:z,angle:angle,color:branchColors[idx%branchColors.length]});
  });

  var commonNodes=[];
  commonList.forEach(function(name,idx){
    var r=2.6;
    var angle=idx*(Math.PI*2/Math.max(1,commonList.length));
    var y=TREE.baseY+4+idx*1.2;
    commonNodes.push({name:name,x:Math.cos(angle)*r,y:y,z:Math.sin(angle)*r});
  });

  var nodePositions=[];
  var lineSegments=[];

  commonNodes.forEach(function(n){
    lineSegments.push({a:{x:0,y:n.y,z:0},b:{x:n.x,y:n.y,z:n.z}});
  });

  branchNodes.forEach(function(n){
    lineSegments.push({a:{x:0,y:n.y,z:0},b:{x:n.x,y:n.y,z:n.z}});
  });

  DATA.dishInfo.forEach(function(d,i){
    var ings=dishIngredients[i]||[];
    var primary=dishBranch[i];
    if(branchNames.indexOf(primary)===-1) primary='Other';
    var branch=branchMap[primary];
    var rarity=0;
    if(ings.length){
      var sum=0;
      ings.forEach(function(ing){sum+=1-((ingredientCounts[ing]||1)/total);});
      rarity=sum/ings.length;
    }
    var y=branch.y+2+rarity*7+Math.random()*1.6;
    y=Math.min(TREE.canopyTop,y);
    var angle=branch.angle+(Math.random()-0.5)*0.9;
    var r=1.8+rarity*3.5;
    var x=branch.x+Math.cos(angle)*r;
    var z=branch.z+Math.sin(angle)*r;
    nodePositions[i]={x:x,y:y,z:z};
    lineSegments.push({a:{x:branch.x,y:branch.y,z:branch.z},b:{x:x,y:y,z:z}});
  });

  var linePositions=new Float32Array(lineSegments.length*6);
  lineSegments.forEach(function(seg,i){
    var j=i*6;
    linePositions[j]=seg.a.x;linePositions[j+1]=seg.a.y;linePositions[j+2]=seg.a.z;
    linePositions[j+3]=seg.b.x;linePositions[j+4]=seg.b.y;linePositions[j+5]=seg.b.z;
  });

  return {
    nodePositions:nodePositions,
    branchNodes:branchNodes,
    commonNodes:commonNodes,
    linePositions:linePositions,
    commonList:commonList,
    branchCounts:branchCounts,
    branchNames:branchNames
  };
}

var layout=buildTreeLayout();
var nodePositions=layout.nodePositions;

function renderTreeSummary(layout){
  var el=document.getElementById('tree-summary');
  if(!el) return;
  var common=(layout.commonList||[]);
  var branchNames=(layout.branchNames||[]);
  var counts=layout.branchCounts||{};
  var branchList=branchNames.slice().sort(function(a,b){return (counts[b]||0)-(counts[a]||0);});
  var html='<h3>Ingredient Map</h3>';
  html+='<div class="muted">Rice at base • branches by dominant ingredient family • leaves are biryanis</div>';
  html+='<h4>Common Ingredients</h4><ul>';
  if(common.length){
    common.forEach(function(name){html+='<li>'+titleCase(name)+'</li>';});
  }else{
    html+='<li class="muted">No common ingredients detected</li>';
  }
  html+='</ul>';
  html+='<h4>Branch Families</h4><ul>';
  branchList.forEach(function(name){html+='<li>'+name+' <span class="muted">('+ (counts[name]||0) +')</span></li>';});
  html+='</ul>';
  el.innerHTML=html;
}
renderTreeSummary(layout);

// Base rice
var baseGeo=new THREE.CircleGeometry(7,64);
var baseMat=new THREE.MeshPhongMaterial({color:0xf4e2c3,transparent:true,opacity:0.96,side:THREE.DoubleSide});
var base=new THREE.Mesh(baseGeo,baseMat);
base.rotation.x=-Math.PI/2;
base.position.y=TREE.baseY;
scene.add(base);

var baseGlow=new THREE.Mesh(new THREE.CircleGeometry(9,64),new THREE.MeshBasicMaterial({color:0xe7cfa4,transparent:true,opacity:0.08,side:THREE.DoubleSide}));
baseGlow.rotation.x=-Math.PI/2;
baseGlow.position.y=TREE.baseY+0.02;
scene.add(baseGlow);

// Trunk
var trunkGeo=new THREE.CylinderGeometry(1.2,1.6,TREE.trunkTop-TREE.baseY,18,1,true);
var trunkMat=new THREE.MeshPhongMaterial({color:0x6b4a2e,shininess:10,transparent:true,opacity:0.98});
var trunk=new THREE.Mesh(trunkGeo,trunkMat);
trunk.position.y=(TREE.baseY+TREE.trunkTop)/2;
scene.add(trunk);

// Canopy glow
var canopy=new THREE.Mesh(new THREE.SphereGeometry(11,28,28),new THREE.MeshBasicMaterial({color:0x466b3a,transparent:true,opacity:0.06}));
canopy.position.y=TREE.canopyTop-2;
scene.add(canopy);

// Ingredient nodes
layout.commonNodes.forEach(function(n){
  var node=new THREE.Mesh(new THREE.SphereGeometry(0.55,18,18),new THREE.MeshPhongMaterial({color:0xd9c199,emissive:0x3b2b1b,shininess:30,transparent:true,opacity:0.95}));
  node.position.set(n.x,n.y,n.z);
  scene.add(node);
});

layout.branchNodes.forEach(function(n){
  var col=new THREE.Color(n.color);
  var node=new THREE.Mesh(new THREE.SphereGeometry(0.7,20,20),new THREE.MeshPhongMaterial({color:col,emissive:col.clone().multiplyScalar(0.25),shininess:40,transparent:true,opacity:0.95}));
  node.position.set(n.x,n.y,n.z);
  scene.add(node);
});

// Connection lines
var lineMat=new THREE.LineBasicMaterial({color:0x9c8b6b,transparent:true,opacity:0});
var lineGeo=new THREE.BufferGeometry();
lineGeo.setAttribute('position',new THREE.BufferAttribute(layout.linePositions,3));
var lines=new THREE.LineSegments(lineGeo,lineMat);
scene.add(lines);

// Biryani leaves
var orbs=[];
DATA.nodes.forEach(function(n,i){
  var hex=n.color.replace('#','');
  var r=parseInt(hex.slice(0,2),16)/255;
  var g=parseInt(hex.slice(2,4),16)/255;
  var b=parseInt(hex.slice(4,6),16)/255;
  var col=new THREE.Color(r,g,b);

  var geo=new THREE.SphereGeometry(0.7,28,28);
  var mat=new THREE.MeshPhongMaterial({color:col,emissive:col.clone().multiplyScalar(0.4),shininess:70,transparent:true,opacity:0.96});
  var mesh=new THREE.Mesh(geo,mat);
  var pos=nodePositions[i];
  mesh.position.set(pos.x,pos.y,pos.z);
  mesh.userData={name:n.name,id:i,cuisine:n.cuisine,cluster:n.cluster,color:n.color,clusterLabel:n.clusterLabel||''};

  var glowGeo=new THREE.SphereGeometry(1.0,16,16);
  var glowMat=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.12});
  var glow=new THREE.Mesh(glowGeo,glowMat);
  mesh.add(glow);

  orbs.push(mesh);
  scene.add(mesh);
});

// Selection lines (clicked node to top 3 similar)
var selLinePos=new Float32Array(3*2*3);
var selLineGeo=new THREE.BufferGeometry();
selLineGeo.setAttribute('position',new THREE.BufferAttribute(selLinePos,3));
var selLineMat=new THREE.LineBasicMaterial({color:0xffdd88,linewidth:2,transparent:true,opacity:0.95});
var selLines=new THREE.LineSegments(selLineGeo,selLineMat);
selLines.visible=false;
scene.add(selLines);

// Lights
scene.add(new THREE.AmbientLight(0x3b2b1b,0.9));
var dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(20,30,10);scene.add(dl);
var pl=new THREE.PointLight(0xe89b3a,0.5,120);pl.position.set(0,10,0);scene.add(pl);

/* ======================== CAMERA INTRO (4 sec, single direction) ======================== */
var introActive=true;
var introStartPos=new THREE.Vector3(0,10,46);
var introEndPos=new THREE.Vector3(0,6,32);
var introDuration=4;
var introElapsed=0;

function updateIntro(dt){
  introElapsed+=dt;
  var t=Math.min(1,introElapsed/introDuration);
  camera.position.lerpVectors(introStartPos,introEndPos,t);
  camera.lookAt(0,4,0);
  lineMat.opacity=Math.min(1,introElapsed/2)*0.35;
  if(t>=1){ introActive=false; finishIntro(); }
}
function finishIntro(){
  introActive=false;
  document.getElementById('intro-label').classList.add('hide');
  document.getElementById('skip').classList.add('hide');
  lineMat.opacity=0.35;
  orbitEnabled=true;cameraRadius=34;updateCam();
}
document.getElementById('skip').onclick=finishIntro;

/* ======================== ORBIT CONTROLS ======================== */
var orbitCenter=new THREE.Vector3(0,4,0);
var orbitEnabled=false,cameraRadius=50,cTheta=0.3,cPhi=1.2,dragStart=null;
function updateCam(){
  var dx=cameraRadius*Math.sin(cPhi)*Math.cos(cTheta);
  var dy=cameraRadius*Math.cos(cPhi);
  var dz=cameraRadius*Math.sin(cPhi)*Math.sin(cTheta);
  camera.position.set(orbitCenter.x+dx,orbitCenter.y+dy,orbitCenter.z+dz);
  camera.lookAt(orbitCenter);
}

/* ======================== SELECTION + ZOOM ======================== */
var selectedIdx=null;
var zooming=false,zoomStartPos=new THREE.Vector3(),zoomTargetPos=new THREE.Vector3(),zoomLookAt=new THREE.Vector3(),zoomDuration=0.65,zoomT=0;
var ZOOM_DIST=14;

function updateSelectionLines(clickedId,similarIds){
  if(!similarIds||similarIds.length===0){selLines.visible=false;return}
  var c=nodePositions[clickedId];
  for(var i=0;i<3;i++){
    var s=nodePositions[similarIds[i]];
    if(!s){s=c}
    var j=i*6;
    selLinePos[j]=c.x;selLinePos[j+1]=c.y;selLinePos[j+2]=c.z;
    selLinePos[j+3]=s.x;selLinePos[j+4]=s.y;selLinePos[j+5]=s.z;
  }
  selLineGeo.attributes.position.needsUpdate=true;
  selLines.visible=true;
}
function setHighlight(idx,similarIds){
  orbs.forEach(function(o,i){
    if(i===idx||(similarIds&&similarIds.indexOf(i)>=0))
      o.scale.setScalar(1.35);
    else
      o.scale.setScalar(1);
  });
}
function clearHighlight(){
  orbs.forEach(function(o){o.scale.setScalar(1)});
  selLines.visible=false;
  selectedIdx=null;
}
function zoomToNode(idx){
  var n=nodePositions[idx];
  var nodePos=new THREE.Vector3(n.x,n.y,n.z);
  zoomStartPos.copy(camera.position);
  var dir=camera.position.clone().sub(nodePos);
  var len=dir.length();
  if(len<0.001) dir.set(0,2,12); else dir.normalize();
  zoomTargetPos.copy(nodePos).add(dir.multiplyScalar(ZOOM_DIST));
  zoomLookAt.copy(nodePos);
  zooming=true;
  zoomT=0;
}
function selectInViz(idx){
  selectedIdx=idx;
  var topSim=DATA.dishInfo[idx].topSimilar||[];
  var similarIds=topSim.slice(0,3).map(function(s){return s.id});
  updateSelectionLines(idx,similarIds);
  setHighlight(idx,similarIds);
  zoomToNode(idx);
}
function updateZoom(dt){
  if(!zooming) return;
  zoomT+=dt/zoomDuration;
  if(zoomT>=1){
    zooming=false;
    camera.position.copy(zoomTargetPos);
    camera.lookAt(zoomLookAt);
    orbitCenter.copy(zoomLookAt);
    cameraRadius=ZOOM_DIST;
    var d=camera.position.clone().sub(orbitCenter).normalize();
    cPhi=Math.acos(Math.max(-1,Math.min(1,d.y)));
    cTheta=Math.atan2(d.z,d.x);
    updateCam();
    return;
  }
  var t=1-Math.pow(1-zoomT,2);
  camera.position.lerpVectors(zoomStartPos,zoomTargetPos,t);
  camera.lookAt(zoomLookAt);
}
var didDrag=false;
canvas.onmousedown=function(e){if(!orbitEnabled||e.button!==0)return;didDrag=false;dragStart={x:e.clientX,y:e.clientY,t:cTheta,p:cPhi}};
canvas.onmousemove=function(e){
  if(dragStart){
    var dx=e.clientX-dragStart.x,dy=e.clientY-dragStart.y;
    if(Math.abs(dx)>3||Math.abs(dy)>3) didDrag=true;
    cTheta=dragStart.t+dx*0.007;
    cPhi=Math.max(0.15,Math.min(Math.PI-0.15,dragStart.p+dy*0.007));
    updateCam();
    document.getElementById('tooltip').classList.remove('show');
    return;
  }
  // Tooltip via raycaster (only when not dragging and in user mode)
  if(introActive) return;
  var mx=(e.clientX/innerWidth)*2-1;
  var my=-(e.clientY/innerHeight)*2+1;
  rc.setFromCamera(new THREE.Vector2(mx,my),camera);
  var hits=rc.intersectObjects(orbs,false);
  var tt=document.getElementById('tooltip');
  if(hits.length){
    var obj=hits[0].object;
    var u=obj.userData;
    if(!u||!u.name){tt.classList.remove('show');return}
    document.getElementById('tt-name').textContent=u.name;
    document.getElementById('tt-cuisine').textContent=u.cuisine||'';
    document.getElementById('tt-img').src=PLACEHOLDER_IMG;
    tt.style.left=Math.min(e.clientX+14,innerWidth-200)+'px';
    tt.style.top=Math.min(e.clientY+14,innerHeight-80)+'px';
    tt.classList.add('show');
  } else {
    tt.classList.remove('show');
  }
};
canvas.onmouseup=function(){dragStart=null;};
canvas.onmouseleave=function(){dragStart=null;document.getElementById('tooltip').classList.remove('show')};
canvas.onwheel=function(e){if(!orbitEnabled)return;e.preventDefault();cameraRadius=Math.max(8,Math.min(80,cameraRadius+e.deltaY*0.04));updateCam()};

/* ======================== CLICK TO SELECT ======================== */
var rc=new THREE.Raycaster();
canvas.onclick=function(e){
  if(introActive) return;
  if(didDrag) return;
  var mx=(e.clientX/innerWidth)*2-1;
  var my=-(e.clientY/innerHeight)*2+1;
  rc.setFromCamera(new THREE.Vector2(mx,my),camera);
  var hits=rc.intersectObjects(orbs);
  if(!hits.length) return;
  var idx=hits[0].object.userData.id;
  showDish(idx);
};

/* ======================== ANIMATION LOOP ======================== */
var clock=new THREE.Clock();
var elapsed=0;
function loop(){
  requestAnimationFrame(loop);
  var dt=Math.min(clock.getDelta(),0.1);
  elapsed+=dt;
  if(introActive) updateIntro(dt);
  else if(zooming) updateZoom(dt);
  // Gentle bob for orbs
  orbs.forEach(function(o,i){o.position.y=nodePositions[i].y+Math.sin(elapsed*0.6+i*0.7)*0.15});
  renderer.render(scene,camera);
}
loop();
window.onresize=function(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)};
})();
</script>
</body>
</html>"""
    )


if __name__ == "__main__":
    main()
