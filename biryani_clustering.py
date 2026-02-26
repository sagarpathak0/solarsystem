"""
Biryani clustering and final similarity score using biryani_similarity_data.xlsx.

Runs separate clusters per feature (ingredient, flavour 1, flavour 2, cooking style,
compound, cuisine, cooking time). Then computes a single weighted final similarity
for every pair of biryanis. Adjust weights in WEIGHTS below.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG: adjust weights (must sum to 1.0) and number of clusters per feature
# ---------------------------------------------------------------------------
WEIGHTS = {
    "ingredient": 0.25,      # ingredient overlap (Jaccard)
    "flavour1": 0.20,        # taste profile 1 intensity vectors
    "flavour2": 0.20,       # taste profile 2 intensity vectors
    "cooking_style": 0.15,  # cooking methods used
    "compound": 0.10,       # chemical compounds in dish
    "cuisine": 0.05,        # cuisine type match
    "cooking_time": 0.05,   # total cooking time similarity
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"

N_CLUSTERS = {
    "ingredient": 8,
    "flavour1": 6,
    "flavour2": 6,
    "cooking_style": 6,
    "compound": 8,
    "cuisine": 6,
    "cooking_time": 5,
    "overall": 8,  # composite cluster from final weighted similarity
}

DATA_PATH = Path(__file__).resolve().parent / "biryani_similarity_data.xlsx"
OUTPUT_PATH = Path(__file__).resolve().parent / "biryani_clusters_and_similarity.xlsx"


def load_sheet(name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(DATA_PATH, sheet_name=name)
    except Exception as e:
        print(f"Warning: Could not load '{name}': {e}")
        return pd.DataFrame()


def jaccard_similarity_matrix(sets_per_dish: dict, dish_ids: list) -> np.ndarray:
    """Pairwise Jaccard similarity: |A cap B| / |A cup B|."""
    n = len(dish_ids)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = sets_per_dish.get(dish_ids[i], set())
            b = sets_per_dish.get(dish_ids[j], set())
            if not a and not b:
                sim = 1.0
            elif not a or not b:
                sim = 0.0
            else:
                inter = len(a & b)
                union = len(a | b)
                sim = inter / union if union else 0.0
            mat[i, j] = mat[j, i] = sim
    return mat


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print("Loading", DATA_PATH)
    dish_list = load_sheet("Dish_List")
    ingredients_by_dish = load_sheet("Ingredients_By_Dish")
    taste1 = load_sheet("Taste_Profile_1")
    taste2 = load_sheet("Taste_Profile_2")
    cooking_style_by_dish = load_sheet("Cooking_Style_By_Dish")
    dish_compounds = load_sheet("Dish_Compounds")
    dish_metadata = load_sheet("Dish_Metadata")

    dish_ids = dish_list["dish_id"].astype(int).tolist()
    id_to_idx = {did: i for i, did in enumerate(dish_ids)}
    n = len(dish_ids)
    print(f"Found {n} dishes.")

    # --- 1. Ingredient similarity (sets of ingredient_id per dish) ---
    print("Computing ingredient similarity...")
    ing_sets = {}
    for _, row in ingredients_by_dish.iterrows():
        did = row.get("dish_id")
        if pd.isna(did):
            continue
        did = int(did)
        iid = row.get("ingredient_id")
        if pd.isna(iid):
            continue
        ing_sets.setdefault(did, set()).add(int(iid))
    for did in dish_ids:
        ing_sets.setdefault(did, set())
    S_ingredient = jaccard_similarity_matrix(ing_sets, dish_ids)
    print("Computing flavour 1 & 2...")
    # --- 2. Flavour 1 & 2: dish x descriptor intensity matrix, then cosine ---
    id_to_name = dish_list.set_index("dish_id")["dish_name"].to_dict()

    def taste_to_matrix(taste_df, id_to_idx, dish_ids, id_to_name):
        if taste_df.empty or "dish_name" not in taste_df.columns or "descriptor_name" not in taste_df.columns:
            return np.eye(len(dish_ids))
        pivot = taste_df.pivot_table(
            index="dish_name", columns="descriptor_name", values="intensity",
            aggfunc="mean", fill_value=0
        )
        descriptors = pivot.columns.tolist()
        n_d = len(dish_ids)
        mat = np.zeros((n_d, len(descriptors)))
        for i, did in enumerate(dish_ids):
            name = id_to_name.get(did)
            if name and name in pivot.index:
                mat[i, :] = pivot.loc[name].values
        sim = cosine_similarity(mat)
        sim = np.clip(sim, 0, 1)
        np.fill_diagonal(sim, 1.0)
        return sim

    S_flavour1 = taste_to_matrix(taste1, id_to_idx, dish_ids, id_to_name)
    S_flavour2 = taste_to_matrix(taste2, id_to_idx, dish_ids, id_to_name)
    print("Computing cooking style & compound similarity...")
    # --- 3. Cooking style: set of style names per dish ---
    style_sets = {}
    for _, row in cooking_style_by_dish.iterrows():
        did = row.get("dish_id")
        if pd.isna(did):
            continue
        did = int(did)
        style = row.get("cooking_style_name")
        if pd.notna(style) and str(style).strip():
            style_sets.setdefault(did, set()).add(str(style).strip())
    for did in dish_ids:
        style_sets.setdefault(did, set())
    S_cooking_style = jaccard_similarity_matrix(style_sets, dish_ids)

    # --- 4. Compound: set of compound_name per dish ---
    compound_sets = {}
    for _, row in dish_compounds.iterrows():
        did = row.get("dish_id")
        if pd.isna(did):
            continue
        did = int(did)
        c = row.get("compound_name")
        if pd.notna(c) and str(c).strip():
            compound_sets.setdefault(did, set()).add(str(c).strip())
    for did in dish_ids:
        compound_sets.setdefault(did, set())
    S_compound = jaccard_similarity_matrix(compound_sets, dish_ids)
    print("Computing cuisine & cooking time similarity...")
    # --- 5. Cuisine: 1 if same cuisine, else 0 ---
    cuisine_per_dish = {}
    if not dish_metadata.empty and "cuisine_type" in dish_metadata.columns:
        for _, row in dish_metadata.iterrows():
            did = row.get("dish_id")
            if pd.notna(did):
                cuisine_per_dish[int(did)] = str(row.get("cuisine_type", "") or "").strip()
    for did in dish_ids:
        cuisine_per_dish.setdefault(did, "")
    S_cuisine = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = cuisine_per_dish.get(dish_ids[i], ""), cuisine_per_dish.get(dish_ids[j], "")
            sim = 1.0 if c1 and c2 and c1 == c2 else 0.0
            S_cuisine[i, j] = S_cuisine[j, i] = sim

    # --- 6. Cooking time: 1 - normalized difference of mean total time ---
    time_per_dish = {}
    if not dish_metadata.empty:
        for _, row in dish_metadata.iterrows():
            did = row.get("dish_id")
            if pd.isna(did):
                continue
            mn = row.get("min_total_cooking_time_minutes")
            mx = row.get("max_total_cooking_time_minutes")
            t = np.nanmean([mn, mx]) if (pd.notna(mn) or pd.notna(mx)) else np.nan
            time_per_dish[int(did)] = t
    for did in dish_ids:
        time_per_dish.setdefault(did, np.nan)
    times = np.array([time_per_dish[did] for did in dish_ids])
    valid = np.isfinite(times)
    if valid.any():
        t_min, t_max = times[valid].min(), times[valid].max()
        if t_max > t_min:
            times_norm = (times - t_min) / (t_max - t_min)
            times_norm[~valid] = np.nanmedian(times_norm[valid])
        else:
            times_norm = np.ones(n)
    else:
        times_norm = np.zeros(n)
    # Similarity = 1 - |a - b|
    S_cooking_time = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(times_norm[i] - times_norm[j])
            sim = 1.0 - min(diff, 1.0)
            S_cooking_time[i, j] = S_cooking_time[j, i] = sim

    # --- Weighted final similarity ---
    S_final = (
        WEIGHTS["ingredient"] * S_ingredient
        + WEIGHTS["flavour1"] * S_flavour1
        + WEIGHTS["flavour2"] * S_flavour2
        + WEIGHTS["cooking_style"] * S_cooking_style
        + WEIGHTS["compound"] * S_compound
        + WEIGHTS["cuisine"] * S_cuisine
        + WEIGHTS["cooking_time"] * S_cooking_time
    )
    np.fill_diagonal(S_final, 1.0)
    S_final = np.clip(S_final, 0, 1)
    print("Running clustering per feature...")
    # --- Clustering: use distance = 1 - similarity ---
    def cluster_from_similarity(S, n_clusters):
        D = 1 - np.clip(S, 0, 1)
        np.fill_diagonal(D, 0)
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        cond = squareform(D, checks=False)
        # Ward-like: use complete linkage for more balanced clusters
        Z = linkage(cond, method="complete")
        return fcluster(Z, n_clusters, criterion="maxclust")

    labels = {}
    for key, n_clu in N_CLUSTERS.items():
        if key == "ingredient":
            S = S_ingredient
        elif key == "flavour1":
            S = S_flavour1
        elif key == "flavour2":
            S = S_flavour2
        elif key == "cooking_style":
            S = S_cooking_style
        elif key == "compound":
            S = S_compound
        elif key == "cuisine":
            S = S_cuisine
        elif key == "cooking_time":
            S = S_cooking_time
        else:
            continue
        lab = cluster_from_similarity(S, n_clu)
        labels[key] = lab

    # --- Overall composite cluster: spectral clustering for balanced groups ---
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=N_CLUSTERS["overall"],
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=20,
    )
    labels["overall"] = sc.fit_predict(np.clip(S_final, 0, 1)) + 1  # 1-indexed

    # --- Build output DataFrames ---
    name_per_id = dish_list.set_index("dish_id")["dish_name"].to_dict()
    cluster_rows = []
    for i, did in enumerate(dish_ids):
        row = {"dish_id": did, "dish_name": name_per_id.get(did, "")}
        for key in labels:
            row[f"cluster_{key}"] = int(labels[key][i])
        cluster_rows.append(row)
    df_clusters = pd.DataFrame(cluster_rows)

    # Final similarity: long form (dish_a, dish_b, final_score + per-feature scores)
    rows_sim = []
    for i in range(n):
        for j in range(i + 1, n):
            rows_sim.append({
                "dish_id_a": dish_ids[i],
                "dish_name_a": name_per_id.get(dish_ids[i], ""),
                "dish_id_b": dish_ids[j],
                "dish_name_b": name_per_id.get(dish_ids[j], ""),
                "final_similarity": round(float(S_final[i, j]), 4),
                "sim_ingredient": round(S_ingredient[i, j], 4),
                "sim_flavour1": round(S_flavour1[i, j], 4),
                "sim_flavour2": round(S_flavour2[i, j], 4),
                "sim_cooking_style": round(S_cooking_style[i, j], 4),
                "sim_compound": round(S_compound[i, j], 4),
                "sim_cuisine": round(S_cuisine[i, j], 4),
                "sim_cooking_time": round(S_cooking_time[i, j], 4),
            })
    df_pairs = pd.DataFrame(rows_sim).sort_values("final_similarity", ascending=False).reset_index(drop=True)

    # --- Auto-label overall clusters by dominant traits ---
    overall_labels = labels["overall"]
    cluster_meta_rows = []
    for cid in sorted(set(overall_labels)):
        members = [i for i, c in enumerate(overall_labels) if c == cid]
        member_ids = [dish_ids[i] for i in members]
        member_names = [name_per_id.get(d, "") for d in member_ids]

        # Dominant cuisine
        cuisine_counts = {}
        for did in member_ids:
            row = dish_metadata[dish_metadata["dish_id"] == did]
            if not row.empty and pd.notna(row["cuisine_type"].iloc[0]):
                c = str(row["cuisine_type"].iloc[0])
                cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
        top_cuisine = max(cuisine_counts, key=cuisine_counts.get) if cuisine_counts else ""

        # Dominant cooking styles
        style_counts = {}
        for did in member_ids:
            rows = cooking_style_by_dish[cooking_style_by_dish["dish_id"] == did] if not cooking_style_by_dish.empty else pd.DataFrame()
            for _, r in rows.iterrows():
                s = str(r.get("cooking_style_name", ""))
                if s and s != "nan":
                    style_counts[s] = style_counts.get(s, 0) + 1
        top_styles = sorted(style_counts, key=style_counts.get, reverse=True)[:2]

        # Most shared ingredients (appear in >60% of cluster members)
        ing_counts = {}
        for did in member_ids:
            rows = ingredients_by_dish[ingredients_by_dish["dish_id"] == did] if not ingredients_by_dish.empty else pd.DataFrame()
            seen = set()
            for _, r in rows.iterrows():
                nm = str(r.get("ingredient_name", ""))
                if nm and nm != "nan" and nm not in seen:
                    ing_counts[nm] = ing_counts.get(nm, 0) + 1
                    seen.add(nm)
        threshold = max(1, len(members) * 0.6)
        shared_ings = sorted([k for k, v in ing_counts.items() if v >= threshold],
                             key=lambda k: -ing_counts[k])[:4]

        # Build a readable label: cuisine (if dominant) + key ingredient + cooking style
        parts = []
        if top_cuisine:
            pct = cuisine_counts[top_cuisine] / len(members)
            if pct >= 0.5:
                parts.append(top_cuisine)
        # Add a distinguishing ingredient (skip generic ones like salt, rice, oil, water)
        generic = {"salt", "rice", "oil", "water", "onions", "ginger", "garlic", "ghee"}
        signature_ing = [i for i in shared_ings if i.lower() not in generic][:1]
        if signature_ing:
            parts.append(signature_ing[0])
        if top_styles:
            parts.append(top_styles[0])
        label = " · ".join(parts) if parts else f"Group {cid}"

        cluster_meta_rows.append({
            "cluster_id": int(cid),
            "label": label,
            "size": len(members),
            "dominant_cuisine": top_cuisine,
            "top_cooking_styles": ", ".join(top_styles),
            "shared_ingredients": ", ".join(shared_ings),
            "members": ", ".join(member_names),
        })

    # Ensure unique labels (append cluster id suffix on collision)
    seen_labels = {}
    for r in cluster_meta_rows:
        lbl = r["label"]
        if lbl in seen_labels:
            seen_labels[lbl] += 1
            r["label"] = f"{lbl} ({seen_labels[lbl]})"
        else:
            seen_labels[lbl] = 1

    df_cluster_meta = pd.DataFrame(cluster_meta_rows)
    df_clusters["cluster_overall_label"] = df_clusters["cluster_overall"].map(
        {r["cluster_id"]: r["label"] for r in cluster_meta_rows}
    )

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df_clusters.to_excel(writer, sheet_name="Cluster_Labels", index=False)
        df_pairs.to_excel(writer, sheet_name="Pairwise_Final_Similarity", index=False)
        df_cluster_meta.to_excel(writer, sheet_name="Cluster_Meta", index=False)

    print("Written to", OUTPUT_PATH)
    print("\nCluster summary:")
    for r in cluster_meta_rows:
        print(f"  [{r['cluster_id']}] {r['label']} ({r['size']} dishes) — {r['shared_ingredients'][:60]}")
    print("\nWeights used:", WEIGHTS)
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
