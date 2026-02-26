"""
Extract relevant biryani data from sorted.xlsx and write one Excel file
(biryani_similarity_data.xlsx) with all sheets needed for similarity or other use.
"""

import pandas as pd
from pathlib import Path

INPUT_EXCEL = Path(__file__).resolve().parent / "sorted.xlsx"
OUTPUT_EXCEL = Path(__file__).resolve().parent / "biryani_similarity_data.xlsx"


def load_sheet(name: str) -> pd.DataFrame:
    """Load a sheet from the source Excel; return empty DataFrame if missing."""
    try:
        return pd.read_excel(INPUT_EXCEL, sheet_name=name)
    except Exception as e:
        print(f"Warning: Could not load sheet '{name}': {e}")
        return pd.DataFrame()


def main():
    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_EXCEL}")

    print("Loading sheets from", INPUT_EXCEL)

    # --- Core tables ---
    dish = load_sheet("DISH")
    dish_ingredient = load_sheet("DISH_INGREDIENT")
    taste_profile_1 = load_sheet("DISH_TASTE_PROFILE_1")
    taste_profile_2 = load_sheet("DISH_TASTE_PROFILE_2")
    cooking_step = load_sheet("COOKING_STEP")
    taste_descriptor = load_sheet("TASTE_DESCRIPTOR")
    ingredient = load_sheet("INGREDIENT")
    ingredient_compound = load_sheet("INGREDIENT_CHEMICAL_COMPOUND")
    compound_synergy_rule = load_sheet("COMPOUND_SYNERGY_RULE")
    compound_synergy_member = load_sheet("COMPOUND_SYNERGY_MEMBER")
    ingredient_taste = load_sheet("INGREDIENT_TASTE")
    cooking_style_master = load_sheet("COOKING_STYLE")
    step_ingredient = load_sheet("STEP_INGREDIENT")

    # --- 1. Dish list (reference for picking 2 biryanis) ---
    dish_list = dish[["dish_id", "dish_name"]].drop_duplicates().sort_values("dish_id").reset_index(drop=True)
    dish_list = dish_list.rename(columns={"dish_id": "dish_id", "dish_name": "dish_name"})

    def _taste_sheet(profile: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Build one taste profile sheet: only rows with intensity > 0, add dish_id."""
        if profile.empty or "intensity" not in profile.columns:
            return pd.DataFrame()
        out = profile.loc[profile["intensity"].fillna(0) > 0].copy()
        out = out.dropna(subset=["dish_name", "descriptor_name"])
        if not dish_list.empty and "dish_id" not in out.columns:
            out = out.merge(dish_list[["dish_name", "dish_id"]], on="dish_name", how="left")
        return out.sort_values(["dish_name", "descriptor_name"]).reset_index(drop=True)

    # --- 2a. Taste profile 1 (non-zero intensity only) ---
    taste_sheet_1 = _taste_sheet(taste_profile_1, "Taste_Profile_1")
    # --- 2b. Taste profile 2 (non-zero intensity only) ---
    taste_sheet_2 = _taste_sheet(taste_profile_2, "Taste_Profile_2")

    # --- 3. Ingredients by dish (for "ingredients common" score) ---
    ing_cols = ["dish_id", "dish_name", "ingredient_id", "ingredient_name", "quantity", "unit_of_measure", "preparation_method"]
    ing_cols = [c for c in ing_cols if c in dish_ingredient.columns]
    ingredients_by_dish = dish_ingredient[ing_cols].drop_duplicates().sort_values(["dish_id", "ingredient_id"]).reset_index(drop=True)

    # --- 4. Cooking style by dish (aggregated) ---
    if not cooking_step.empty and "cooking_style_name" in cooking_step.columns:
        style_cols = ["dish_id", "dish_name", "cooking_style_name"]
        style_cols = [c for c in style_cols if c in cooking_step.columns]
        cooking_style_by_dish = cooking_step[style_cols].drop_duplicates().sort_values(["dish_id", "cooking_style_name"]).reset_index(drop=True)
    else:
        cooking_style_by_dish = pd.DataFrame(columns=["dish_id", "dish_name", "cooking_style_name"])

    # --- 4b. Cooking steps (step-by-step per dish) ---
    step_cols = [c for c in ["dish_id", "dish_name", "step_number", "description", "cooking_style_name", "duration_minutes", "temperature_c"] if c in cooking_step.columns]
    cooking_steps = cooking_step[step_cols].sort_values(["dish_id", "step_number"]).reset_index(drop=True) if not cooking_step.empty else pd.DataFrame()

    # --- 5. Dish metadata (cuisine, category, difficulty, times) ---
    meta_cols = ["dish_id", "dish_name", "dish_category", "cuisine_type", "difficulty_level",
                 "min_prep_time_minutes", "max_prep_time_minutes",
                 "min_marination_time_minutes", "max_marination_time_minutes",
                 "min_total_cooking_time_minutes", "max_total_cooking_time_minutes", "serves_count"]
    meta_cols = [c for c in meta_cols if c in dish.columns]
    dish_metadata = dish[meta_cols].copy().sort_values("dish_id").reset_index(drop=True)

    # --- 6. Taste descriptor reference (for labelling flavour dimensions) ---
    desc_cols = [c for c in ("taste_descriptor_id", "descriptor_name", "category_name", "category_id", "recognition_threshold", "base_harmony_score") if c in taste_descriptor.columns]
    taste_descriptor_ref = taste_descriptor[desc_cols].drop_duplicates() if not taste_descriptor.empty else pd.DataFrame()

    # --- 7. Ingredient master (for lookup when comparing ingredients) ---
    ing_master_cols = [c for c in ("ingredient_id", "ingredient_name", "origin") if c in ingredient.columns]
    ingredient_master = ingredient[ing_master_cols].drop_duplicates().sort_values("ingredient_id").reset_index(drop=True) if not ingredient.empty else pd.DataFrame()

    # --- 8. Ingredient scientific / dominant compounds (pH, primary_compounds, composition) ---
    sci_cols = [c for c in ("ingredient_id", "ingredient_name", "primary_compounds", "pH_value", "moisture_percentage",
                            "fat_percentage", "protein_percentage", "carbs_percentage", "maillard_responsive", "origin", "source") if c in ingredient.columns]
    ingredient_scientific = ingredient[sci_cols].drop_duplicates().sort_values("ingredient_id").reset_index(drop=True) if not ingredient.empty and sci_cols else pd.DataFrame()

    # --- 9. Ingredient–chemical compound (compound per ingredient: name, concentration, descriptor, intensity) ---
    icc_cols = [c for c in ("ingredient_compound_id", "ingredient_id", "ingredient_name", "compound_name", "concentration_percentage",
                            "descriptor_name", "contributes_to_descriptor_id", "contribution_intensity", "volatile_at_temperature_c",
                            "stable_in_acidic", "stable_in_alkaline", "pH_loss_factor", "sources") if c in ingredient_compound.columns]
    ingredient_chemical_compound = ingredient_compound[icc_cols].drop_duplicates().sort_values(["ingredient_id", "compound_name"]).reset_index(drop=True) if not ingredient_compound.empty and icc_cols else pd.DataFrame()

    # --- 10. Compounds per dish (each biryani’s ingredients → their compounds; dominant compounds per biryani) ---
    if not dish_ingredient.empty and not ingredient_compound.empty and "ingredient_id" in dish_ingredient.columns and "ingredient_id" in ingredient_compound.columns:
        di = dish_ingredient[["dish_id", "dish_name", "ingredient_id", "ingredient_name"]].drop_duplicates()
        ic = ingredient_compound[["ingredient_id", "compound_name", "concentration_percentage", "descriptor_name", "contribution_intensity", "volatile_at_temperature_c"]].copy()
        dish_compounds = di.merge(ic, on="ingredient_id", how="inner")
        dish_compounds = dish_compounds.sort_values(["dish_id", "ingredient_name", "compound_name"]).reset_index(drop=True)
    else:
        dish_compounds = pd.DataFrame(columns=["dish_id", "dish_name", "ingredient_id", "ingredient_name", "compound_name", "concentration_percentage", "descriptor_name", "contribution_intensity", "volatile_at_temperature_c"])

    # --- 11. Compound synergy rules (reference: which compound combos amplify which descriptors) ---
    csr_cols = [c for c in ("synergy_rule_id", "synergy_name", "synergy_type", "amplification_factor", "affected_descriptor_name",
                            "affected_descriptor_id", "minimum_compounds_required", "amplification_mechanism", "Source") if c in compound_synergy_rule.columns]
    compound_synergy_rules = compound_synergy_rule[csr_cols].drop_duplicates() if not compound_synergy_rule.empty and csr_cols else pd.DataFrame()

    # --- 12. Compound synergy members (which compounds belong to which synergy rule) ---
    csm_cols = [c for c in compound_synergy_member.columns if not c.startswith("Unnamed") and "synergy_rule_name.1" != c]
    compound_synergy_members = compound_synergy_member[csm_cols].drop_duplicates() if not compound_synergy_member.empty else pd.DataFrame()

    # --- 13. Ingredient taste profile (taste descriptors per ingredient) ---
    it_cols = [c for c in ingredient_taste.columns if not c.startswith("Unnamed")] if not ingredient_taste.empty else []
    ingredient_taste_df = ingredient_taste[it_cols].drop_duplicates() if it_cols else pd.DataFrame()

    # --- 14. Cooking style master (style details: temp, duration, browning, moisture) ---
    cs_cols = [c for c in cooking_style_master.columns if not c.startswith("Unnamed")] if not cooking_style_master.empty else []
    cooking_style_master_df = cooking_style_master[cs_cols].drop_duplicates() if cs_cols else pd.DataFrame()

    # --- 15. Step ingredients (which ingredients are used at which cooking step) ---
    si_cols = [c for c in step_ingredient.columns if not c.startswith("Unnamed")] if not step_ingredient.empty else []
    step_ingredient_df = step_ingredient[si_cols].drop_duplicates() if si_cols else pd.DataFrame()

    # --- Write all to one Excel ---
    sheet_names = ["Dish_List", "Taste_Profile_1", "Taste_Profile_2", "Ingredients_By_Dish", "Cooking_Style_By_Dish", "Cooking_Steps",
                   "Dish_Metadata", "Taste_Descriptor_Reference", "Ingredient_Master", "Ingredient_Scientific",
                   "Ingredient_Chemical_Compound", "Dish_Compounds", "Compound_Synergy_Rule", "Compound_Synergy_Member",
                   "Ingredient_Taste", "Cooking_Style_Master", "Step_Ingredient"]
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        dish_list.to_excel(writer, sheet_name="Dish_List", index=False)
        if not taste_sheet_1.empty:
            taste_sheet_1.to_excel(writer, sheet_name="Taste_Profile_1", index=False)
        if not taste_sheet_2.empty:
            taste_sheet_2.to_excel(writer, sheet_name="Taste_Profile_2", index=False)
        ingredients_by_dish.to_excel(writer, sheet_name="Ingredients_By_Dish", index=False)
        cooking_style_by_dish.to_excel(writer, sheet_name="Cooking_Style_By_Dish", index=False)
        cooking_steps.to_excel(writer, sheet_name="Cooking_Steps", index=False)
        dish_metadata.to_excel(writer, sheet_name="Dish_Metadata", index=False)
        taste_descriptor_ref.to_excel(writer, sheet_name="Taste_Descriptor_Reference", index=False)
        ingredient_master.to_excel(writer, sheet_name="Ingredient_Master", index=False)
        if not ingredient_scientific.empty:
            ingredient_scientific.to_excel(writer, sheet_name="Ingredient_Scientific", index=False)
        if not ingredient_chemical_compound.empty:
            ingredient_chemical_compound.to_excel(writer, sheet_name="Ingredient_Chemical_Compound", index=False)
        dish_compounds.to_excel(writer, sheet_name="Dish_Compounds", index=False)
        if not compound_synergy_rules.empty:
            compound_synergy_rules.to_excel(writer, sheet_name="Compound_Synergy_Rule", index=False)
        if not compound_synergy_members.empty:
            compound_synergy_members.to_excel(writer, sheet_name="Compound_Synergy_Member", index=False)
        if not ingredient_taste_df.empty:
            ingredient_taste_df.to_excel(writer, sheet_name="Ingredient_Taste", index=False)
        if not cooking_style_master_df.empty:
            cooking_style_master_df.to_excel(writer, sheet_name="Cooking_Style_Master", index=False)
        if not step_ingredient_df.empty:
            step_ingredient_df.to_excel(writer, sheet_name="Step_Ingredient", index=False)

    print("Written to", OUTPUT_EXCEL)
    print("Sheets:")
    for sh in sheet_names:
        try:
            df = pd.read_excel(OUTPUT_EXCEL, sheet_name=sh)
            print(f"  - {sh}: {len(df)} rows")
        except Exception:
            print(f"  - {sh}: (empty or missing)")
    return OUTPUT_EXCEL


if __name__ == "__main__":
    main()
