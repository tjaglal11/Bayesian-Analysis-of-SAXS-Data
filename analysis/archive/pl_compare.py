import pandas as pd

# Paths
weights_path_1 = "/Users/timothyjaglal/Desktop/Tau_teams_local/Data/structure_weights_sorted_MD003_t100.txt"
weights_path_2 = "/Users/timothyjaglal/Desktop/compare_weights/Sorted_Results_MD_Cterm.csv"
save_path = "/Users/timothyjaglal/Desktop/Tau_teams_local/Data"
cutoff = 0.001

def match_and_rank_weights():
    # 1. Read reference dataset using regex \s+ to handle inconsistent tab spacing
    # Skip the original header and assign strict column names
    w1_df = pd.read_csv(weights_path_1, sep=r'\s+', header=None, skiprows=1, names=["Index", "Ref t1000", "PDB File"])

    # Clean the PDB names
    w1_df["PDB File"] = w1_df["PDB File"].str.replace(".pdb", "", regex=False)

    # Drop the index column as it's no longer needed
    w1_df = w1_df[["PDB File", "Ref t1000"]]

    # 2. Read the NMR dataset
    w2_df = pd.read_csv(weights_path_2, header=None, names=["PDB File", "NMR"])

    # 3. Vectorized Left Join (Replaces the slow for-loop)
    # This aligns the entire dataset instantly
    combined_df = pd.merge(w1_df, w2_df, on="PDB File", how="left")

    # Fill missing NMR matches with 0.0 so the math calculates correctly
    combined_df["NMR"] = combined_df["NMR"].fillna(0.0)

    # 4. Calculate average
    combined_df["Average weight"] = (combined_df["Ref t1000"] + combined_df["NMR"]) / 2

    # 5. Apply cutoff, sort, and organize columns
    final_df = combined_df[combined_df["Average weight"] >= cutoff]
    final_df = final_df.sort_values(by="Average weight", ascending=False)

    return final_df

## Main ----------------
final = match_and_rank_weights()
print(final)

## Optional save file as csv
final.to_csv(f"{save_path}/compared_weights_cterm_t1000.csv", index=False)