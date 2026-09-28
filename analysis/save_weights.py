import os
import glob
import pandas as pd
import numpy as np
import re
from datetime import date
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml", help="Path to main YAML file")
args = parser.parse_args()

with open(args.config, "r") as f:
    master_config = yaml.safe_load(f)

sw_config = master_config.get("save_weights", {})

save_path = sw_config.get("save_path", "")
ibme_out_dir = sw_config.get("ibme_out_dir", "")
ty = sw_config.get("type", "")
custom_dro = sw_config.get("custom_dro", "")
custom_r0 = sw_config.get("custom_r0", "")

grid_file_path = os.path.join(save_path, "grid_full.txt")
grid_sum_path = os.path.join(ibme_out_dir, "GRID_sum.txt")
today = date.today()

# --- Reload Data ---
print("Loading previously computed grid data...")
GRID_DF = pd.read_csv(grid_file_path, sep=r'\s+', header=None, names=['index', 'dro', 'r0'])
grid = np.genfromtxt(grid_sum_path, skip_header=1, delimiter=',', filling_values=np.nan)

# --- Recalculate Best dro and r0 ---
chi2 = np.clip(grid[:,4], 1e-12, None)
phi  = np.clip(grid[:,5], 1e-12, None)
gamma = np.log(chi2 / phi)

# Find the exact row index of the minimum gamma
best_idx = np.nanargmin(gamma)
best_dro = grid[best_idx, 1]
best_r0 = grid[best_idx, 2]

print(f"Recovered Best Parameters -> δρ={best_dro:.2f}, r0={best_r0:.3f}")

#Execute SAVE Logic
if ty == "custom":
    match = GRID_DF.index[np.isclose(GRID_DF['dro'], float(custom_dro)) & np.isclose(GRID_DF['r0'], float(custom_r0))].tolist()
    if not match:
        raise ValueError(f"Grid point (dro={custom_dro}, r0={custom_r0}) not found in {grid_file_path}")
    weight_idx = match[0]
else:
    weight_idx = int(grid[best_idx, 0])

best_gp_dir = os.path.join(ibme_out_dir, f"GP{weight_idx}")

#Dynamically find the last .weights.dat file
weight_files = glob.glob(os.path.join(best_gp_dir, "*.weights.dat"))
if not weight_files:
    raise FileNotFoundError(f"No .weights.dat files found in {best_gp_dir}")

weight_files_sorted = sorted(weight_files, key=lambda x: int(re.search(r"_(\d+)\.weights\.dat", os.path.basename(x)).group(1)))
best_weight_file = weight_files_sorted[-1]

#Use the manifest written by concat_fractions: row i of the iBME calc file <-> line i of the manifest
manifest_path = os.path.join(save_path, "compiled_GPs", f"GP{weight_idx}_manifest.txt")
if not os.path.isfile(manifest_path):
    raise FileNotFoundError(f"Missing manifest {manifest_path}. Re-run the grid with do_gp_fraction_e.sh so calc_saxs_pdb.txt is written.")
contents = pd.read_csv(manifest_path, header=None)

#Map and save
opt_weight = pd.read_csv(best_weight_file, sep=r'\s+', header=None)
if opt_weight.empty or len(opt_weight.columns) < 2:
    raise ValueError(f"Weight file {best_weight_file} is empty or has insufficient columns")

if len(contents) != len(opt_weight):
    raise ValueError(f"Manifest has {len(contents)} structures but {best_weight_file} has {len(opt_weight)} weights.")

if set(opt_weight.iloc[:, 0].astype(int)) != set(range(len(contents))):
    raise ValueError("Weight indices are not 0..N-1; cannot map to manifest by position.")

opt_weight['PDB_Name'] = opt_weight.iloc[:, 0].astype(int).map(contents.iloc[:, 0])
opt_sorted = opt_weight.sort_values(by=1, ascending=False)

if ty == "custom":
    weights_out = os.path.join(ibme_out_dir, f'structure_weights_sorted_{custom_dro}_{custom_r0}.txt')
else:
    weights_out = os.path.join(ibme_out_dir, f'structure_weights_sorted_{today}.txt')

opt_sorted.to_csv(weights_out, index=None, sep='\t')

print(f"Success! Top structure weights saved to: {weights_out}")
