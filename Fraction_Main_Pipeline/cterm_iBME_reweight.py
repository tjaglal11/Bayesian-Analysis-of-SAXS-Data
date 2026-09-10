import os
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
import re
import shutil
import matplotlib.pyplot as plt
from datetime import date
from natsort import natsorted
import iBME_script
import polars as pl

#########--------- ARGUMENTS
parser = argparse.ArgumentParser(description="Run iBME reweighting on fractions")
parser.add_argument("theta", type=float)
parser.add_argument("save_path", type=str)
exp_path = "/users/t/j/tjaglal/experimental_data/SASDLS4.dat"
trun_path = "/users/t/j/tjaglal/experimental_data/SASDLS4_trun_edit.dat"
struct_path = "/users/t/j/tjaglal/cterm_structures"
args = parser.parse_args()


today = date.today()

#########---------- FUNCTIONS

def concat_fractions(save_path):
    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None, names=['index', 'dro', 'r0'])
    num_gps = len(grid_df)
    compiled_dir = os.path.join(save_path, "compiled_GPs")
    os.makedirs(compiled_dir, exist_ok=True)

    print(f"Found {num_gps} grid points. Starting concatenation...")
    for i in range(num_gps):
        search_pattern = os.path.join(save_path, "MD*_*", f"GP{i}", "calc_saxs.txt")
        found_files = glob.glob(search_pattern)
        if not found_files:
            continue

        sorted_files = natsorted(found_files)
        compiled_data = []
        pdb_names = []
        for file in sorted_files:
            md_folder = os.path.basename(os.path.dirname(os.path.dirname(file)))

            df = pd.read_csv(file, sep='\s+', header=None)
            compiled_data.append(df)

            pdb_search = os.path.join(struct_path, "MD*", md_folder, "*.pdb")
            pdbs = natsorted(glob.glob(pdb_search))
            pdb_names.extend(os.path.basename(p) for p in pdbs)

            if len(pdbs) != len(df):
                print(f"Warning: {len(pdbs)} PDBs found for {md_folder}, but {len(df)} frames in {file}.")

        final_df = pd.concat(compiled_data, ignore_index=True)
        output_file = os.path.join(compiled_dir, f"GP{i}_all_saxs.txt")
        final_df.to_csv(output_file, sep=' ', index=False, header=False)

        if len(final_df) != len(pdb_names):
            raise ValueError(f"GP{i} has {len(final_df)} calculation rows but {len(pdb_names)} PDB names.")
        named_df = final_df.copy()
        named_df.iloc[:, 0] = pdb_names
        named_df.to_csv(
            os.path.join(compiled_dir, f"GP{i}_all_saxs_pdb.txt"), sep=' ', index=False, header=False
        )

        manifest_file = os.path.join(compiled_dir, f"GP{i}_manifest.txt")
        pd.Series(pdb_names).to_csv(manifest_file, header=None, index=False)

# ========= 2. RUN iBME LOOP =========
concat_fractions(args.save_path)

grid_file_path = os.path.join(args.save_path, "grid_full.txt")
GRID_DF = pd.read_csv(grid_file_path, sep='\s+', header=None, names=['index', 'dro', 'r0'])

compiled_dir = os.path.join(args.save_path, "compiled_GPs")
ibme_out_dir = os.path.join(args.save_path, "iBME_results")

os.makedirs(ibme_out_dir, exist_ok=True)

sample_saxs = os.path.join(compiled_dir, "GP0_all_saxs.txt")
sample_df = pd.read_csv(sample_saxs, sep='\s+', header=None)
sim_length = len(sample_df.columns) - 1

exp_pd = pd.read_csv(exp_path, header=None, sep='\s+')
exp_trun = exp_pd.iloc[:sim_length]
exp_trun.to_csv(trun_path, header=False, index=False, sep=' ')

with open(trun_path, "r+") as f:
    content = f.read()
    f.seek(0, 0)
    f.write("# DATA=SAXS BOUNDS=UPPER\n" + content)

results = []
print(f"Starting iBME optimizations for {len(GRID_DF)} grid points...")

for i in range(len(GRID_DF)):
    dro = GRID_DF.iloc[i]['dro']
    r0 = GRID_DF.iloc[i]['r0']
    
    gp_in = os.path.join(compiled_dir, f"GP{i}_all_saxs.txt")
    gp_out_dir = os.path.join(ibme_out_dir, f"GP{i}")
    os.makedirs(gp_out_dir, exist_ok=True)
    
    # Add # DATA=SAXS header to the compiled file
    calc_path = os.path.join(gp_out_dir, "calc_rows.txt")
    with open(calc_path, 'w') as f:
        f.write("# DATA=SAXS\n")
    df = pd.read_csv(gp_in, sep='\s+', header=None)
    df.to_csv(calc_path, mode='a', header=False, index=False, sep=' ')
    
    chi2b = chi2a = phi = np.nan
    try:
        # Run iBME
        iBME_script.iBMEf(trun_path, calc_path, args.theta, f"{gp_out_dir}/")
        
        # Parse Logs
        logs = glob.glob(os.path.join(gp_out_dir, "_ibme_*.log"))
        logs_sorted = sorted(logs, key=lambda x: int(re.search(r"_ibme_(\d+)\.log", x).group(1)))
        log_file = logs_sorted[-1] if logs_sorted else None

        if log_file:
            with open(log_file) as lf:
                for L in lf:
                    if "CHI2 before optimization:" in L: chi2b = float(L.split()[-1])
                    elif "CHI2 after optimization:" in L: chi2a = float(L.split()[-1])
                    elif "Fraction of effective frames:" in L: phi = float(L.split()[-1])
        print(f"GP{i} optimized.")
    except Exception as e:
        print(f"iBME failed for GP{i}: {e}")

    rows = [[i, dro, r0, chi2b, chi2a, phi]]
    grid = np.array(rows, dtype=float)
    np.savetxt(os.path.join(gp_out_dir, f"GRID_opt_{i}"), grid, header="idx d_rho r0 CHI2_before CHI2_after PHI_eff", fmt="%.6g")
    
    frames = pd.DataFrame(rows, columns=["idx", "d_rho", "r0", "CHI2_before", "CHI2_after", "PHI_eff"])
    results.append(frames)

# Compile Master Summary
points = pd.concat(results, ignore_index=True)
grid_sum_path = os.path.join(ibme_out_dir, "GRID_sum.txt")
points.to_csv(grid_sum_path, index=False)

#########----- PLOT

print("Generating Heatmaps...")
grid = np.loadtxt(grid_sum_path, skiprows=1, delimiter=',') # Skip pandas header

dro_vals = np.unique(grid[:,1])
r0_vals  = np.unique(grid[:,2])
order = np.lexsort((grid[:,1], grid[:,2]))
grid = grid[order]

chi2 = np.clip(grid[:,4], 1e-12, None)
phi  = np.clip(grid[:,5], 1e-12, None)
gamma = np.log(chi2 / phi)

chi2_mat = np.log(chi2).reshape(len(r0_vals), len(dro_vals))
phi_mat  = phi.reshape(len(r0_vals), len(dro_vals))
gam_mat  = gamma.reshape(len(r0_vals), len(dro_vals))

min_y, min_x = np.unravel_index(np.nanargmin(gam_mat), gam_mat.shape)
best_dro = dro_vals[min_x]
best_r0  = r0_vals[min_y]

fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

im0 = axs[0].imshow(chi2_mat, origin='upper', aspect='auto')
axs[0].set_title(r'$\ln(\chi^2_{\mathrm{after}})$')
axs[0].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(phi_mat, origin='upper', aspect='auto')
axs[1].set_title(r'$\phi_{\mathrm{eff}}$')
axs[1].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(gam_mat, origin='upper', aspect='auto')
axs[2].set_title(r'$\gamma=\ln(\chi^2_{\mathrm{after}}/\phi_{\mathrm{eff}})$')
axs[2].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

xticks = np.arange(0, len(dro_vals), 2)
yticks = np.arange(0, len(r0_vals), 2)
for ax in axs:
    ax.set_xticks(xticks); ax.set_xticklabels([f'{dro_vals[i]:.2f}' for i in xticks], rotation=300)
    ax.set_yticks(yticks); ax.set_yticklabels([f'{r0_vals[i]:.3f}' for i in yticks])
    ax.set_xlabel(r'$\delta\rho$  [$e/\mathrm{nm}^3$]')
axs[0].set_ylabel(r'$r_0/r_m$')

fig.suptitle(f'Best: δρ={best_dro:.2f}, r0={best_r0:.3f}', y=1.0)
plt.tight_layout()

heatmap_path = os.path.join(ibme_out_dir, f'grid_heatmaps_{today}.png')
fig.savefig(heatmap_path, dpi=300)
print(f"Heatmap saved to {heatmap_path}")

#### SAVE

weight_idx = GRID_DF.index[(GRID_DF['dro'] == best_dro) & (GRID_DF['r0'] == best_r0)].tolist()[0]
best_gp_dir = os.path.join(ibme_out_dir, f"GP{weight_idx}")

# Dynamically find the last .weights.dat file (in case it isn't always _19)
weight_files = glob.glob(os.path.join(best_gp_dir, "*.weights.dat"))
weight_files_sorted = sorted(weight_files, key=lambda x: int(re.search(r"_(\d+)\.weights\.dat", os.path.basename(x)).group(1)))
best_weight_file = weight_files_sorted[-1]

manifest_path = os.path.join(compiled_dir, f"GP{weight_idx}_manifest.txt")
contents = pd.read_csv(manifest_path, header=None)

opt_weight = pd.read_csv(best_weight_file, sep='\s+', header=None)

if len(opt_weight) != len(contents):
    raise ValueError(
        f"GP{weight_idx} has {len(contents)} PDB names but {len(opt_weight)} optimized weights."
    )

expected_indices = set(range(len(contents)))
observed_indices = set(opt_weight.iloc[:, 0].astype(int))
if observed_indices != expected_indices:
    raise ValueError(f"Weight indices for GP{weight_idx} are not exactly 0 through {len(contents) - 1}.")

opt_weight['PDB_Name'] = opt_weight.iloc[:, 0].map(contents.iloc[:, 0])
opt_sorted = opt_weight.sort_values(by=1, ascending=False)

weights_out = os.path.join(ibme_out_dir, f'structure_weights_sorted_{today}.txt')
opt_sorted.to_csv(weights_out, index=None, sep='\t')
print(f"Top structure weights saved to {weights_out}")
