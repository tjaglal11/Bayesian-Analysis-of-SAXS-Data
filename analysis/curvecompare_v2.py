import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import glob
from natsort import natsorted
from freesas.autorg import auto_guinier
import matplotlib.lines as mlines
import mdtraj as md

#####----- ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("pdb_path", type=str, default="", help="Path to pdb files")
parser.add_argument("main_path", type=str, default="", help="Path to the main directory")
parser.add_argument("ibme_path", type=str, default="", help="Path to the IBME results")
parser.add_argument("out_path", type=str, default="", help="Path to the output directory")
parser.add_argument("exp_path", type=str, default="", help="Path to the experimental data")
args = parser.parse_args()

#####----- FUNCTIONS
def pdb_manifest(pdb_path):
    search_pattern = os.path.join(pdb_path, "*.pdb")
    found_files = glob.glob(search_pattern)

    print(f"Found {len(found_files)} PDB files.")

    sorted = natsorted(found_files)
    manifest = np.array(sorted)

    return manifest

def find_data(main_path, ibme_path, manifest):
    #Find the best gp
    gridsum_df = np.genfromtxt(f"{ibme_path}/GRID_sum.txt", skip_header=1, delimiter=',', filling_values=np.nan)
    chi2 = np.clip(gridsum_df[:,4], 1e-12, None)
    phi  = np.clip(gridsum_df[:,5], 1e-12, None)
    gamma = np.log(chi2 / phi )

    best_idx = np.nanargmin(gamma)
    best_dro = gridsum_df[best_idx, 1]
    best_r0 = gridsum_df[best_idx, 2]

    print(f"Recovered Best Parameters at GP{best_idx} -> delta rho={best_dro:.2f}, r0={best_r0:.3f}")

    #Find the GP files in compiled_GPs
    weight_idx = int(gridsum_df[best_idx, 0])
    saxs_data = pd.read_csv(f"{main_path}/compiled_GPs/GP{weight_idx}_all_saxs.txt", sep=r'\s+', header=None, index_col=0)
    q_vals = pd.read_csv(f"{main_path}/amm16_100/qvals.txt", sep=',', header=None) #not sure this is the best way to grab

    print(f"Recovered file {main_path}/compiled_GPs/GP{weight_idx}_all_saxs.txt containing {len(saxs_data)} Saxs Data Points.")

    #Find weights file and sort
    w_path_pattern = os.path.join(f"{ibme_path}/structure_weights_*.txt")
    matching_files = glob.glob(w_path_pattern)
    w_path = matching_files[0]

    weights_file = pd.read_csv(w_path, sep=r'\s+', header=[0])
    weights_file["PDB_Name"] = weights_file["PDB_Name"].str.replace(".pdb", "", regex=False)
    weight_map = dict(zip(weights_file["PDB_Name"], weights_file["1"]))

    pdb_names = [os.path.splitext(os.path.basename(p))[0] for p in manifest]
    missing = [n for n in pdb_names if n not in weight_map]
    if missing:
        raise ValueError(f"Missing weights for the following PDBs: {missing}")
    w = np.array([weight_map[n] for n in pdb_names], dtype=float)

    print(f"Recovered weights file {ibme_path}/structure_weights_*.txt containing {len(weights_file)} weights and sorted by PDB name.")

    print("rows in weights file:", len(weights_file), "| manifest:", len(pdb_names))
    print("NaN weights:", np.isnan(w).sum(), "| sum:", w.sum())
    print(weights_file[weights_file["1"].isna()].head())

    return saxs_data, q_vals, w

def create_profiles(saxs_data, q_vals, w):
    saxs_num = saxs_data.select_dtypes(include="number")
    saxs_prior = saxs_data.mean(axis=0).to_numpy()

    if q_vals.empty:
        q = np.asarray(q_vals.columns, dtype=float)
    else:
        q = q_vals.to_numpy(dtype=float).ravel()

    if len(saxs_num) != len(w):
        raise ValueError(f"{len(saxs_num)} SAXS rows but {len(w)} weights")
    saxs_weights = saxs_data.multiply(w, axis=0).sum(axis=0) / w.sum()

    return q, saxs_prior, saxs_weights

def experiment_analysis(exp_path):
    #Find the saxs data for the experiment
    exp_data = pd.read_csv(exp_path, sep=r'\s+', header=None)

    e_s = exp_data.iloc[:,0].to_numpy(dtype=float)
    e_iq = exp_data.iloc[:,1].to_numpy(dtype=float)
    e_err = exp_data.iloc[:,2].to_numpy(dtype=float)

    data = np.column_stack((e_s, e_iq, e_err))

    #Determine the experimental radius of gyration
    e_rg = auto_guinier(data).Rg

    return e_s, e_iq, e_err, e_rg

def pdb_to_rg(manifest, w):

    rg_list = []
    for i in range(manifest.shape[0]):
        #print(f"Processing {manifest[i]}")

        traj = md.load(manifest[i])
        rg = md.compute_rg(traj)

        rg_list.append(rg)

    rg_vals = np.array([rg[0] for rg in rg_list])
    prior_rg = np.mean(rg_vals)
    post_rg = np.sum(rg_vals * w) / w.sum()

    return prior_rg, post_rg

def fit_scale(q, I_model, e_s, e_iq, e_err):
    I_interp = np.interp(e_s, q, I_model)
    wts = 1.0 / e_err**2
    c = np.sum(wts * e_iq * I_interp) / np.sum(wts * I_interp**2)

    return c * I_model

def plot_saxs(q, saxs_prior, saxs_weights, e_s, e_iq, e_err, e_rg, prior_rg, post_rg, out_path):
    fix, ax = plt.subplots(figsize=(10,10))

    ax.errorbar(e_s, e_iq, yerr=e_err, fmt='o', markersize=3, ecolor="lightgray", label="Experiment", zorder=1)
    ax.set_yscale("log")

    ax.plot(q, fit_scale(q, saxs_prior, e_s, e_iq, e_err), label="Prior", color='lightcoral')
    ax.plot(q, fit_scale(q, saxs_weights, e_s, e_iq, e_err), label="Posterior", color='red')
    ax.set_xlabel("Q")
    ax.set_ylabel("I(q)")

    leg_post = ax.legend(loc="upper right")
    ax.add_artist(leg_post)

    rg_handles_post = [mlines.Line2D([], [], color='none', label=f"Exp Rg: {e_rg:.2f} nm")]

    prior_label = fr"Prior rg: {prior_rg:.2f} nm"
    post_label = fr"Posterior rg: {post_rg:.2f} nm"
    rg_handles_post.append(mlines.Line2D([], [], color='none', label=prior_label))
    rg_handles_post.append(mlines.Line2D([], [], color='none', label=post_label))

    ax.legend(handles=rg_handles_post, loc="lower left", title="Radius of Gyration ($R_g$)", handlelength=0, handletextpad=0)

    plt.savefig(f"{out_path}/weighted_saxs_profile.png", dpi=300, bbox_inches='tight')

#####----- MAIN
def main():
    print("Starting...")
    print(f"Generated pdb manifest...")
    manifest = pdb_manifest(args.pdb_path)
    print(f"Finding SAXS data...")
    saxs_data, q_vals, w = find_data(args.main_path, args.ibme_path, manifest)
    print(f"Creating profiles with weights...")
    q, saxs_prior, saxs_weights = create_profiles(saxs_data, q_vals, w)
    print(f"Experimental analysis...")
    e_s, e_iq, e_err, e_rg = experiment_analysis(args.exp_path)
    print(f"Radius of gyration analysis...")
    prior_rg, post_rg = pdb_to_rg(manifest, w)
    print("Plotting...")
    plot_saxs(q, saxs_prior, saxs_weights, e_s, e_iq, e_err, e_rg, prior_rg, post_rg, args.out_path)
    print("Done!")

if __name__ == "__main__":
    main()
