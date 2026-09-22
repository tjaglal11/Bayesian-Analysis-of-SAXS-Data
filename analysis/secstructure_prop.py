#####----- Analyze a collection of PDB files to determine per residue secondary structure via MDTraj and plot propensity

import os
import pandas as pd
import numpy as np
import argparse
import glob
from natsort import natsorted, index_natsorted
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns

#####----- ARGUMENTS
#parser = argparse.ArgumentParser()
#parser.add_argument("pdb_path", type=str, default="", help="Path to PDB files")
#parser.add_argument("save_path", type=str, default="", help="Path to save output")
#args = parser.parse_args()

pdb_path = "/Users/timothyjaglal/Desktop/a-syn"
save_path = "/Users/timothyjaglal/Desktop/a-syn"

#####----- FUNCTIONS

def pdb_manifest(pdbs, pdb):
    search_pattern = os.path.join(pdbs, "*.pdb")
    found_files = glob.glob(search_pattern)

    print(f"Found {len(found_files)} PDB files.")

    sorted = natsorted(found_files)
    manifest = np.array(sorted)

    return manifest

def pdb_to_traj(manifest):

    traj_list = []

    for i in range(manifest.shape[0]):
        print(f"Processing {manifest[i]}")

        traj = md.load(manifest[i])
        sstruc = md.compute_dssp(traj)

        traj_list.append(sstruc)

    return traj_list

def plot_traj(traj_list, outpath):
    colors = sns.color_palette(palette='Set1')
    unique_struc = np.unique(traj_list)

    for i, struc in enumerate(unique_struc):
        print(f"Processing trajectories with secondary structure {struc}")
        traj_2d = [item[0] for item in traj_list]
        traj_df = pd.DataFrame(traj_2d)

        traj_df_nums = traj_df.copy()
        traj_df_nums = pd.DataFrame(np.where(traj_df_nums == struc, 1, 0))

        res_means = traj_df_nums.mean()

        fig, ax = plt.subplots(figsize=(12, 4))

        res_means.plot(kind="line", ax=ax, title=f"Propensity of {struc} secondary structure", color=colors[i])
        plt.xlabel("Residue number")
        plt.ylabel("Propensity")
        plt.savefig(
            os.path.join(outpath, f"propensity_of_{struc}_secondary_structure.png")
        )
        plt.close()

    return unique_struc

def weighted_traj(traj_list, manifest, outpath, weights_path):
    colors = sns.color_palette(palette='Set1')
    unique_struc = np.unique(traj_list)

    weights_df = pd.read_csv(weights_path, sep='\s+', header=[0])
    weights_df["PDB_Name"] = weights_df["PDB_Name"].str.replace(".pdb", "", regex=False)
    weight_map = dict(zip(weights_df["PDB_Name"], weights_df["1"]))

    pdb_names = [os.path.splitext(os.path.basename(p))[0] for p in manifest]
    missing = [n for n in pdb_names if n not in weight_map]
    if missing:
        raise ValueError(f"Missing weights for the following PDBs: {missing}")
    w = np.array([weight_map[n] for n in pdb_names], dtype=float)

    traj_df = pd.DataFrame([item[0] for item in traj_list])

    for i, struc in enumerate(unique_struc):
        print(f"Processing trajectories with secondary structure {struc}")
        indicator = pd.DataFrame(np.where(traj_df == struc, 1, 0))

        prior = indicator.mean()
        post = indicator.multiply(w, axis=0).sum() / w.sum()

        fig, ax = plt.subplots(figsize=(12,4))

        prior.plot(ax=ax, color="0.5", lw=1.2, label="Prior")
        post.plot(ax=ax, color=colors[i], lw=1.5, label="Posterior")
        ax.fill_between(prior.index, prior, post, color=colors[i], alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Propensity")
        ax.set_xlabel("Residue number")
        ax.set_title(f"Propensity of {struc} secondary structure")
        ax.legend()
        plt.savefig(
            os.path.join(outpath, f"weighted_propensity_of_{struc}_secondary_structure.png")
        )
        plt.close()

def least_prop(manifest, traj_list):
    if len(manifest) != len(traj_list):
        raise("ERROR: Manifest and trajectory lists are not the same length.")

    unique_struc = np.unique(traj_list)
    least_files = {}

    for struc in unique_struc:
        propensities = np.array([np.mean(traj[0] == struc) for traj in traj_list])
        least_file = manifest[np.argmin(propensities)]
        least_files[struc] = least_file
        print(f"Least propensity for {struc} secondary structure: {least_file}")

    return least_files

def main():
    manifest = pdb_manifest(pdb_path, "pdb")
    traj_list = pdb_to_traj(manifest)
    plot_traj(traj_list, save_path)
    #Optional
    least_prop(manifest, traj_list)
    print("Plotting complete.")

weights = pd.read_csv("/home/malab/Downloads/asyn_t100_weights.txt", sep='\s+', header=[0])
sorted = weights.loc[natsorted(weights.index, key=weights["PDB_Name"].get)]
#if __name__ == "__main__":
#    main()


