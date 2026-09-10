#####----- Analyze a collection of PDB files to determine per residue secondary structure via MDTraj and plot propensity

import os
import pandas as pd
import numpy as np
import argparse
import glob
from natsort import natsorted
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

        res_means.plot(kind="area", ax=ax, title=f"Propensity of {struc} secondary structure", color=colors[i])
        plt.xlabel("Residue number")
        plt.ylabel("Propensity")
        plt.savefig(
            os.path.join(outpath, f"propensity_of_{struc}_secondary_structure.png")
        )
        plt.close()

    return unique_struc

def main():
    manifest = pdb_manifest(pdb_path, "pdb")
    traj_list = pdb_to_traj(manifest)
    plot_traj(traj_list, save_path)
    print("Plotting complete.")
if __name__ == "__main__":
    main()


