import pandas as pd
import os
import numpy as np
import mdtraj as md
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob


#####----- ARGUEMNTS
parser = argparse.ArgumentParser()
parser.add_argument("pdb_path", type=str, default="", help="Path to pdb files")
parser.add_argument("out_path", type=str, default="", help="Path to the output directory")
parser.add_argument("weights_path", type=str, default="", help="Path to the weights txt file")
args = parser.parse_args()

#####-----FUNCTIONS
def pdb_manifest(pdbs, pdb):
    search_pattern = os.path.join(pdbs, "*.pdb")
    found_files = glob.glob(search_pattern)

    print(f"Found {len(found_files)} PDB files.")

    sorted = natsorted(found_files)
    manifest = np.array(sorted)

    return manifest

def pdb_to_rg(manifest):

    rg_list = []
    for i in range(manifest.shape[0]):
        print(f"Processing {manifest[i]}")

        traj = md.load(manifest[i])
        rg = md.compute_rg(traj)

        rg_list.append(rg)

    return rg_list

def weighted_rg(traj_list, manifest, outpath, weights_path):
    colors = sns.color_palette(palette='Set1')
    
    weights_df = pd.read_csv(weights_path, sep='\s+', header=[0])
    weights_df["PDB_Name"] = weights_df["PDB_Name"].str.replace(".pdb", "" , regex=False)
    weight_map = dict(zip(weights_df["PDB_Name"], weights_df["1"]))

    pdb_names = [os.path.splitext(os.path.basename(p))[0] for p in manifest]
    missing = [n for n in pdb_names if n not in weight_map]
    if missing:
        raise ValueError(f"Missing weights for the following PDBs: {missing}")
    w = np.array([weight_map[n] for n in pdb_names], dtype=float)

    traj_df = pd.DataFrame([item[0] for item in traj_list])

    prior_weights = np.ones(len(traj_df)) / len(traj_df)
    
    plt.figure(figsize=(10,10))
    sns.set_style("ticks")

    sns.kdeplot(x=traj_df, weights=prior_weights, color='red', label='Prior ensemble')
    sns.kdeplot(x=traj_df, weights=w, color='blue', label='Posterior ensemble')
    plt.axvline(x=experimental_rg, color='green', linestyle='--', label='Experimental Rg')

    plt.xlabel('Rg in Angstrom')
    plt.ylabel('Density')
    plt.title('Rg distribution in prior and posterior ensembles')
    plt.legend()

    plt.savefig("{}/MDTraj_rg_plot.png".format(outpath), dpi=300, bbox_inches='tight')

def main():
    manifest = pdb_manifest(args.pdb_path, "pdb")
    traj_list = pdb_to_rg(manifest)
    weighted_rg(traj_list, manifest, args.out_path, args.weights_path)
    print("Plotting complete")

if __name__ == "__main__":
    main()
