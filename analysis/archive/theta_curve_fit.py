import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
import glob

#####----- INFORMATION

#This script will compare simulated saxs curves from a defined collection of theta values against the experimental curve
#A separate directory that contains the calculated SAXS data for each grid point is required. Each theta should be in separate folders
    #The folders should all have the same name besides the end which will specify its theta value. EXAMPLE: "..._theta2000/"
        #The specific folder names can be modified in the code below
    #Within each folder, a .txt file should be created that contains the best fit grid point for that simulation. This is so theta_curve_fit.py uses
        #the correct saxs data. EXAMPLE: 22
    #In addition, each folder should have the optimized weights for each structure. If these weights are best fit, then they will correlate with the
        #correct grid point data
#The SAXS curves must already be compiled

#####----- PATHS AND ARGUMENTS
parser = argparse.ArgumentParser(description="SAXS curve analysis for different theta scans")
parser.add_argument("main_path", type=str, help="path to main directory containing specific theta scans")
parser.add_argument("saxs_path", type=str, help="path to SAXS simulations")

args = parser.parse_args()

##EDIT
theta_vals = np.array([])
theta_path = os.path.join(args.main_path, "saxs_dro_30_to_50_r0_1.22_to_1.42_theta{}")
best_gp = 10
##

path_exp_file = "/path/to/experimental/data"
structure_path = "/path/to/structure/files"

#####----- FUNCTIONS

def experimental_curve(path_exp_file, sim_length):
    exp_pd = pd.DataFrame(pd.read_csv("{}/SASDLU4.dat".format(path_exp_file),
                                      header=None, sep=r"\s+"))

    s = exp_pd.loc[:,0]
    iq = exp_pd.loc[:,1]
    err = exp_pd.loc[:,2]

    s_trun = s.iloc[:sim_length].values
    iq_trun = iq.iloc[:sim_length].values
    err_trun = err.iloc[:sim_length].values

    plt.errorbar(s, iq, yerr=err, fmt='o', ecolor="lightgray", markersize=3)
    plt.yscale("log")
    plt.ylabel("i(q)")
    plt.xlabel("s")
    plt.title("Experimental SAXS curve with x log")
    save_path_1 = "{}/experiment.png".format(args.save_path)
    plt.savefig(save_path_1, dpi=300)

    return s_trun, iq_trun, err_trun, s, iq, err

def match_files(real_theta):
    sim_path = "{}/structure_weights_sorted_*.txt".format(real_theta)

    matching_files = glob.glob(sim_path)
    real_file = matching_files[0]

    #sim_pd = pd.read_csv(real_file, sep='\\t', header=0)
    #sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb', '', regex=False)

    #grid_path = os.path.join(real_theta, "best_grid.txt")
    #with open(grid_path, 'r') as f:
        #gp = f.read().strip()

    compiled_saxs = np.genfromtxt("{}/compiled_GPs/GP{}_all_saxs.txt".format(args.saxs_path, str(best_gp)))
    compiled_df = pd.DataFrame(compiled_saxs).reset_index(drop=True)
    iq_sim_matrix = pd.DataFrame(compiled_df.drop(columns=[0]).values)

    true_pdb_order = []

    search_pattern = os.path.join(args.saxs_path, "mm*", f"GP{best_gp}", "calc_saxs.txt")
    sorted_files = natsorted(glob.glob(search_pattern))

    for file in sorted_files:
        parts = file.split(os.sep)
        frac_folder = [p for p in parts if p.startswith('mm')][0]

        struct_frac_path = os.path.join(structure_path, frac_folder)
        pdbs = glob.glob(os.path.join(struct_frac_path, "mm*.pdb"))

        pdbs.sort()

        for pdb in pdbs:
            true_pdb_order.append(os.path.basename(pdb).replace('.pdb', ''))

    pdb_names = pd.Series(true_pdb_order)

    s_full = np.genfromtxt("{}/mm016_100/qvals.txt".format(args.saxs_path), skip_header=0)
    s_values = s_full[:iq_sim_matrix.shape[1]]

    return s_values, iq_sim_matrix, None, pdb_names, real_file

def VACC_average_curve(real_file, pdb_names, s_val, iq_val):
    sim_pd = pd.read_csv(real_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb','',regex=False)

    weight_map = dict(zip(sim_pd['PDB_Name'], sim_pd['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])
    iq_array = np.array(iq_val)

    prior_iq = np.mean(iq_array, axis=1)

    weighted_matrix = iq_array * ordered_weights[:, np.newaxis]
    avg_iq = np.sum(weighted_matrix, axis=0)

    sim_merge = pd.concat([pd.DataFrame(s_val), pd.DataFrame(avg_iq), pd.DataFrame(prior_iq)], axis=1)

    return len(sim_merge), sim_merge.iloc[:, 0], sim_merge.iloc[:, 1], sim_merge.iloc[:, 2]

def plot_compare(s_weighted_dict, iq_weighted_dict, iq_prior, s, iq, err, s_full, iq_full, err_full):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.errorbar(s, iq, yerr=err, fmt='o', markersize=3, ecolor="lightgray", label="Experiment")
    ax.set_yscale("log")

    for theta in theta_vals:
        a = 0.9
        scale = np.sum(iq * iq_weighted_dict[theta]) / np.sum(iq_weighted_dict[theta] ** 2)
        scaled_iq_sim = iq_weighted_dict[theta] * scale
        ax.plot(s_weighted_dict[theta], scaled_iq_sim, lw=3, label=str(theta), color="red", alpha=a)
        a -= .1
    ax.set_ylabel("i(q)")
    ax.set_xlabel("s")
    ax.set_title("Simulated SAXS fit with Experiment - truncated")
    ax.legend()

    save_path_2 = "{}/theta_scan_curves.png".format(args.main_path)
    fig.savefig(save_path_2, dpi=300)

def main():
    s_weighted_dict = {}
    iq_weighted_dict = {}
    for theta in theta_vals:
        real_theta = theta_path.format(theta)
        s, concat_merge, weights, f_name, w_file = match_files(real_theta)
        lent, s_weighted, iq_weighted, iq_prior = VACC_average_curve(w_file, f_name, s, concat_merge)

        s_weighted_dict[theta] = s_weighted
        iq_weighted_dict[theta] = iq_weighted

    angletrun, intensetrun, errtrun, anglefull, intensefull, errfull = experimental_curve(path_exp_file, lent)

    plot_compare(s_weighted_dict, iq_weighted_dict, iq_prior, angletrun, intensetrun, errtrun, anglefull, intensefull, errfull)

if __name__ == "__main__":
    main()