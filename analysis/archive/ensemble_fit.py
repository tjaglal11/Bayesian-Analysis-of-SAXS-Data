import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsort_keygen, natsorted
import argparse
import os
import glob
import yaml
from scipy.stats import linregress
import matplotlib.lines as mlines

#####----- PLOT SAXS CURVE COMPARISON OF EXPERIMENT VS WEIGHTED AVERAGE SIMULATION

#####----- FUNCTIONS

def rg_guinier(s, iq, fit_points):
    valid_idx = iq > 0
    s_valid = s[valid_idx]
    iq_valid = iq[valid_idx]


    s_low = s_valid[:fit_points]
    iq_low = iq_valid[:fit_points]

    x = s_low**2
    y = np.log(iq_low)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    if slope >= 0:
        print("Positive slope")
        return np.nan, r_value**2
    rg = np.sqrt(-3 * slope)

    return rg, r_value**2

def grab_rg(sim_file, save_path, dro, r0, pdb_names):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace(".pdb", "", regex=False)
    weight_map = dict(zip(sim_pd['PDB_Name'], sim_pd['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])

    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None,
                          names=['index', 'dro', 'r0'])
    gp = grid_df.loc[(grid_df['dro'] == dro) & (grid_df['r0'] == r0), 'index'].iloc[0]

    search_pattern = os.path.join(save_path, "MD*_*", f"GP{gp}", "Rg_env.dat")
    sorted_files = natsorted(glob.glob(search_pattern))

    rg_list = []
    for file in sorted_files:
        data = pd.read_csv(file, sep='\s+', header=None, names=['Rg'])
        rg_list.extend(data['Rg'].tolist())

    rg_array = np.array(rg_list)

    prior_rg_real = np.mean(rg_array)
    post_rg_real = np.sum(rg_array * ordered_weights)

    return post_rg_real, prior_rg_real

def match_files(sim_file, dro, r0, save_path, structure_path):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb', '', regex=False)

    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None,
                          names=['index', 'dro', 'r0'])
    gp = grid_df.loc[(grid_df['dro'] == dro) & (grid_df['r0'] == r0), 'index'].iloc[0]

    compiled_saxs = np.genfromtxt("{}/compiled_GPs/GP{}_all_saxs.txt".format(save_path, str(gp)))
    compiled_df = pd.DataFrame(compiled_saxs).reset_index(drop=True)
    iq_sim_matrix = pd.DataFrame(compiled_df.drop(columns=[0]).values)

    true_pdb_order = []

    #search_pattern = os.path.join(save_path, "mm*", f"GP{gp}", "calc_saxs.txt")
    search_pattern = os.path.join(save_path, "MD*_*", f"GP{gp}", "calc_saxs.txt")
    sorted_files = natsorted(glob.glob(search_pattern))

    for file in sorted_files:
        parts = file.split(os.sep)
        #frac_folder = [p for p in parts if p.startswith('mm') and '_' in p][0]
        frac_folder = [p for p in parts if p.startswith('MD') and '_' in p][0]

        parent_folder = frac_folder.split('_')[0]

        struct_frac_path = os.path.join(structure_path, parent_folder, frac_folder)
        #pdbs = glob.glob(os.path.join(struct_frac_path, "mm*.pdb"))
        pdbs = glob.glob(os.path.join(struct_frac_path, "MD*_center*.pdb"))

        pdbs.sort()

        for pdb in pdbs:
            true_pdb_order.append(os.path.basename(pdb).replace('.pdb', ''))

    pdb_names = pd.Series(true_pdb_order)

    if not sorted_files:
        raise FileNotFoundError(f"Could not find calc_saxs.txt in {save_path}/MD*_*/GP{gp}.")

    e_file = "/users/t/j/tjaglal/experimental_data/SASDLS4.dat"
    s_full = np.genfromtxt(e_file, comments='#', usecols=0)
    s_values = s_full[:iq_sim_matrix.shape[1]]

    #s_full = np.genfromtxt("{}/mm016_100/qvals.txt".format(save_path), skip_header=0)
    #s_values = s_full[:iq_sim_matrix.shape[1]]

    return s_values, iq_sim_matrix, None, pdb_names

def VACC_average_curve(sim_file, pdb_names, s_val, iq_val):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb','',regex=False)

    weight_map = dict(zip(sim_pd['PDB_Name'], sim_pd['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])
    iq_array = np.array(iq_val)

    prior_iq = np.mean(iq_array, axis=0)

    weighted_matrix = iq_array * ordered_weights[:, np.newaxis]
    avg_iq = np.sum(weighted_matrix, axis=0)

    sim_merge = pd.concat([pd.DataFrame(s_val), pd.DataFrame(avg_iq), pd.DataFrame(prior_iq)], axis=1)
    print("Breakpt")

    return len(sim_merge), sim_merge.iloc[:,0], sim_merge.iloc[:,1], sim_merge.iloc[:,2]
def simulated_curves(sim_file, structure_path):
    sim_pd = pd.read_csv(sim_file, header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb','',regex=False)

    all_weights = sim_pd[1]
    curves = len(sim_pd)
    all_sim_s = []
    all_sim_iq = []
    filenames = []

    for i in range(curves):
        if "mm" in sim_pd.iloc[i,2]:
            file = sim_pd.iloc[i,2]
        else:
            continue
        read_file = pd.read_csv("{}/{}.out".format(structure_path, file), sep=r"\s+", skiprows=5)
        s_sim = read_file.iloc[:,0] * 10
        iq_sim = read_file.iloc[:,1]

        all_sim_s.append(s_sim)
        all_sim_iq.append(iq_sim)
        filenames.append(file)

    df_s = pd.DataFrame(all_sim_s, index=filenames)
    df_iq = pd.DataFrame(all_sim_iq, index=filenames)

    fixed_weights = all_weights.drop(all_weights.index[0]).reset_index(drop=True)
    df_weights = pd.DataFrame({"weights": fixed_weights.values, "structure file": filenames})

    length = s_sim.iloc[-1]

    return length, df_s, df_iq, file, df_weights, all_weights

def average_curve(s_array, iq_array, weights):
    s_values = s_array.iloc[0, :]
    iq_tran = iq_array.transpose()
    merge = pd.concat([s_values, iq_tran], axis=1)

    merge.columns.values[0] = "s"
    weighted_merge = merge.copy()

    for i in range(len(merge.columns)):
        if "mm" in merge.columns[i]:
            structure = merge.columns[i]
        else:
            continue

        match = weights.loc[weights["structure file"] == structure, "weights"]
        if match.empty:
            w = 0
        else:
            w = match.iloc[0]
            if pd.isna(w):
                w = 0

        for x in range(len(merge)):
            iq_i = merge.loc[x, structure]
            weighted_merge.loc[x, structure] = iq_i * w

    averaged_merge = pd.DataFrame({"s": s_values, "iq": np.nan})
    for i in range(len(merge)):
        series = weighted_merge.iloc[i, 1:]
        series_np = np.array(series)
        average = np.sum(series_np)
        averaged_merge.loc[i, "iq"] = average

    return averaged_merge.loc[:,"s"], averaged_merge.loc[:,"iq"]

def experimental_curve(path_exp_file, sim_length, simu_path):
    exp_pd = pd.DataFrame(pd.read_csv(path_exp_file,
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
    save_path_1 = "{}/ref_experiment.png".format(simu_path)
    plt.savefig(save_path_1, dpi=300)

    return s_trun, iq_trun, err_trun, s, iq, err
    print("Breakpt")

def plot_compare(s_sim, iq_sim, iq_prior, s, iq, err, s_full, iq_full, err_full, f_name, simu_path, post_rg):
    scale = np.sum(iq * iq_sim) / np.sum(iq_sim**2)

    scaled_iq_sim = iq_sim * scale

    scaled_prior = np.sum(iq * iq_prior) / np.sum(iq_prior**2)
    scaled_iq_prior = iq_prior * scaled_prior

    fig, ax = plt.subplots(figsize = (10,10))
    ax.errorbar(s, iq, yerr = err, fmt= 'o', zorder=1, markersize=3, ecolor="lightgray", label="Experiment")
    ax.set_yscale("log")

    ax.plot(s_sim, scaled_iq_sim, zorder=2, lw=3, label="Posterior", color="orange")
    ax.plot(s_sim, scaled_iq_prior, zorder=3, lw=3, label="Prior", color="green")
    ax.set_ylabel("i(q)")
    ax.set_xlabel("s")
    ax.set_title("Simulated SAXS fit with Experiment - truncated")

    leg_post = ax.legend(loc="upper right")
    ax.add_artist(leg_post)

    rg_handles_post = [mlines.Line2D([], [], color="none", label=f"Post rg: {post_rg:.2f} nm")]
    ax.legend(handles=rg_handles_post, loc="lower left", title="Radius of Gyration ($R_g$)", handlelength=0, handletextpad=0)
    
    save_path_2 = "{}/truncated_fit.png".format(simu_path)
    fig.savefig(save_path_2, dpi=300)

    fig_2, ax_2 = plt.subplots(figsize = (10,10))
    ax_2.errorbar(s_full, iq_full, yerr = err_full, fmt= 'o', zorder=1, markersize=3, ecolor="lightgray", label="Experiment")
    ax_2.set_yscale("log")

    ax_2.plot(s_sim, scaled_iq_sim, zorder=2, lw=3, label="Posterior", color="orange")
    ax_2.plot(s_sim, scaled_iq_prior, zorder=3, lw=3, label="Prior", color="green")
    ax_2.set_ylabel("i(q)")
    ax_2.set_xlabel("s")
    ax_2.set_title("Simulated SAXS fit with Experiment - full")
    ax.legend()
    
    save_path_3 = "{}/full_fit.png".format(simu_path)
    fig_2.savefig(save_path_3, dpi=300)

##########------ MAIN

def main():
    #####----- Initialize CLI arguments and configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to main yaml file")
    args = parser.parse_parser_args() if hasattr(parser, 'parse_parser_args') else parser.parse_args()

    with open(args.config, "r") as f:
        master_config = yaml.safe_load(f)

    ef_config = master_config.get("ensemble_fit", {})
    if not ef_config:
        print("ensemble_fit config not found")
        return

    dro = ef_config.get["dro", ""]
    r0 = ef_config.get["r0", ""]
    path_exp_file = ef_config.get["path_exp_file", ""]
    structure_path = ef_config.get["structure_path", ""]
    save_path = ef_config.get["save_path", ""]
    simu_path = ef_config.get["sim_path", ""]

    sim_path = "{}/structure_weights_sorted_*.txt".format(simu_path)

    matching_files = glob.glob(sim_path)
    real_file = matching_files[0]

    #Match Files
    s, concat_merge, weights, f_name = match_files(real_file, dro, r0, save_path, structure_path)
    lent, s_weighted, iq_weighted, iq_prior = VACC_average_curve(real_file, f_name, s, concat_merge)

    #Get the RG value
    post_rg, prior_rg = grab_rg(real_file, save_path, dro, r0, f_name)

    # Create the SAXS curves from experimental data for plotting
    angletrun, intensetrun, errtrun, anglefull, intensefull, errfull = experimental_curve(path_exp_file, lent, simu_path)

    # Plot experiment vs weighted average simulation
    plot_compare(s_weighted, iq_weighted, iq_prior, angletrun, intensetrun, errtrun, anglefull, intensefull, errfull, "GP0_all_saxs", simu_path, post_rg)

if __name__ == '__main__':
    main()

print("Breakpt")
