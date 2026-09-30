#####----- Compare SAXS curves between multiple sets of reweighed structures
import argparse
#####----- IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted
import seaborn as sns
from scipy.stats import linregress 
import matplotlib.lines as mlines
import yaml

type = "full"
#####----- FUNCTIONS
def experimental_curve(path_exp_file, sim_length):
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
    #save_path_1 = "{}/experiment.png".format(save_path)
    #plt.savefig(save_path_1, dpi=300)

    return s_trun, iq_trun, err_trun

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

    search_pattern = os.path.join(save_path, "mm*", f"GP{gp}", "Rg_env.dat")
    sorted_files = natsorted(glob.glob(search_pattern))
    rg_list = []

    for file in sorted_files:
        data = pd.read_csv(file, sep='\s+', header=None, names=['Rg'])
        rg_list.extend(data['Rg'].tolist())
    rg_array = np.array(rg_list)

    prior_rg_real = np.mean(rg_array)
    post_rg_real = np.sum(rg_array * ordered_weights)

    return post_rg_real / 10, prior_rg_real / 10


def cterm_grab_rg(sim_file, save_path, dro, r0, pdb_names):
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

    return post_rg_real / 10, prior_rg_real / 10

def cterm_match_files(sim_file, save_path, dro, r0, structure_path, path_exp_file):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb', '', regex=False)

    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None,
                          names=['index', 'dro', 'r0'])
    matching_rows = grid_df.loc[(grid_df['dro'] == dro) & (grid_df['r0'] == r0), 'index']
    if matching_rows.empty:
        print("\n" + "="*50)
        print("ERROR: Could not find matching grid point!")
        print(f"Looking for: dro = {dro} (type: {type(dro)}), r0 = {r0} (type: {type(r0)})")
        print("\nAvailable unique 'dro' values in grid_full.txt:")
        print(grid_df['dro'].unique()[:10]) # Print first 10 to inspect
        print("\nAvailable unique 'r0' values in grid_full.txt:")
        print(grid_df['r0'].unique()[:10])
        print("="*50 + "\n")
        raise ValueError(f"Grid coordinates (dro={dro}, r0={r0}) do not exist in {os.path.join(save_path, 'grid_full.txt')}")

    gp = matching_rows.iloc[0]

    compiled_saxs = np.genfromtxt("{}/compiled_GPs/GP{}_all_saxs.txt".format(save_path, str(gp)))
    compiled_df = pd.DataFrame(compiled_saxs).reset_index(drop=True)
    iq_sim_matrix = pd.DataFrame(compiled_df.drop(columns=[0]).values)

    true_pdb_order = []

    search_pattern = os.path.join(save_path, "MD*_*", f"GP{gp}", "calc_saxs.txt")
    sorted_files = natsorted(glob.glob(search_pattern))

    print("\n" + "-"*50)
    print(f"DEBUGGING DIRECTORY WALK FOR GP {gp}")
    print(f"  Number of 'calc_saxs.txt' files found: {len(sorted_files)}")

    for file in sorted_files:
        parts = file.split(os.sep)
        frac_folder = [p for p in parts if p.startswith('MD') and '_' in p][0]

        parent_folder = frac_folder.split('_')[0]
        struct_frac_path = os.path.join(structure_path, parent_folder, frac_folder)
        if "MD632" in struct_frac_path:
            print(f"DEBUG: Searching for MD632 PDBs in: '{struct_frac_path}'")
            # Print up to 5 actual files in that folder to see what they are named
            if os.path.exists(struct_frac_path):
                print(f"       Actual files inside this folder: {os.listdir(struct_frac_path)[:5]}")
            else:
                print(f"       ⚠️ This directory does not exist!")
        pdbs = glob.glob(os.path.join(struct_frac_path, "MD*_center*.pdb"))

        pdbs.sort()

        for pdb in pdbs:
            true_pdb_order.append(os.path.basename(pdb).replace('.pdb', ''))

    pdb_names = pd.Series(true_pdb_order)

    s_full = np.genfromtxt(path_exp_file, comments="#", usecols=0)
    s_values = s_full[:iq_sim_matrix.shape[1]]

    return s_values, iq_sim_matrix, None, pdb_names


def match_files(sim_file, save_path, dro, r0, structure_path):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb', '', regex=False)

    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None,
                          names=['index', 'dro', 'r0'])
    matching_rows = grid_df.loc[(grid_df['dro'] == dro) & (grid_df['r0'] == r0), 'index']
    if matching_rows.empty:
        print("\n" + "="*50)
        print("ERROR: Could not find matching grid point!")
        print(f"Looking for: dro = {dro} (type: {type(dro)}), r0 = {r0} (type: {type(r0)})")
        print("\nAvailable unique 'dro' values in grid_full.txt:")
        print(grid_df['dro'].unique()[:10]) # Print first 10 to inspect
        print("\nAvailable unique 'r0' values in grid_full.txt:")
        print(grid_df['r0'].unique()[:10])
        print("="*50 + "\n")
        raise ValueError(f"Grid coordinates (dro={dro}, r0={r0}) do not exist in {os.path.join(save_path, 'grid_full.txt')}")

    gp = matching_rows.iloc[0]

    compiled_saxs = np.genfromtxt("{}/compiled_GPs/GP{}_all_saxs.txt".format(save_path, str(gp)))
    compiled_df = pd.DataFrame(compiled_saxs).reset_index(drop=True)
    iq_sim_matrix = pd.DataFrame(compiled_df.drop(columns=[0]).values)

    true_pdb_order = []

    search_pattern = os.path.join(save_path, "mm*", f"GP{gp}", "calc_saxs.txt")
    sorted_files = natsorted(glob.glob(search_pattern))

    for file in sorted_files:
        parts = file.split(os.sep)
        frac_folder = [p for p in parts if p.startswith('mm') and '_' in p][0]

        parent_folder = frac_folder.split('_')[0]
        struct_frac_path = os.path.join(structure_path, parent_folder, frac_folder)
        pdbs = glob.glob(os.path.join(struct_frac_path, "mm*.pdb"))

        pdbs.sort()

        for pdb in pdbs:
            true_pdb_order.append(os.path.basename(pdb).replace('.pdb', ''))

    pdb_names = pd.Series(true_pdb_order)

    s_full = np.genfromtxt("{}/mm016_100/qvals.txt".format(save_path), skip_header=0)
    s_values = s_full[:iq_sim_matrix.shape[1]]

    return s_values, iq_sim_matrix, None, pdb_names

def VACC_average_curve(sim_file, pdb_names, s_val, iq_val):
    sim_pd = pd.read_csv(sim_file, sep='\\t', header=0)
    sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace('.pdb','',regex=False)

    weight_map = dict(zip(sim_pd['PDB_Name'], sim_pd['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])
    iq_array = np.array(iq_val)

    ##Debug
    print("\n" + "="*50)
    print("DEBUGGING: VACC_average_curve")
    print(f"  Total SAXS simulation frames (iq_array rows): {iq_array.shape[0]}")
    print(f"  Total mapped physical PDBs (ordered_weights): {len(ordered_weights)}")
    print(f"  PDBs passed from match_files:                 {len(pdb_names)}")
    print(f"  Unique weight entries parsed from sim_file:   {len(weight_map)}")

    missing_weights = np.sum(ordered_weights == 0.0)
    print(f"  PDBs mapped with a weight of 0.0 (or missing): {missing_weights}")

    if iq_array.shape[0] != len(ordered_weights):
        print("\n!!! ERROR TRIPPED !!!")
        print(f"  Your SAXS matrix expects {iq_array.shape[0]} structures, "
              f"but your PDB path searching only resolved {len(ordered_weights)} files.")
        print("="*50 + "\n")
    else:
        print("Shapes match perfectly. Continuing multiplication...")
        print("="*50 + "\n")


    prior_iq = np.mean(iq_array, axis=0)

    weighted_matrix = iq_array * ordered_weights[:, np.newaxis]
    avg_iq = np.sum(weighted_matrix, axis=0)

    sim_merge = pd.concat([pd.DataFrame(s_val), pd.DataFrame(avg_iq), pd.DataFrame(prior_iq)], axis=1)

    return len(sim_merge), sim_merge.iloc[:, 0], sim_merge.iloc[:, 1], sim_merge.iloc[:, 2]
    print("Breakpt")

def plot_curves(s_sim_arr, iq_sim_arr, iq_prior_arr, s, iq, err, post_rg_arr, pri_rg_arr, exp_rg, wam, dro_list, r0_list):
    colors_post = sns.color_palette("flare", n_colors=wam)
    colors_pri = sns.color_palette("crest", n_colors=wam)

    fig, ax = plt.subplots(figsize = (10,10))
    ax.errorbar(s, iq, yerr = err, fmt= 'o', markersize=3, ecolor="lightgray", label="Experiment", zorder=1)
    ax.set_yscale("log")

    for i in range(wam):
        scale = np.sum(iq * iq_sim_arr[i]) / np.sum(iq_sim_arr[i] ** 2)
        scaled_iq_sim = iq_sim_arr[i] * scale

        ax.plot(s_sim_arr[i], scaled_iq_sim, lw=3, zorder= i + 10, label=fr"$\delta \rho$: {dro_list[i]} | $r_0$: {r0_list[i]}", color=colors_post[i])
    ax.set_ylabel("i(q)")
    ax.set_xlabel("s")
    ax.set_title("Posterior SAXS curves with experiment")

    leg_post = ax.legend(loc="upper right")
    ax.add_artist(leg_post)

    rg_handles_post = [mlines.Line2D([], [], color='none', label=f"Exp Rg: {exp_rg:.2f} nm")]
    for i in range(wam):
        label_text = fr"$\delta \rho$: {dro_list[i]} | $r_0$: {r0_list[i]} Rg: {post_rg_arr[i]:.2f}"
        rg_handles_post.append(mlines.Line2D([], [], color='none', label=label_text))

    ax.legend(handles=rg_handles_post, loc="lower left", title="Radius of Gyration ($R_g$)", handlelength=0,
              handletextpad=0)

    fig_2, ax_2 = plt.subplots(figsize=(10, 10))
    ax_2.errorbar(s, iq, yerr=err, fmt='o', markersize=3, ecolor="lightgray", label="Experiment")
    ax_2.set_yscale("log")

    for i in range(wam):
        scale_p = np.sum(iq * iq_prior_arr[i]) / np.sum(iq_prior_arr[i] ** 2)
        scaled_iq_prior = iq_prior_arr[i] * scale_p

        ax_2.plot(s_sim_arr[i], scaled_iq_prior, lw=3, label=fr"$\delta \rho$: {dro_list[i]} | $r_0$: {r0_list[i]}", color=colors_pri[i])
    ax_2.set_ylabel("i(q)")
    ax_2.set_xlabel("s")
    ax_2.set_title("Prior SAXS curves with experiment")

    leg_pri = ax_2.legend(loc="upper right")
    ax_2.add_artist(leg_pri)

    # 2nd Legend (Rg Values)
    rg_handles_pri = [mlines.Line2D([], [], color='none', label=f"Exp Rg: {exp_rg:.2f} nm")]
    for i in range(wam):
        label_text = fr"$\delta \rho$: {dro_list[i]} | $r_0$: {r0_list[i]} Rg: {pri_rg_arr[i]:.2f} nm"
        rg_handles_pri.append(mlines.Line2D([], [], color='none', label=label_text))

    ax_2.legend(handles=rg_handles_pri, loc="lower left", title="Radius of Gyration ($R_g$)", handlelength=0,
                handletextpad=0)

    return fig, fig_2

def main():

    #####----- Initialize CLI arguments and configuration
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to main YAML file")
    args = parser.parse_parser_args() if hasattr(parser, 'parse_parser_args') else parser.parse_args()

    with open(args.config, "r") as f:
        master_config = yaml.safe_load(f)

    cc_config = master_config.get("curve_compare", {})
    if not cc_config:
        print("curve_compare config not found")
        return

    experimental_data = cc_config.get("experimental_data", "")
    structure_path = cc_config.get("structure_path", "")
    output_path = cc_config.get("output_path", "")

    lent_val = cc_config["lent"]
    lent = int(lent_val) if lent_val not in ["", None] else 1000

    dro_list = [cc_config.get("best_dro_1", ""), cc_config.get("best_dro_2", "")]
    r0_list = [cc_config.get("best_r0_1", ""), cc_config.get("best_r0_2","")]
    post_weights_list = [cc_config.get("post_weights_1", ""), cc_config.get("post_weights_2", "")]
    grid_paths_list = [cc_config.get("grid_sum_1", ""), cc_config.get("grid_sum_2", "")]
    save_paths_list = [cc_config.get("save_path_1", ""), cc_config.get("save_path_2", "")]

    wam = len([x for x in post_weights_list if x not in ["", None]])
    if wam == 0:
        print("No weights found")
        return

    #####----- Run main script
    s_arr = []
    wiq_arr = []
    piq_arr = []
    pos_rg_arr = []
    pri_rg_arr = []

    #Will be overwritten for real plotting
    angletrun, intensetrun, errtrun = experimental_curve(experimental_data, lent)
    exp_rg, exp_r2 = rg_guinier(angletrun, intensetrun, 10)

    for i in range(wam):
        if type == "c_term":
            s, concat_merge, weights, f_name = cterm_match_files(post_weights_list[i], save_paths_list[i], dro_list[i], r0_list[i], structure_path, experimental_data)
        else:
            s, concat_merge, weights, f_name = match_files(post_weights_list[i], save_paths_list[i], dro_list[i], r0_list[i], structure_path)
        lent, s_weighted, iq_weighted, iq_prior = VACC_average_curve(post_weights_list[i], f_name, s, concat_merge)
        s_arr.append(s_weighted)
        wiq_arr.append(iq_weighted)
        piq_arr.append(iq_prior)

        if type == "c_term":
            post_rg, prior_rg = cterm_grab_rg(post_weights_list[i], save_paths_list[i], dro_list[i], r0_list[i], f_name)
        else:
             post_rg, prior_rg = grab_rg(post_weights_list[i], save_paths_list[i], dro_list[i], r0_list[i], f_name)
        pos_rg_arr.append(post_rg)
        pri_rg_arr.append(prior_rg)

    angletrun, intensetrun, errtrun = experimental_curve(experimental_data, lent)
    post_figure, pri_figure = plot_curves(s_arr, wiq_arr, piq_arr, angletrun, intensetrun, errtrun, pos_rg_arr, pri_rg_arr, exp_rg, wam, dro_list, r0_list)

    post_figure.savefig(f"{output_path}/test_posterior_saxs_curves.png", dpi=300, bbox_inches='tight')
    pri_figure.savefig(f"{output_path}/test_prior_saxs_curves.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
