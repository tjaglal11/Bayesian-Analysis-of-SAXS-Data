#####----- Compare radius of gyration distributions between multiple sets of reweighed structures

#####----- IMPORTS
import numpy as np
import pandas as pd
import os
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

#####----- FUNCTIONS
def concat_rg(best_dro, best_r0, grid_path, save_path):
    grid_df = pd.read_csv(grid_path, sep=',')
    gp = grid_df.index[(grid_df['d_rho'] == best_dro) & (grid_df['r0'] == best_r0)].tolist()[0]

    search_pattern = os.path.join(save_path, "mm*", f"GP{gp}", "Rg_env.dat")
    found_files =  glob.glob(search_pattern)

    sorted_files = natsorted(found_files)
    rg_list = []

    for file in sorted_files:
        data = pd.read_csv(file, sep='\s+', header=None, names=['Rg'])

        parts = file.split(os.sep)
        frac_folder = [p for p in parts if p.startswith('mm')][0]
        system, suffix = frac_folder.split('_')

        if suffix == "end":
            data['frame'] = range(len(data))
            data['is_end'] = True
        else:
            block_end = int(suffix)
            block_start = block_end - len(data) + 1
            data['frame'] = data.index + block_start
            data['is_end'] = False

        data['system'] = system
        rg_list.append(data)

    rg_df = pd.concat(rg_list, ignore_index=True)
    for system in rg_df['system'].unique():
        mask_sys = rg_df['system'] == system
        mask_end = mask_sys & (rg_df['is_end'])

        if mask_end.any():
            max_frame = rg_df.loc[mask_sys & (~rg_df['is_end']), 'frame'].max()
            n_end = mask_end.sum()

            rg_df.loc[mask_end, 'frame'] = range(int(max_frame) + 1, int(max_frame) + 1 + n_end)

    rg_df = rg_df.drop(columns=['is_end'])

    return rg_df

def init_plot(rg_df_list, post_weights, best_dro, best_r0, wam, experimental_rg):

    #Posterior plot
    post_fig, post_ax = plt.subplots(figsize = (10,10))
    sns.set_style("ticks")
    colors_post = sns.color_palette("flare", n_colors=wam)
    colors_pri = sns.color_palette("crest", n_colors=wam)
    for i in range(wam):
        a = 0.9
        post_weights_ind = pd.read_csv(post_weights[i], sep='\s+')
        post_weights_ind.columns = ['index', 'weight', 'PDB_Name']

        post_weights_ind[['system', 'frame']] = post_weights_ind['PDB_Name'].str.extract(r'(mm\d+)_(\d+)\.pdb')
        post_weights_ind['frame'] = post_weights_ind['frame'].astype(int)

        # merge to align correctly
        merged = pd.merge(rg_df_list[i], post_weights_ind, on=['system', 'frame'], how='inner')

        rg_sim_post = merged['Rg'].to_numpy()
        post_weights_arr = merged['weight'].to_numpy()

        sns.kdeplot(x=rg_sim_post, weights=post_weights_arr, color=colors_post[i], label=fr'$\delta \rho$: {best_dro[i]} | $r_0$: {best_r0[i]}', linewidth= a, ax=post_ax)
        a -= 0.1
    post_ax.axvline(x=experimental_rg, color='green', linestyle='--', label='Experimental Rg')

    post_ax.set_xlabel('Rg distribution in Angstrom')
    post_ax.set_ylabel('Density')
    post_ax.set_title('Reweighted Rg distribution shift')
    post_ax.legend()

    #Prior plot
    pri_fig, pri_ax = plt.subplots(figsize = (10,10))
    sns.set_style("ticks")

    for i in range(wam):
        a_2 = 0.9
        rg_sim_prior = rg_df_list[i]['Rg'].to_numpy()
        prior_weights = np.ones(len(rg_sim_prior)) / len(rg_sim_prior)
        sns.kdeplot(x=rg_sim_prior, weights=prior_weights, color=colors_pri[i], label=fr'$\delta \rho$: {best_dro[i]} | $r_0$: {best_r0[i]}', linewidth=a_2, ax=pri_ax)
        a_2 -= 0.1
    pri_ax.axvline(x=experimental_rg, color='green', linestyle='--', label='Experimental Rg')

    pri_ax.set_xlabel('Rg distribution in Angstrom')
    pri_ax.set_ylabel('Density')
    pri_ax.set_title('Unweighted Rg distribution shift')
    pri_ax.legend()

    return post_fig, pri_fig

def main():

    #####----- Initialize CLI arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_parser_args() if hasattr(parser, 'parse_parser_args') else parser.parse_args()

    with open(args.config, "r") as f:
        master_config = yaml.safe_load(f)

    rg_config = master_config.get("rg_gps", {})
    if not rg_config:
        print("rg_gps config not found")
        return

    experimental_data = rg_config.get("experimental_data", "")
    experimental_rg = rg_config.get("experimental_rg", "")
    output_path = rg_config.get("output_path", "")

    best_dro = [rg_config.get("best_dro_1", ""), rg_config.get("best_dro_2", ""),]
    best_r0 = [rg_config.get("best_r0_1", ""), rg_config.get("best_r0_2", ""),]
    post_weights = [rg_config.get("post_weights_1", ""), rg_config.get("post_weights_2", ""),]
    grid_paths = [rg_config.get("grid_sum_1", ""), rg_config.get("grid_sum_2", ""),]
    save_paths = [rg_config.get("save_path_1", ""), rg_config.get("save_path_2", ""),]

    wam = len([x for x in post_weights if x not in ["", None]])
    if wam == 0:
        print("No weights found")
        return

    #####----- Run main script
    rg_df_list = []
    for i in range(wam):
        print(f"Processing run {i+1}/{wam}...")
        rg_df = concat_rg(best_dro[i], best_r0[i], grid_paths[i], save_paths[i])
        rg_df_list.append(rg_df)

    print("Generating plots...")
    posterior_plot, prior_plot = init_plot(rg_df_list, post_weights, best_dro, best_r0, wam, experimental_rg)
    posterior_plot.savefig(f"{output_path}/posterior_rg_distribution_ref.png", dpi=300, bbox_inches='tight')
    prior_plot.savefig(f"{output_path}/prior_rg_distribution_ref.png", dpi=300, bbox_inches='tight')

    print("Success!")
if __name__ == '__main__':
    main()

