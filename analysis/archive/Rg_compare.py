#####----- Compare the radius of gyration of a IDP ensemble between i) experimental data ii) theoretical data and iii) reweighted data

#####----- IMPORTS
import numpy as np
import pandas as pd
import argparse
import os
import glob
from natsort import natsorted
import seaborn as sns
import matplotlib.pyplot as plt

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

def rg_dist_plot(rg_df, post_weights_path, experimental_rg, out_path):
    prior_weights = np.ones(len(rg_df)) / len(rg_df)
    post_weights = pd.read_csv(post_weights_path, sep='\s+')
    post_weights.columns = ['index', 'weight', 'PDB_Name']

    post_weights[['system', 'frame']] = post_weights['PDB_Name'].str.extract(r'(mm\d+)_(\d+)\.pdb')
    post_weights['frame'] = post_weights['frame'].astype(int)

    # merge to align correctly
    merged = pd.merge(rg_df, post_weights, on=['system', 'frame'], how='inner')

    rg_sim_prior = rg_df['Rg'].to_numpy()
    rg_sim_post = merged['Rg'].to_numpy()
    post_weights = merged['weight'].to_numpy()

    plt.figure(figsize=(10,10))
    sns.set_style("ticks")

    sns.kdeplot(x=rg_sim_prior, weights=prior_weights, color='red', label='Prior ensemble')
    sns.kdeplot(x=rg_sim_post, weights=post_weights, color='blue', label='Posterior ensemble')
    plt.axvline(x=experimental_rg, color='green', linestyle='--', label='Experimental Rg')

    plt.xlabel('Rg in Angstrom')
    plt.ylabel('Density')
    plt.title('Rg distrubtion in prior and posterior ensembles')
    plt.legend()

    plt.savefig("{}/rg_plot.png".format(out_path), dpi=300, bbox_inches='tight')
#####----- MAIN
parser= argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml", help="Path to main yaml file")
args = parser.parse_parser_args() if hasattr(parser, 'parse_parser_args') else p
arser.parse_args()

with open(args.config, "r") as f:
    master_config = yaml.safe_load(f)

rc_config = master_config.get("rg_compare", {})
if not rc_config:
    print("rg_compare config not found")
    return

experimental_data = rg_config.get["experimental_data", ""]
experimental_rg = rg_config.get["experimental_rg", ""]
post_weightss_path = rg_config.get["post_weights_path", ""]
save_path = rg_config.get["save_path", ""]
grid_path = rg_config.get["grid_path", ""]
out_path = rg_config.get["out_path", ""]
best_dro = rg_config.get["best_dro", ""]
best_r0 = rg_config.get["best_r0", ""]

rg_df = concat_rg(best_dro, best_r0, grid_path, save_path)
rg_dist_plot(rg_df, post_weights_path, experimental_rg, out_path)

