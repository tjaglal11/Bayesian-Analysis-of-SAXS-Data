from os import name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse
import yaml

#####----- FUNCTIONs

def experimental_curve(path_exp_file, sim_length):
    exp_pd = pd.DataFrame(pd.read_csv(path_exp_file,
                                      header=None, sep=r"\s+"))

    s = exp_pd.loc[:,0]
    iq = exp_pd.loc[:,1]
    err = exp_pd.loc[:,2]

    s_trun = s.iloc[:sim_length].values
    iq_trun = iq.iloc[:sim_length].values
    err_trun = err.iloc[:sim_length].values

    return s_trun, iq_trun, err_trun

def find_curves(files, search_path):
    q = []
    iq = []
    column_names = ["q", "Iq exp", "dl exp", "Iq fit"]
    for i in range(len(files)):
        pdb = pd.read_csv(f"{search_path}/{files[i]}-SASDLU4.fit", sep=r"\s{2,}", skiprows=5, header=0, comment="#", names=column_names, engine="python")
        pdb.columns = pdb.columns.str.lstrip("#").str.strip()

        q_np = pdb["q"].to_numpy() * 10
        iq_np = pdb["Iq fit"].to_numpy()

        q.append(q_np)
        iq.append(iq_np)

    lent = len(q_np)
    return lent, q, iq

def plot_curves(s, iq, err, files, q_arr, iq_arr, save_path):
    colors = sns.color_palette("bright", n_colors=len(files))

    fig, ax = plt.subplots(figsize = (10,10))
    ax.errorbar(s, iq, yerr = err, fmt= 'o', markersize=3, ecolor="lightgray", label="Experiment")
    ax.set_yscale("log")

    for i in range(len(files)):
        sim_iq = iq_arr[i]
        scale = np.sum(iq * sim_iq) / np.sum(sim_iq ** 2)
        scaled_sim_iq = sim_iq * scale
        ax.plot(q_arr[i], scaled_sim_iq, zorder= i+2, lw=3, label= files[i], color=colors[i])
    ax.set_ylabel("i(q)")
    ax.set_xlabel("s")
    ax.set_title("SAXS curves vs Experiment")
    ax.legend()

    save_path_2 = "{}/ind_curves.png".format(save_path)
    fig.savefig(save_path_2, dpi=300)

def main():

    #####----- Initialize CLI arguments and configuration
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to main YAML file")
    args = parser.parse_parser_args() if hasattr(parser, 'parse_parser_args') else parser.parse_args()

    with open(args.config, "r") as f:
        master_config = yaml.safe_load(f)

    fc_config = master_config.get("file_curve", {})
    if not fc_config:
        print("file_curve not found in config")
        return

    path_exp_file = fc_config.get["path_exp_file", ""]
    search_path = fc_config.get["search_path", ""]
    save_path = fc_config.get["save_path", ""]

    file_1 = fc_config.get["file_1", ""]
    file_2 = fc_config.get["file_2", ""]
    file_3 = fc_config.get["file_3", ""]
    file_4 = fc_config.get["file_4", ""]
    file_5 = fc_config.get["file_5", ""]
    file_6 = fc_config.get["file_6", ""]

    #####----- MAIN
    files = [file_1, file_2, file_3, file_4, file_5, file_6]
    length, q_files, iq_files = find_curves(files, search_path)
    s_exp, iq_exp, err_exp = experimental_curve(path_exp_file, length)
    reference_q = q_files[0]

    #iq_exp_interp = np.interp(reference_q, s_exp, iq_exp)
    #err_exp_interp = np.interp(reference_q, s_exp, err_exp)

    # 5. Plot using the synchronized reference_q
    plot_curves(
        s_exp, iq_exp, err_exp, files, q_files, iq_files, save_path
    )

if __name__ == "__main__":
    main()

