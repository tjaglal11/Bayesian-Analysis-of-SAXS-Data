import os
from natsort import natsorted
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import re
import matplotlib.lines as mlines

def concat_gp(save_path):
    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None, names=['index', 'dro', 'r0'])
    num_gps = len(grid_df)
    compiled_dir = os.path.join(save_path, "compiled_GPs")
    os.makedirs(compiled_dir, exist_ok=True)

    print(f"Found {num_gps} grid points. Starting concatenation...")
    for i in range(num_gps):
        search_pattern = os.path.join(save_path, "mm*", f"GP{i}", "calc_saxs.txt")
        found_files = glob.glob(search_pattern)
        if not found_files:
            continue

        sorted_files = natsorted(found_files)
        compiled_data = [pd.read_csv(file, sep='\s+', header=None) for file in sorted_files]
        final_df = pd.concat(compiled_data, ignore_index=True)

        output_file = os.path.join(compiled_dir, f"GP{i}_all_saxs.txt")
        final_df.to_csv(output_file, sep=' ', index=False, header=False)

def concat_gp_cterm(save_path, struct_path):
    grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None, names=['index', 'dro', 'r0'])
    num_gps = len(grid_df)
    compiled_dir = os.path.join(save_path, "compiled_GPs")
    os.makedirs(compiled_dir, exist_ok=True)

    print(f"Found {num_gps} grid points. Starting concatenation...")
    for i in range(num_gps):
        search_pattern = os.path.join(save_path, "MD*_*", f"GP{i}", "calc_saxs.txt")
        found_files = glob.glob(search_pattern)
        if not found_files:
            continue

        compiled_data = []
        pdb_names = []
        for file in found_files:
            md_folder = os.path.basename(os.path.dirname(os.path.dirname(file)))

            df = pd.read_csv(file, sep='\s+', header=None)
            compiled_data.append(df)

            pdb_search = os.path.join(struct_path, "MD*", md_folder, "*.pdb")
            pdbs = natsorted(glob.glob(pdb_search))
            pdb_names.extend(os.path.basename(p) for p in pdbs)

            if len(pdbs) != len(df):
                print(f"Warning: {len(pdbs)} PDBs found for {md_folder}, but {len(df)} frames in {file}.")

        final_df = pd.concat(compiled_data, ignore_index=True)
        output_file = os.path.join(compiled_dir, f"GP{i}_all_saxs.txt")
        final_df.to_csv(output_file, sep=' ', index=False, header=False)

        manifest_file = os.path.join(compiled_dir, f"GP{i}_manifest.txt")
        pd.Series(pdb_names).to_csv(manifest_file, header=None, index=False)

def slurm_wait(job_ids, poll_interval=30):
    if not job_ids:
        print("No SLURM jobs submitted; nothing to wait for.")
        return

    job_ids = [str(job_id).strip() for job_id in job_ids]
    print(f"Waiting for {len(job_ids)} SLURM jobs to finish...")

    while True:
        remaining_jobs = []

        for job_id in job_ids:
            check = subprocess.run(
                ["squeue", "-j", job_id, "-h"],
                capture_output=True,
                text=True
            )

            if check.returncode == 0 and check.stdout.strip():
                remaining_jobs.append(job_id)

        if not remaining_jobs:
            print("All grid_scan.sh jobs have finished.")
            break

        print(f"{len(remaining_jobs)} jobs still running/pending. Checking again in {poll_interval} seconds...")
        time.sleep(poll_interval)

def set_experiment(saxs_path, exp_path, trun_path):
    sample_saxs = f"{saxs_path}/GP1/calc_saxs.txt" #saxs_path here to sample all_saxs calculation
    sample_df = pd.read_csv(sample_saxs, sep='\s+', header=None)
    sim_length = len(sample_df.columns) - 1

    exp_pd = pd.read_csv(exp_path, header=None, sep='\s+')
    exp_trun = exp_pd.iloc[:sim_length]
    exp_trun.to_csv(trun_path, header=False, index=False, sep=' ')

    with open(trun_path, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write("# DATA=SAXS BOUNDS=UPPER\n" + content)

    return trun_path

def ibme_worker(i, dro, r0, theta, calc_path, gp_out_dir, trun_path):
    os.makedirs(gp_out_dir, exist_ok=True)
    chi2b = chi2a = phi = np.nan

    try:
        # Run iBME
        iBME_script.iBMEf(trun_path, calc_path, theta, f"{gp_out_dir}/")

        # Parse Logs
        logs = glob.glob(os.path.join(gp_out_dir, "_ibme_*.log"))
        logs_sorted = sorted(logs, key=lambda x: int(re.search(r"_ibme_(\d+)\.log", x).group(1)))
        log_file = logs_sorted[-1] if logs_sorted else None

        if log_file:
            with open(log_file) as lf:
                for L in lf:
                    if "CHI2 before optimization:" in L:
                        chi2b = float(L.split()[-1])
                    elif "CHI2 after optimization:" in L:
                        chi2a = float(L.split()[-1])
                    elif "Fraction of effective frames:" in L:
                        phi = float(L.split()[-1])
        print(f"GP{i} optimized.")
    except Exception as e:
        print(f"iBME failed for GP{i}: {e}")

    rows = [[i, dro, r0, chi2b, chi2a, phi]]
    grid = np.array(rows, dtype=float)
    np.savetxt(os.path.join(gp_out_dir, f"GRID_opt_{i}"), grid, header="idx d_rho r0 CHI2_before CHI2_after PHI_eff",
               fmt="%.6g")

    return {"idx": i, 'd_rho': dro, "r0": r0, "CHI2_before": chi2b, "CHI2_after": chi2a, "PHI": phi}

def best_params(grid):
    #Extract dRho and r0 values from the grid
    dro_vals = np.unique(grid[:, 1])
    r0_vals = np.unique(grid[:, 2])
    order = np.lexsort((grid[:, 1], grid[:, 2]))
    grid = grid[order]

    #Extract CHI2 and phi values
    chi2 = np.clip(grid[:, 4], 1e-12, None)
    phi = np.clip(grid[:, 5], 1e-12, None)

    #Convert all phi values to Skl
    skl = -np.log(phi)

    #Find gamma using formula transformation
    gamma = np.log(chi2) + skl

    #reshape for minimum values
    chi2_mat = np.log(chi2).reshape(len(r0_vals), len(dro_vals))
    phi_mat = phi.reshape(len(r0_vals), len(dro_vals))
    skl_mat = skl.reshape(len(r0_vals), len(dro_vals))
    gam_mat = gamma.reshape(len(r0_vals), len(dro_vals))

    #find the best SAXS parameters
    min_y, min_x = np.unravel_index(np.nanargmin(gam_mat), gam_mat.shape)
    best_dro = dro_vals[min_x]
    best_r0 = r0_vals[min_y]

    f_chi2 = chi2_mat[min_y, min_x]
    f_skl = skl_mat[min_y, min_x]
    f_gamma = gam_mat[min_y, min_x]

    return best_dro, best_r0, f_chi2, f_skl, f_gamma

def heatmap(grid_sum_path, ibme_out_dir, theta):
    grid = np.loadtxt(grid_sum_path, skiprows=1, delimiter=',')  # Skip pandas header

    # Extract dRho and r0 values from the grid
    dro_vals = np.unique(grid[:, 1])
    r0_vals = np.unique(grid[:, 2])
    order = np.lexsort((grid[:, 1], grid[:, 2]))
    grid = grid[order]

    # Extract CHI2 and phi values
    chi2 = np.clip(grid[:, 4], 1e-12, None)
    phi = np.clip(grid[:, 5], 1e-12, None)

    # Convert all phi values to Skl
    skl = -np.log(phi)

    # Find gamma using formula transformation
    gamma = np.log(chi2) + skl

    # reshape for minimum values
    chi2_mat = np.log(chi2).reshape(len(r0_vals), len(dro_vals))
    phi_mat = phi.reshape(len(r0_vals), len(dro_vals))
    skl_mat = skl.reshape(len(r0_vals), len(dro_vals))
    gam_mat = gamma.reshape(len(r0_vals), len(dro_vals))

    # find the best SAXS parameters
    min_y, min_x = np.unravel_index(np.nanargmin(gam_mat), gam_mat.shape)
    best_dro = dro_vals[min_x]
    best_r0 = r0_vals[min_y]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    #####----- Plot heatmap for specific theta
    im0 = axs[0].imshow(chi2_mat, origin='upper', aspect='auto')
    axs[0].set_title(r'$\ln(\chi^2_{\mathrm{after}})$')
    axs[0].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(phi_mat, origin='upper', aspect='auto')
    axs[1].set_title(r'$\phi_{\mathrm{eff}}$')
    axs[1].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(gam_mat, origin='upper', aspect='auto')
    axs[2].set_title(r'$\gamma=\ln(\chi^2_{\mathrm{after}}/\phi_{\mathrm{eff}})$')
    axs[2].scatter(min_x, min_y, s=60, marker='o', facecolors='none', edgecolors='k')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    xticks = np.arange(0, len(dro_vals), 2)
    yticks = np.arange(0, len(r0_vals), 2)
    for ax in axs:
        ax.set_xticks(xticks);
        ax.set_xticklabels([f'{dro_vals[i]:.2f}' for i in xticks], rotation=300)
        ax.set_yticks(yticks);
        ax.set_yticklabels([f'{r0_vals[i]:.3f}' for i in yticks])
        ax.set_xlabel(r'$\delta\rho$  [$e/\mathrm{nm}^3$]')
    axs[0].set_ylabel(r'$r_0/r_m$')

    fig.suptitle(f'Best: δρ={best_dro:.2f}, r0={best_r0:.3f}', y=1.0)
    plt.tight_layout()

    heatmap_path = os.path.join(ibme_out_dir, f'grid_heatmaps_{theta}.png')
    fig.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to {heatmap_path}")

def save_weights(ibme_out_dir, struc_path, grid_path, dro, r0):
    GRID_DF = pd.read_csv(grid_path, sep='\s+', header=None, names=['index', 'dro', 'r0'])

    dro_val = float(dro)
    r0_val = float(r0)

    weight_idx = GRID_DF.index[(GRID_DF['dro'] == dro_val) & (GRID_DF['r0'] == r0_val)].tolist()[0]
    best_gp_dir = os.path.join(ibme_out_dir, f"GP1")

    # Dynamically find the last .weights.dat file
    weight_files = glob.glob(os.path.join(ibme_out_dir, "*.weights.dat"))
    if not weight_files:
        raise FileNotFoundError(f"No .weights.dat files found in {best_gp_dir}")

    weight_files_sorted = sorted(weight_files,
                                 key=lambda x: int(re.search(r"_(\d+)\.weights\.dat", os.path.basename(x)).group(1)))
    best_weight_file = weight_files_sorted[-1]

    #Get a sorted list of ALL structure names to map the weights back to the PDBs
    all_structures = glob.glob(os.path.join(struc_path, "*.pdb"))
    contents = pd.DataFrame(natsorted([os.path.basename(x) for x in all_structures]))

    # Map and save
    opt_weight = pd.read_csv(best_weight_file, sep=r'\s+', header=None)
    if opt_weight.empty or len(opt_weight.columns) < 2:
        raise ValueError(f"Weight file {best_weight_file} is empty or has insufficient columns")

    if contents.empty:
        raise ValueError(f"No structures found in {struc_path}")

    # Create a mapping dictionary from the contents DataFrame
    pdb_name_map = {i: name for i, name in enumerate(contents.iloc[:, 0])}
    opt_weight['PDB_Name'] = opt_weight.iloc[:, 0].map(contents.iloc[:, 0])
    opt_sorted = opt_weight.sort_values(by=1, ascending=False)

    weights_out = os.path.join(ibme_out_dir, f'structure_weights_sorted_{dro}_{r0}.txt')
    opt_sorted.to_csv(weights_out, index=None, sep='\t')

    print(f"Success! Top structure weights saved to: {weights_out}")

    return str(weights_out)

def plot_saxs_results(compiled_calc_path, experiment_path, weights_file, save_path, pdb_names, prior_rg, post_rg, exp_rg):
    #Adapted from ensemble_fit.py
    exp_pd = pd.read_csv(experiment_path, header=None, sep=r"\s+")
    s = exp_pd.iloc[:, 0].values
    iq_exp = exp_pd.iloc[:, 1].values
    err_exp = exp_pd.iloc[:, 2].values if exp_pd.shape[1] > 2 else np.zeros_like(s)

    sim_df = pd.read_csv(compiled_calc_path, sep='\s+', header=None, comment='#')
    iq_sim_matrix = sim_df.drop(columns=[0]).values

    weights_df = pd.read_csv(weights_file, sep='\t', header=0)
    weight_map = dict(zip(weights_df['PDB_Name'], weights_df['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])

    prior_iq = np.mean(iq_sim_matrix, axis=0)
    weighted_matrix = iq_sim_matrix * ordered_weights[:, np.newaxis]
    posterior_iq = np.sum(weighted_matrix, axis=0)

    sim_length = iq_sim_matrix.shape[1]
    s_trun = s[:sim_length]
    iq_trun = iq_exp[:sim_length]
    err_trun = err_exp[:sim_length]

    scale_post = np.sum(iq_trun * posterior_iq) / np.sum(posterior_iq ** 2)
    scaled_posterior = posterior_iq * scale_post
    scale_prior = np.sum(iq_trun * prior_iq) / np.sum(prior_iq ** 2)
    scaled_prior = prior_iq * scale_prior

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.errorbar(s_trun, iq_trun, yerr=err_trun, fmt='o', zorder=1, markersize=3, ecolor="lightgray",
                label="Experiment")
    ax.plot(s_trun, scaled_posterior, zorder=2, lw=3, label="Posterior", color="orange")
    ax.plot(s_trun, scaled_prior, zorder=3, lw=3, label="Prior", color="green")
    ax.set_yscale("log")
    ax.set_ylabel("i(q)")
    ax.set_xlabel("s")
    ax.set_title("Simulated SAXS fit with Experiment")

    leg_post = ax.legend(loc="upper right")
    ax.add_artist(leg_post)

    rg_handles = [mlines.Line2D([], [], color='none', label=f"Exp Rg: {exp_rg:.2f} nm")]
    pri_label_text = f"Prior rg: {prior_rg:.2f} nm"
    rg_handles.append(mlines.Line2D([], [], color='none', label=pri_label_text))
    post_label_text = f"Posterior rg: {post_rg:.2f} nm"
    rg_handles.append(mlines.Line2D([], [], color='none', label=post_label_text))

    ax.legend(handles=rg_handles, loc='lower left', title="Radius of gyration", handlength=0, handletextpad=0)

    plot_out = os.path.join(save_path, "truncated_fit.png")
    fig.savefig(plot_out, dpi=300)
    print(f"Plot saved to {plot_out}")

    return str(plot_out)

def cterm_grab_rg(sim_file, save_path, pdb_names):
    sim_pd = pd.read_csv(sim_file, sep='\t', header=0)
    #sim_pd["PDB_Name"] = sim_pd["PDB_Name"].str.replace(".pdb", "", regex=False)
    weight_map = dict(zip(sim_pd['PDB_Name'], sim_pd['1']))
    ordered_weights = np.array([weight_map.get(name, 0.0) for name in pdb_names])

    #grid_df = pd.read_csv(os.path.join(save_path, "grid_full.txt"), sep='\s+', header=None,
                          #names=['index', 'dro', 'r0'])
    #gp = grid_df.loc[(grid_df['dro'] == dro) & (grid_df['r0'] == r0), 'index'].iloc[0]

    search_pattern = os.path.join(save_path, f"GP1", "Rg_env.dat")
    sorted_files = natsorted(glob.glob(search_pattern))

    rg_list = []
    for file in sorted_files:
        data = pd.read_csv(file, sep='\s+', header=None, names=['Rg'])
        rg_list.extend(data['Rg'].tolist())

    rg_array = np.array(rg_list)

    prior_rg_real = np.mean(rg_array)
    post_rg_real = np.sum(rg_array * ordered_weights)

    return post_rg_real / 10, prior_rg_real / 10

def parse_grid(grid_line):
    grid_df = pd.read_csv(grid_line, sep='\s+', header=None, names=['index', 'dro', 'r0'])
    dro = grid_df['dro'].iloc[0]
    r0 = grid_df['r0'].iloc[0]

    return str(dro), str(r0)