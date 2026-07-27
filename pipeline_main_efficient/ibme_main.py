#####----- MAIN RUN FILE FOR iBME REWEIGHTING
import os
import argparse
import subprocess
import glob
import pandas as pd
import ibme_tools
import pipeline_main_efficient.ibme_tools
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

#####----- ARGUMENTS
parser = argparse.ArgumentParser(description="Main iBME run")
parser.add_argument("path_experimental", type=str)
parser.add_argument("path_grid", type=str)
parser.add_argument("path_structures", type=str)
parser.add_argument("theta_vals", type=float)
parser.add_argument("path_save", type=str)

args = parser.parse_args()

#####----- RUN

def main():

    ##Grid scan for SAXS sim

    grid_df = pd.read_csv(args.path_grid, sep="\s+", header=None, names=['index', 'dro', 'r0'])
    dro_min = grid_df["dro"].min()
    dro_max = grid_df["dro"].max()
    r0_min = grid_df["r0"].min()
    r0_max = grid_df["r0"].max()
    grid_job_ids = []

    for grid_point in range(len(grid_df)):
        #Path arguments to shell script to run Pepsi scan
        cmd = ["sbatch", "--parsable", "grid_scan.sh", args.path_structures, args.path_experimental, args.theta, str(grid_point), args.path_grid, args.path_save] #output path last arg
        run = subprocess.run(cmd, capture_output=True, text=True)

        if run.returncode == 0:
            job_id = run.stdout.strip()
            grid_job_ids.append(job_id)
        else:
            print(f"Error running grid_scan.sh for grid point {grid_point}: {run.stderr}")

    #Wait until all grid points have finished scanning before proceeding
    ibme_tools.slurm_wait(grid_job_ids)

    ##iBME
    exp_parent = os.path.dirname(args.path_experimental)
    trun_path = ibme_tools.set_experiment(args.path_save, args.path_experimental, os.path.join(exp_parent, "exp_trun.dat"))

    all_chi2 = []
    all_skl = []
    all_dro = []
    all_r0 = []
    all_gamma = []

    for theta in args.theta_vals:
        run_fol = f"iBME_dro_{dro_min}_to_{dro_max}_r0_{r0_min}_to_{r0_max}_theta_{theta}"
        ibme_out_dir = os.path.join(args.path_save, run_fol)
        os.makedirs(ibme_out_dir, exist_ok=True)

        results = []

        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(len(grid_df)):
                dro = grid_df.iloc[i]['dro']
                r0 = grid_df.iloc[i]['r0']
                calc_path = os.path.join(args.path_save, f"GP{i}_all_saxs.txt")
                gp_out_dir = os.path.join(ibme_out_dir, f"GP{i}")

                #Run iBME worker function
                futures.append(executor.submit(ibme_tools.ibme_worker, i, dro, r0, theta, calc_path, gp_out_dir, trun_path))

            for future in as_completed(futures):
                res = future.result()
                if 'error' not in res:
                    results.append(res)
                else:
                    print(f"iBME failed for GP{res['idx']}: {res['error']}")

        points = pd.DataFrame(results)
        grid_sum_path = os.path.join(ibme_out_dir, "GRID_sum.txt")
        points.to_csv(grid_sum_path, index=False)

        grid_sum = np.loadtxt(grid_sum_path, skiprows=1, delimiter=',')

        #Find the best parameters across all simulations
        best_dro, best_r0, f_chi2, f_skl, f_gamma = ibme_tools.best_params(grid_sum_path)

        all_chi2.append(f_chi2)
        all_skl.append(f_skl)
        all_dro.append(best_dro)
        all_r0.append(best_r0)
        all_gamma.append(f_gamma)

    ##Analysis
    chi_np = np.array(all_chi2)
    skl_np = np.array(all_skl)
    dro_np = np.array(all_dro)
    r0_np = np.array(all_r0)
    gamma_np = np.array(all_gamma)

    all_data = pd.DataFrame({"theta": args.theta_vals, "chi2": chi_np, "skl": skl_np, "gamma": gamma_np, "dro": dro_np, "r0": r0_np})
    best_gamma = all_data["gamma"].idxmin()
    best_row = all_data.loc[best_gamma]

    best_theta = best_row["theta"]
    best_run_fol = f"iBME_dro_{dro_min}_to_{dro_max}_r0_{r0_min}_to_{r0_max}_theta_{best_theta}"
    best_grid_sum_path = os.path.join(args.path_save, best_run_fol, "GRID_sum.txt")

    print(f"The best theta value is {best_row["theta"]} at dro {best_row['dro']} and r0 {best_row['r0']}. The chi2 value is {best_row['chi2']}, "
          f"and the skl value is {best_row['skl']}.")

    #Generate a heatmap for the best params
    ibme_tools.heatmap(best_grid_sum_path, best_run_fol, best_theta)

    #Save the weights for the best params
    ibme_tools.save_weights(best_run_fol, args.path_structures, args.path_grid, best_row["dro"], best_row["r0"])


