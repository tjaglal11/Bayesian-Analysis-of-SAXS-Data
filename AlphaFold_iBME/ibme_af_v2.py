#####----- New pipeline for implementation of SAXS simulation and iBME refinement into AlphaFold framework
import os
import argparse
import subprocess
import ibme_tools
import pandas as pd
import numpy as np
import iBME_script
import glob
import re
from natsort import natsorted

#####----- ARGUMENTS
parser = argparse.ArgumentParser(description="Run iBME on AlphaFold output")
parser.add_argument("structure_path", type=str)
parser.add_argument("pepsi_path", type=str)
parser.add_argument("dro", type=str)
parser.add_argument("r0", type=str)
parser.add_argument("grid_line", type=str) #index for grid file must be 1, not 0
parser.add_argument("theta", type=float)
parser.add_argument("experiment_path", type=str)
parser.add_argument("save_path", type=str)
args = parser.parse_args()

exp_rg = 4.6
#####----- RUN

def run_pepsi(structure_path, pepsi_path, experiment_path, save_path, grid_line, dro, r0):

    #Run Pepsi SAXS on input structures
    run = subprocess.run([f"{pepsi_path}/do_gp_af.sh", structure_path, experiment_path, "0", "1", grid_line, save_path],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if run.returncode == 0:
        print(f"Pepsi SAXS simulation at dro={dro} and r0={r0} ran successfully with return code {run.returncode}")
    else:
        print(f"Script failed with return code {run.returncode}")

def run_ibme(structure_path, experiment_path, theta, save_path, dro, r0):

    #Initialize experiment
    exp_parent = os.path.dirname(experiment_path)
    trun_path = ibme_tools.set_experiment(save_path, experiment_path, os.path.join(exp_parent, "exp_trun.dat"))

    #Create output directory
    run_fol = f"iBME_dro_{dro}_r0_{r0}_theta_{theta}"
    ibme_out_dir = os.path.join(save_path, run_fol)
    os.makedirs(ibme_out_dir, exist_ok=True)

    results = []

    # Add # DATA=SAXS header to the compiled file
    calc_path = os.path.join(save_path, "GP1/calc_saxs.txt")
    df = pd.read_csv(calc_path, sep='\s+', header=None)
    with open(calc_path, 'w') as f:
        f.write("# DATA=SAXS\n")
    df.to_csv(calc_path, mode='a', header=False, index=False, sep=' ')
    gp_out_dir = os.path.join(save_path, "GP1")

    chi2b = chi2a = phi = np.nan 
    try:
        # Run iBME
        iBME_script.iBMEf(trun_path, calc_path, args.theta, f"{run_fol}/")

        # Parse Logs
        logs = glob.glob(os.path.join(run_fol, "_ibme_*.log"))
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
        print(f"GP optimized.")
    except Exception as e:
        print(f"iBME failed for GP: {e}")

    rows = [[0, dro, r0, chi2b, chi2a, phi]]
    grid = np.array(rows, dtype=float)
    np.savetxt(os.path.join(gp_out_dir, f"GRID_opt_1"), grid, header="idx d_rho r0 CHI2_before CHI2_after PHI_eff",
               fmt="%.6g")

    frames = pd.DataFrame(rows, columns=["idx", "d_rho", "r0", "CHI2_before", "CHI2_after", "PHI_eff"])
    results.append(frames)

    return str(run_fol), results

def main(structure_path, pepsi_path, experiment_path, save_path, grid_line, dro, r0, theta):

    ##Run pepsi
    print(f"Running Pepsi SAXS simulation at dro={dro} and r0={r0}...")
    run_pepsi(structure_path, pepsi_path, experiment_path, save_path, grid_line, dro, r0)

    ##Run iBME
    print(f"Running iBME at dro={dro} and r0={r0}...")
    run_fol, results = run_ibme(structure_path, experiment_path, theta, save_path, dro, r0)

    #results_sorted = natsorted(results, key=lambda x: x[0])
    all_pdbs = glob.glob(os.path.join(structure_path, "*.pdb"))
    pdb_names = [os.path.basename(f) for f in natsorted(all_pdbs)]

    ##Analysis

    #Save posterior weights
    print(f"Saving posterior weights to {run_fol}/structure_weights_sorted_{dro}_{r0}.txt...")
    weights_path = ibme_tools.save_weights(run_fol, structure_path, grid_line, dro, r0)

    gp0_dir = os.path.join(save_path, "GP1")
    compiled_calc_path = os.path.join(gp0_dir, "calc_saxs.txt")

    #Grab the Rg values
    print(f"Grabbing Rg values...")
    prior_rg, post_rg = ibme_tools.cterm_grab_rg(weights_path, save_path, pdb_names)

    #Plot the results on a SAXS trajectory
    print(f"Saving SAXS curve plot to {run_fol}...")
    plot_path = ibme_tools.plot_saxs_results(compiled_calc_path, experiment_path, weights_path, run_fol, pdb_names, prior_rg, post_rg, exp_rg)
    return weights_path, plot_path

#####----- MAIN
if __name__ == "__main__":
    path_weights, plot_path = main(args.structure_path, args.pepsi_path, args.experiment_path, args.save_path, args.grid_line, args.dro, args.r0, args.theta)
