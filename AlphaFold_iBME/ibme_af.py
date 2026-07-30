#####----- New pipeline for implementation of SAXS simulation and iBME refinement into AlphaFold framework
import sys
import os
import argparse
import subprocess
from pipeline_main_efficient import ibme_tools
import pandas as pd
import numpy as np
from gp_files import iBME_script
import glob
import re

#####----- ARGUMENTS
parser = argparse.ArgumentParser(description="Run iBME on AlphaFold output")
parser.add_argument("structure_path", type=str)
parser.add_argument("pepsi_path", type=str)
parser.add_argument("dro", type=str)
parser.add_argument("r0", type=str)
parser.add_argument("grid_line", type=str)
parser.add_argument("theta", type=float)
parser.add_argument("experiment_path", type=str)
parser.add_argument("save_path", type=str)
args = parser.parse_args()

#####----- RUN

def run_pepsi(structure_path, pepsi_path, experiment_path, save_path, grid_line, dro, r0):

    #Run Pepsi SAXS on input structures
    run = subprocess.run([f"{pepsi_path}/do_gp.sh", structure_path, experiment_path, 0, 0, grid_line, save_path],
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
    calc_path = os.path.join(save_path, "GP0/calc_saxs.txt")
    with open(calc_path, 'w') as f:
        f.write("# DATA=SAXS\n")
    df = pd.read_csv(calc_path, sep='\s+', header=None)
    df.to_csv(calc_path, mode='a', header=False, index=False, sep=' ')
    gp_out_dir = os.path.join(save_path, "GP0")

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
    np.savetxt(os.path.join(gp_out_dir, f"GRID_opt_{0}"), grid, header="idx d_rho r0 CHI2_before CHI2_after PHI_eff",
               fmt="%.6g")

    frames = pd.DataFrame(rows, columns=["idx", "d_rho", "r0", "CHI2_before", "CHI2_after", "PHI_eff"])
    results.append(frames)

    return str(run_fol)

def weights_analysis(run_fol, structure_path, grid_line, dro, r0):

    #Find and save current weights
    weights_out = ibme_tools.save_weights(run_fol, structure_path, grid_line, dro, r0)


