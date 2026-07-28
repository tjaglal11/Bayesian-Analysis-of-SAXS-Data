# Fraction_Main_Pipeline/all_frames_grid.py
import glob
import os
import subprocess
import time
from datetime import date

import numpy as np


def create_grid(dro_min, dro_max, dro_step, r0_min, r0_max, r0_step):
    dro_grid = np.arange(dro_min, dro_max + dro_step, dro_step)
    r0_grid = np.arange(r0_min, r0_max + r0_step, r0_step)

    grid_data = []
    rm = 1.61
    index = 0

    for dro in dro_grid:
        for r0 in r0_grid:
            grid_data.append([
                index,
                round(dro, 2),
                round(r0 / rm, 4),
            ])
            index += 1

    return np.array(grid_data), index


# ---------- Configure these paths ----------
path_frames = "/path/to/structure/files"
path_shell = "/path/to/shell/job/files"
path_results = "/results/output"

theta_val = 100
max_parallel_grid_jobs = 20
# -------------------------------------------

today = date.today()
run_number = 0
run_template = "all_frames_run_{}_{}"
sub_path = os.path.join(path_results, run_template)

while os.path.isdir(sub_path.format(today, run_number)):
    run_number += 1

sub_path = sub_path.format(today, run_number)
os.mkdir(sub_path)

grid, number_of_grid_points = create_grid(30, 60, 5, 1.35, 1.65, 0.05)
np.savetxt(
    os.path.join(sub_path, "grid_full.txt"),
    grid,
    fmt=["%d", "%.2f", "%.4f"],
)

os.chdir(path_shell)

# Slurm task IDs are one-based because do_gp_fraction.sh interprets gl as a one-based line number in grid_full.txt.
scan_cmd = [
    "sbatch",
    "--parsable",
    f"--array=1-{number_of_grid_points}%{max_parallel_grid_jobs}",
    "iBME_all_frames_scan.sh",
    str(theta_val),
    sub_path,
    path_frames,
]

scan_run = subprocess.run(scan_cmd, capture_output=True, text=True)

if scan_run.returncode != 0:
    raise RuntimeError(f"Grid-array submission failed:\n{scan_run.stderr}")

array_job_id = scan_run.stdout.strip()
print(f"Submitted all-frames grid array: {array_job_id}")

reweight_cmd = [
    "sbatch",
    f"--dependency=afterok:{array_job_id}",
    "iBME_reweight.sh",
    str(theta_val),
    sub_path,
]

reweight_run = subprocess.run(reweight_cmd, capture_output=True, text=True)

if reweight_run.returncode != 0:
    raise RuntimeError(f"Reweight submission failed:\n{reweight_run.stderr}")

print(f"Submitted dependent reweight job: {reweight_run.stdout.strip()}")