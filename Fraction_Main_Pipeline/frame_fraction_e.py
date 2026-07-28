############---------- REQUIREMENTS
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import date
import time
import glob
import re

#Date Time
epoch = time.time()
cd = time.strftime("%Y%m%d-%H%M%S", time.localtime(epoch))
today = date.today()

############---------- PATHS
path_frames = "/path/to/structure/files"
folder_pattern = os.path.join(path_frames, "mm*", "mm*_*")
structure_folders = sorted(glob.glob(folder_pattern))

path_exp_file = "/path/to/experimental/data"
path_main = "/main/path"
path_shell = "/path/to/shell/job/files"

#Make output file
path_results = "/results/output"
sub_dir_num = 0
run_fol = "run_{}_{}"
sub_path = os.path.join(path_results, run_fol)
while os.path.isdir(sub_path.format(today, sub_dir_num)):
    sub_dir_num += 1
sub_path = sub_path.format(today, sub_dir_num)
os.mkdir(sub_path)

############---------- FUNCTIONS

def create_grid(dro_min, dro_max, dro_step, r0_min, r0_max, r0_step):
    dro_grid = np.arange(dro_min, dro_max + dro_step, dro_step)
    r0_grid = np.arange(r0_min, r0_max + r0_step, r0_step)

    grid_data = []
    rm = 1.61
    index = 0
    for d in dro_grid:
        dro_val = round(d, 2)
        for r in r0_grid:
            r0_ratio = r / rm
            r0_val = round(r0_ratio, 4)
            grid_data.append([index, dro_val, r0_val])
            index += 1

    grid_name = np.array(grid_data)

    return grid_name, index

#############----------- MAIN

theta_val = 100
gn, idx = create_grid(30, 60, 5, 1.35, 1.65, .05)
gdf = pd.DataFrame(gn, columns=['index', 'dro', 'r0'])

np.savetxt(os.path.join(sub_path, "grid_full.txt"), gn, fmt=['%d', '%.2f', '%.2f'])
os.chdir(path_shell)

#Run Pepsi for fraction
fraction_job_ids = []

for struct_path in structure_folders:
    if os.path.isdir(struct_path):

        folder_name = os.path.basename(struct_path)
        spec_save = os.path.join(sub_path, folder_name)
        os.makedirs(spec_save, exist_ok=True)

        np.savetxt(
            os.path.join(spec_save, "grid_full.txt"),
            gn,
            fmt=['%d', '%.2f', '%.4f'])

        cmd = ["sbatch", "--parsable", "iBME_scan.sh", str(theta_val), spec_save, struct_path]
        run = subprocess.run(cmd, capture_output=True, text=True)

        if run.returncode == 0:
            job_id = run.stdout.strip()
            fraction_job_ids.append(job_id)
        else:
            print("Failed fulder submit")

if fraction_job_ids:
    # This formats the list into SLURM's required syntax: afterok:1234:1235:1236...
    dependency_string = f"afterok:{':'.join(fraction_job_ids)}"
    
    print(f"\nSubmitting final iBME reweighting job with dependency on {len(fraction_job_ids)} jobs...")
    
    final_cmd = [
        "sbatch", 
        f"--dependency={dependency_string}", 
        "iBME_reweight.sh", 
        str(theta_val), 
        sub_path # Notice we pass the MAIN sub_path here, not the fraction paths
    ]
    
    final_run = subprocess.run(final_cmd, capture_output=True, text=True)
    
    if final_run.returncode == 0:
        print(f"Success! Final job submitted: {final_run.stdout.strip()}")
    else:
        print(f"Failed to submit final job: {final_run.stderr}")
else:
    print("No fraction jobs were successfully submitted. Aborting final job.")
