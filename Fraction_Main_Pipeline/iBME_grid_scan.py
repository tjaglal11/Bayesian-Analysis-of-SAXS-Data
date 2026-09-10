############---------- REQUIREMENTS
import subprocess
import os
import pandas as pd
import glob
from natsort import natsorted
import argparse

#Parse arguments
parser = argparse.ArgumentParser(description='Run iBME for specific grid points')
parser.add_argument("theta", type=float)
parser.add_argument("save_path", type=str)
parser.add_argument("frame_path", type=str)

args = parser.parse_args()

#Other paths
path_main = "/main/path"
path_structures = "/path/to/structure/files"
path_exp_file = "/path/to/experimental/data"
contents = pd.DataFrame(natsorted(os.listdir(path_structures)))
calc_rows_path = "{}/GP{}/calc_rows.txt"
out_name = "{}/GP{}/"

gl = ""

############---------- FUNCTIONS

def concat_fractions(save_path):
    # 1. Read the grid to dynamically find out how many GPs we have
    # (Using the headerless format we set up previously)
    grid_df = pd.read_csv(os.path.join(args.save_path, "grid_full.txt"), sep='\s+', header=None, names=['index', 'dro', 'r0'])
    num_gps = len(grid_df)

    # 2. Create a clean output folder for the compiled results
    compiled_dir = os.path.join(args.save_path, "compiled_GPs")
    os.makedirs(compiled_dir, exist_ok=True)

    print(f"Found {num_gps} grid points. Starting concatenation...")

    # 3. Loop through every grid point dynamically
    for i in range(num_gps):

        # 4. Use glob with a wildcard (*) to find all files for THIS grid point
        # across all mm* structure fractions simultaneously
        search_pattern = os.path.join(save_path, "mm*", f"GP{i}", "calc_saxs.txt")
        found_files = glob.glob(search_pattern)

        if not found_files:
            print(f"Warning: No files found for GP{i}. Skipping.")
            continue

        # 5. CRITICAL: Sort files naturally so mm016_100 comes before mm016_200
        sorted_files = natsorted(found_files)

        # 6. Read and append all found files together

        named_files = [os.path.join(os.path.dirname(file), "calc_saxs_pdb.txt") for file in sorted_files]
        missing_named_files = [path for path in named_files if not os.path.isfile(path)]
        if missing_named_files:
            raise FileNotFoundError(
                f"Missing PDB files. first missing file {missing_named_files[0]}"
            )
        compiled_data = []
        compiled_named_data = []
        for file, named_file in zip(sorted_files, named_files):
            # Assuming these are space-separated without headers
            df = pd.read_csv(file, sep='\s+', header=None)
            compiled_data.append(df)
            compiled_named_data.append(pd.read_csv(named_file, sep='\s+', header=None))

        # 7. Concatenate all rows into one master DataFrame
        final_df = pd.concat(compiled_data, ignore_index=True)
        final_named_df = pd.concat(compiled_named_data, ignore_index=True)
        if len(final_df) != len(final_named_df):
            raise ValueError("Number of rows in named and unnamed files do not match.")

        # 8. Save the master file for this GP
        output_file = os.path.join(compiled_dir, f"GP{i}_all_saxs.txt")
        final_df.to_csv(output_file, sep=' ', index=False, header=False)
        final_named_df.to_csv(os.path.join(compiled_dir, f"GP{i}_all_saxs_pdb.txt"), sep=' ', index=False, header=False)

        pd.Series(final_named_df.iloc[:, 0]).to_csv(os.path.join(compiled_dir, f"GP{i}_manifest.txt"), header=None, index=False)

        print(f"GP{i} compiled successfully: Merged {len(sorted_files)} fractions.")
############---------- MAIN

# structure path, experiment path, theta, gl (optional), grid document
run = subprocess.run(["{}/do_gp_fraction.sh".format(path_main), args.frame_path, "{}/SASDLU4.dat".format(path_exp_file),
                      str(args.theta), gl, os.path.join(args.save_path, "grid_full.txt"), args.save_path],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(f"STDOUT:\n {run.stdout}")
print(f"STDERR:\n {run.stderr}")

if run.returncode == 0:
    print(f"Script successful with return code {run.returncode}")
else:
    print(f"Script failed with return code {run.returncode}")

#Concat


#Run iBME job



