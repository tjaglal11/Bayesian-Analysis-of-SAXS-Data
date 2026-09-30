import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import glob
from natsort import natsorted

#####----- FUNCTIONS
def pdb_manifest(path_to_pdbs):
    search_pattern = os.path.join(pdbs, "*.pdb")
    found_files = glob.glob(search_pattern)

    print(f"Found {len(found_files)} PDB files.")

    sorted = natsorted(found_files)
    manifest = np.array(sorted)

    return manifest

def find_saxs()