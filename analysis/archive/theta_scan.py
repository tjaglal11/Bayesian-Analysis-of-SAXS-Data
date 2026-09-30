#!/usr/bin/env python3
"""Run an iBME theta scan from frame_fraction SAXS output."""

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GP_FILES = os.path.join(PROJECT_ROOT, "gp_files")
if GP_FILES not in sys.path:
    sys.path.insert(0, GP_FILES)
import iBME_script


def parse_args():
    parser = argparse.ArgumentParser(description="Run an iBME theta scan on amm* frame-fraction output.")
    parser.add_argument("experimental", help="Three-column experimental SAXS file (q, I, error).")
    parser.add_argument("save_path", help="Directory containing grid_full.txt and amm*/GP*/calc_saxs.txt.")
    parser.add_argument("out_path", help="Directory for theta-scan results.")
    parser.add_argument("theta", nargs="+", type=float, help="One or more theta values, e.g. 10 50 100.")
    parser.add_argument("--pattern", default="amm*", help="Ensemble-folder glob (default: amm*).")
    parser.add_argument("--structures", help="Parent directory containing amm*/PDB files; save the best posterior weights with PDB names.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent grid-point jobs per theta (default: 1).")
    return parser.parse_args()


def read_grid(save_path):
    path = os.path.join(save_path, "grid_full.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing grid file: {path}")
    grid = pd.read_csv(path, sep=r"\s+", header=None, names=["index", "dro", "r0"])
    if grid.empty:
        raise ValueError(f"Grid file is empty: {path}")
    return grid


def compile_curves(save_path, grid, pattern):
    compiled_dir = os.path.join(save_path, "compiled_GPs")
    os.makedirs(compiled_dir, exist_ok=True)
    missing = []
    for gp in grid["index"].astype(int):
        files = natsorted(glob.glob(os.path.join(save_path, pattern, f"GP{gp}", "calc_saxs.txt")))
        if not files:
            missing.append(str(gp))
            continue
        data = [pd.read_csv(path, sep=r"\s+", header=None, comment="#") for path in files]
        if any(frame.empty for frame in data):
            raise ValueError(f"An empty calc_saxs.txt was found for GP{gp}.")
        pd.concat(data, ignore_index=True).to_csv(
            os.path.join(compiled_dir, f"GP{gp}_all_saxs.txt"), sep=" ", header=False, index=False
        )
    if missing:
        raise FileNotFoundError(
            "Missing SAXS curves for grid points " + ", ".join(missing) + ". "
            f"Expected {save_path}/{pattern}/GP<number>/calc_saxs.txt."
        )
    return compiled_dir


def write_truncated_experiment(experimental, compiled_dir, first_gp, out_path):
    sample = pd.read_csv(os.path.join(compiled_dir, f"GP{first_gp}_all_saxs.txt"), sep=r"\s+", header=None)
    points = sample.shape[1] - 1  # The first column is the frame index.
    experiment = pd.read_csv(experimental, sep=r"\s+", header=None, comment="#")
    if points <= 0 or experiment.shape[1] != 3 or len(experiment) < points:
        raise ValueError("Simulation and experiment are incompatible: need a frame index plus intensities and >= that many experimental q-points.")
    path = os.path.join(out_path, "experimental_truncated.dat")
    with open(path, "w") as handle:
        handle.write("# DATA=SAXS BOUNDS=UPPER\n")
        experiment.iloc[:points].to_csv(handle, sep=" ", header=False, index=False)
    return path


def worker(gp, dro, r0, theta, calc_path, output_dir, experiment):
    os.makedirs(output_dir, exist_ok=True)
    result = {"idx": gp, "d_rho": dro, "r0": r0, "CHI2_before": np.nan,
              "CHI2_after": np.nan, "PHI_eff": np.nan, "error": ""}
    try:
        iBME_script.iBMEf(experiment, calc_path, theta, f"{output_dir}/")
        logs = glob.glob(os.path.join(output_dir, "_ibme_*.log"))
        if not logs:
            raise RuntimeError("iBME wrote no iteration log")
        log = max(logs, key=lambda path: int(re.search(r"_ibme_(\d+)\.log$", path).group(1)))
        with open(log) as handle:
            for line in handle:
                if "CHI2 before optimization:" in line:
                    result["CHI2_before"] = float(line.split()[-1])
                elif "CHI2 after optimization:" in line:
                    result["CHI2_after"] = float(line.split()[-1])
                elif "Fraction of effective frames:" in line:
                    result["PHI_eff"] = float(line.split()[-1])
        if not np.isfinite(result["CHI2_after"]) or not np.isfinite(result["PHI_eff"]):
            raise RuntimeError(f"Optimization did not complete successfully; see {log}")
    except Exception as exc:
        result["error"] = str(exc)
    return result


def make_heatmap(summary, output_dir, theta):
    grid = summary.sort_values(["r0", "d_rho"])
    dro, r0 = np.sort(grid.d_rho.unique()), np.sort(grid.r0.unique())
    if len(grid) != len(dro) * len(r0):
        raise ValueError("The grid is not rectangular; cannot make a heatmap.")
    chi = grid.CHI2_after.to_numpy().reshape(len(r0), len(dro))
    phi = grid.PHI_eff.to_numpy().reshape(len(r0), len(dro))
    gamma = np.log(chi / phi)
    y, x = np.unravel_index(np.argmin(gamma), gamma.shape)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    for axis, values, title in zip(axes, [np.log(chi), phi, gamma], [r"$\ln(\chi^2)$", r"$\phi_{eff}$", r"$\gamma=\ln(\chi^2/\phi_{eff})$"]):
        image = axis.imshow(values, origin="upper", aspect="auto")
        axis.scatter(x, y, s=60, facecolors="none", edgecolors="k")
        axis.set_xticks(range(len(dro))); axis.set_xticklabels([f"{value:.2f}" for value in dro], rotation=300)
        axis.set_yticks(range(len(r0))); axis.set_yticklabels([f"{value:.3f}" for value in r0])
        axis.set_xlabel(r"$\delta\rho$"); axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(r"$r_0/r_m$")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"grid_heatmaps_theta_{theta:g}_{date.today()}.png"), dpi=300)
    plt.close(fig)
    return float(dro[x]), float(r0[y]), float(chi[y, x]), float(-np.log(phi[y, x]))


def save_best_weights(run_dir, best_gp, structures, pattern):
    weights = glob.glob(os.path.join(run_dir, f"GP{best_gp}", "*.weights.dat"))
    if not weights:
        raise FileNotFoundError(f"No iBME weights found for GP{best_gp} in {run_dir}.")
    weight_path = max(weights, key=lambda path: int(re.search(r"_(\d+)\.weights\.dat$", path).group(1)))
    pdbs = natsorted(glob.glob(os.path.join(structures, pattern, "*.pdb")))
    frame_weights = pd.read_csv(weight_path, sep=r"\s+", header=None)
    if len(pdbs) != len(frame_weights):
        raise ValueError(f"GP{best_gp} has {len(frame_weights)} weights but {len(pdbs)} amm* PDB files were found.")
    frame_weights["PDB_Name"] = [os.path.basename(path) for path in pdbs]
    frame_weights.sort_values(by=1, ascending=False).to_csv(
        os.path.join(run_dir, f"structure_weights_sorted_{date.today()}.txt"), sep="\t", index=False
    )


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    grid = read_grid(args.save_path)
    compiled_dir = compile_curves(args.save_path, grid, args.pattern)
    experiment = write_truncated_experiment(args.experimental, compiled_dir, int(grid.iloc[0]["index"]), args.out_path)
    scan_rows = []
    for theta in args.theta:
        run_dir = os.path.join(args.out_path, f"theta_{theta:g}")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, int(row.index), row.dro, row.r0, theta,
                                       os.path.join(compiled_dir, f"GP{int(row.index)}_all_saxs.txt"),
                                       os.path.join(run_dir, f"GP{int(row.index)}"), experiment)
                       for row in grid.itertuples(index=False)]
            results = [future.result() for future in as_completed(futures)]
        summary = pd.DataFrame(results).sort_values("idx")
        os.makedirs(run_dir, exist_ok=True)
        summary_path = os.path.join(run_dir, "GRID_sum.txt")
        summary.to_csv(summary_path, index=False)
        failures = summary[summary.error != ""]
        if not failures.empty:
            print(f"theta={theta:g}: {len(failures)} grid point(s) failed; see {summary_path}")
            continue
        best = make_heatmap(summary, run_dir, theta)
        scan_rows.append({"theta": theta, "best_dro": best[0], "best_r0": best[1], "chi2": best[2], "skl": best[3]})
        if args.structures:
            best_gp = int(summary.loc[(summary.d_rho == best[0]) & (summary.r0 == best[1]), "idx"].iloc[0])
            save_best_weights(run_dir, best_gp, args.structures, args.pattern)
        print(f"theta={theta:g}: best d_rho={best[0]:.4g}, r0={best[1]:.4g}, chi2={best[2]:.4g}")
    if scan_rows:
        pd.DataFrame(scan_rows).to_csv(os.path.join(args.out_path, "theta_summary.csv"), index=False)


if __name__ == "__main__":
    main()
