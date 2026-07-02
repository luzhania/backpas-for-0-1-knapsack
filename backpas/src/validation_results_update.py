#!/usr/bin/env python3
"""
validation_results_update.py
Equivalent script to backpas/jupyter/validation_results_update.ipynb.

Usage:
    python validation_results_update.py [--dataset DATASET] [--partition PARTITION]
                                        [--configuration CONFIGURATION] [--no-save]
                                        [--no-plot]

Defaults (same as the notebook):
    dataset            = 14_bounded_strongly_correlated
    partition          = validation
    configuration      = full
"""

import argparse
import sys
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers to locate utils.py (which lives next to the notebook)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
JUPYTER_DIR = SCRIPT_DIR.parent / "jupyter"

if not JUPYTER_DIR.is_dir():
    print(
        f"[ERROR] Could not find the jupyter directory at '{JUPYTER_DIR}'. "
        "Make sure the script is located inside backpas/src/.",
        file=sys.stderr,
    )
    sys.exit(1)

sys.path.insert(0, str(JUPYTER_DIR))

try:
    from utils import (
        construct_instance_log_dict,
        get_all_logs_for_instance,
        create_temp_file_list,
    )
except ImportError as exc:
    print(
        f"[ERROR] Could not import from utils.py at '{JUPYTER_DIR / 'utils.py'}': {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration map  (mirrors the if/elif block in Cell 2)
# ---------------------------------------------------------------------------
CONFIGURATIONS = {
    "param": {
        "network": "graph_with_variables_2_GCN",
        "method": "thresholded_expected_error",
    },
    "net": {
        "network": "graph_with_literals_8_GTR",
        "method": "fixed_three_ratios",
    },
    "v0": {
        "network": "graph_with_variables_2_GCN",
        "method": "fixed_three_ratios",
    },
    "full": {
        "network": "graph_with_literals_3_GTR",
        "method": "thresholded_expected_error",
    },
}

# Datasets whose objective is minimisation vs maximisation
OBJECTIVE_MAP = {
    "MVC": "min",
    "14_bounded_strongly_correlated": "min",
    "MIS": "max",
    "MIS-mixed-train": "max",
    "CA": "max",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update validation results CSV (equivalent to validation_results_update.ipynb)."
    )
    parser.add_argument(
        "--dataset",
        default="14_bounded_strongly_correlated",
        help="Dataset name (default: 14_bounded_strongly_correlated)",
    )
    parser.add_argument(
        "--partition",
        default="validation",
        help="Partition name (default: validation)",
    )
    parser.add_argument(
        "--configuration",
        default="full",
        choices=list(CONFIGURATIONS.keys()),
        help="Configuration preset (default: full)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do NOT overwrite results.csv (equivalent to save_results=False in the notebook)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering the box-plot",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Path to save the box-plot image (e.g. plot.png). "
             "Defaults to boxplot_<configuration>.png next to results.csv.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Section: Parse method-specific parameter names from method directory names
# ---------------------------------------------------------------------------

def parse_name_fixed_three_ratios(method_name: str, param_name: str) -> str:
    """
    Parses a method name with format:
        k_{k_ratio}_value_0_{value_0_ratio}_delta_{delta_ratio}
    """
    parts = method_name.split("_")
    try:
        if param_name == "k_ratio":
            return parts[1]
        if param_name == "value_0_ratio":
            return parts[4]
        if param_name == "Delta_ratio":
            return parts[6]
    except IndexError:
        pass
    raise ValueError(
        f"Cannot parse param '{param_name}' from method name '{method_name}'. "
        f"Expected format: k_<k>_value_0_<v0>_delta_<delta>."
    )


def parse_name_thresholded_expected_error(method_name: str, param_name: str) -> str:
    """
    Parses a method name with format:
        thresholded_expected_error_{threshold}_alpha_{alpha}
    (notebook uses parts[1] for threshold, parts[3] for alpha based on split("_"))
    """
    parts = method_name.split("_")
    try:
        if param_name == "threshold":
            return parts[1]
        if param_name == "alpha":
            return parts[3]
    except IndexError:
        pass
    raise ValueError(
        f"Cannot parse param '{param_name}' from method name '{method_name}'. "
        f"Expected format: thresholded_expected_error_<threshold>_alpha_<alpha> or similar."
    )


# ---------------------------------------------------------------------------
# Section: Get step from results.csv
# ---------------------------------------------------------------------------

def get_step(df_trials: pd.DataFrame, params: dict) -> Optional[int]:
    """
    Finds the unique 'step' value in df_trials that matches all given params.
    Raises AssertionError if more than one match is found.
    """
    epsilon = 1e-6
    mask = None
    for name_param, value in params.items():
        if name_param not in df_trials.columns:
            raise KeyError(
                f"Column '{name_param}' not found in results.csv. "
                f"Available columns: {list(df_trials.columns)}"
            )
        condition = (df_trials[name_param].astype(float) - float(value)).abs() < epsilon
        mask = condition if mask is None else mask & condition

    selected = df_trials.loc[mask, "step"]
    if selected.shape[0] == 0:
        print(f"  [WARNING] No step found in results.csv for params {params}.")
        return None
    assert selected.shape[0] == 1, f"More than one step found for {params}: {selected.values}"
    return selected.values[0]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve configuration (Cell 2)
    # ------------------------------------------------------------------
    if args.configuration not in CONFIGURATIONS:
        print(
            f"[ERROR] Unknown configuration '{args.configuration}'. "
            f"Valid options: {list(CONFIGURATIONS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = CONFIGURATIONS[args.configuration]
    method: str = cfg["method"]
    network: str = cfg["network"]
    dataset: str = args.dataset
    partition: str = args.partition

    print(f"[INFO] Dataset        : {dataset}")
    print(f"[INFO] Partition      : {partition}")
    print(f"[INFO] Configuration  : {args.configuration}")
    print(f"[INFO] Network        : {network}")
    print(f"[INFO] Method         : {method}")

    # base_path is relative to the jupyter directory (same as the notebook)
    base_path = JUPYTER_DIR / ".." / "wkdir" / dataset / partition
    base_path = base_path.resolve()

    if not base_path.is_dir():
        print(
            f"[ERROR] Base path does not exist: '{base_path}'. "
            "Make sure the wkdir directory has been populated.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Collect method paths (Cell 2 continued)
    # ------------------------------------------------------------------
    backpas_network_method_paths = base_path / network / f"trust_region_{method}"

    if not backpas_network_method_paths.is_dir():
        print(
            f"[ERROR] Network/method directory does not exist: '{backpas_network_method_paths}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    methods_paths = [
        p
        for p in backpas_network_method_paths.iterdir()
        if p.is_dir() and not p.name.endswith("_log")
    ]

    if not methods_paths:
        print(
            f"[ERROR] No method sub-directories found inside '{backpas_network_method_paths}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Found {len(methods_paths)} method variant(s) inside '{backpas_network_method_paths.name}'.")

    baseline_path = base_path / "baseline"
    if not baseline_path.is_dir():
        print(
            f"[ERROR] Baseline directory does not exist: '{baseline_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build instance→log dictionary (Cell 2 continued)
    # ------------------------------------------------------------------
    try:
        instance_log_dict = construct_instance_log_dict(
            baseline_path=baseline_path,
            methods_paths=methods_paths,
        )
    except ValueError as exc:
        print(f"[ERROR] Failed to build instance log dictionary: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Check for missing logs (Cells 4–6)
    # ------------------------------------------------------------------
    missing_logs: List[Tuple] = []
    for instance_name, instance_log in instance_log_dict.items():
        for method_name, log_file_path in instance_log.items():
            if not log_file_path.exists():
                instance_path = (
                    log_file_path.parent.parent
                    / log_file_path.parent.name.replace("_log", "")
                    / log_file_path.name.replace(".log", "")
                )
                missing_logs.append((instance_name, method_name, log_file_path, instance_path))

    if missing_logs:
        df_missing = pd.DataFrame(
            missing_logs, columns=["instance", "method", "log_file_path", "instance_path"]
        )
        counts = df_missing.groupby("method")["log_file_path"].count().sort_values(ascending=False)
        print("\n[WARNING] Missing log files detected:")
        print(counts.to_string())

        # Interactive flow: ask for step number and create temp file list
        file_paths = list({row[3] for row in missing_logs})
        try:
            aux_step = input("\n[INPUT] Enter step number for the missing-instances file: ").strip()
        except EOFError:
            aux_step = ""

        if aux_step:
            output_filename = base_path / f"{args.configuration}_step_{aux_step}.txt"
            try:
                confirm = input(
                    f"[INPUT] The file '{output_filename}' will be created with "
                    f"{len(file_paths)} instances. Type 'ok' to confirm: "
                ).strip()
            except EOFError:
                confirm = ""

            if confirm == "ok":
                try:
                    create_temp_file_list(file_paths, output_filename=str(output_filename))
                    print(f"[INFO] Temp file list written to '{output_filename}'.")
                except Exception as exc:
                    print(f"[ERROR] Could not write temp file list: {exc}", file=sys.stderr)
                    sys.exit(1)
            else:
                print("[ERROR] File not created due to lack of confirmation.", file=sys.stderr)
                sys.exit(1)
        else:
            print("[WARNING] No step entered — skipping temp file creation.")
    else:
        print("[INFO] No missing logs detected. You can continue.")

    # ------------------------------------------------------------------
    # 5. Read logs and compute primal integral (Cell 8)
    # ------------------------------------------------------------------
    if dataset not in OBJECTIVE_MAP:
        print(
            f"[ERROR] Objective sense not defined for dataset '{dataset}'. "
            f"Known datasets: {list(OBJECTIVE_MAP.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    objective = OBJECTIVE_MAP[dataset]
    print(f"[INFO] Objective sense: {objective}")

    try:
        df_primal_integral, _ = get_all_logs_for_instance(
            instance_log_dict, objective=objective
        )
    except Exception as exc:
        print(f"[ERROR] Failed while reading logs: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Build computed_results (Cell 10)
    # ------------------------------------------------------------------
    try:
        if method == "fixed_three_ratios":
            df_val = df_primal_integral[
                ~df_primal_integral["method"].isin(["baseline", "conpas_paper"])
            ].copy()
            df_val["k_ratio"] = df_val["method"].apply(
                lambda x: parse_name_fixed_three_ratios(x, "k_ratio")
            )
            df_val["value_0_ratio"] = df_val["method"].apply(
                lambda x: parse_name_fixed_three_ratios(x, "value_0_ratio")
            )
            df_val["Delta_ratio"] = df_val["method"].apply(
                lambda x: parse_name_fixed_three_ratios(x, "Delta_ratio")
            )
            computed_results = (
                df_val.groupby(["k_ratio", "value_0_ratio", "Delta_ratio"])["primal_integral"]
                .mean()
                .reset_index()
                .rename(columns={"primal_integral": "objective_value"})
                .sort_values("objective_value")
            )

        elif method == "thresholded_expected_error":
            df_val = df_primal_integral[
                ~df_primal_integral["method"].isin(["baseline", "conpas_paper"])
            ].copy()
            df_val["threshold"] = df_val["method"].apply(
                lambda x: parse_name_thresholded_expected_error(x, "threshold")
            )
            df_val["alpha"] = df_val["method"].apply(
                lambda x: parse_name_thresholded_expected_error(x, "alpha")
            )
            computed_results = (
                df_val.groupby(["threshold", "alpha"])["primal_integral"]
                .mean()
                .reset_index()
                .rename(columns={"primal_integral": "objective_value"})
                .sort_values("objective_value")
            )

        else:
            raise ValueError(f"Method '{method}' not recognized.")

    except (ValueError, KeyError) as exc:
        print(f"[ERROR] Could not build computed_results: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 7. Check for empty methods (Cells 11–12)
    # ------------------------------------------------------------------
    empty_method_counter: Dict[str, int] = {}
    for instance_name in instance_log_dict:
        for method_name in instance_log_dict[instance_name]:
            if method_name == "baseline":
                continue
            empty_method_counter.setdefault(method_name, 0)
            if (
                instance_log_dict[instance_name][method_name]
                != instance_log_dict[instance_name]["baseline"]
            ):
                empty_method_counter[method_name] += 1

    empty_methods = [m for m, count in empty_method_counter.items() if count == 0]
    if empty_methods:
        print(
            f"[ERROR] The following methods have zero non-baseline log entries: {empty_methods}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[INFO] No empty methods detected.")

    # ------------------------------------------------------------------
    # 8. Preview top results (Cell 13)
    # ------------------------------------------------------------------
    print("\n[INFO] Top computed results (head):")
    print(computed_results.head().to_string(index=False))

    # ------------------------------------------------------------------
    # 9. Read results.csv and map steps (Cells 14–15)
    # ------------------------------------------------------------------
    results_path = backpas_network_method_paths / "results.csv"
    if not results_path.is_file():
        print(
            f"[ERROR] results.csv not found at '{results_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        df_trials = pd.read_csv(results_path)
    except Exception as exc:
        print(f"[ERROR] Could not read results.csv: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[INFO] Reading results.csv from '{results_path}'")
    incomplete = df_trials["objective_value"].isna().sum() if "objective_value" in df_trials.columns else "N/A"
    print(f"[INFO] Number of trials found: {len(df_trials)}  |  Incomplete trials: {incomplete}")

    # Build get_params depending on method
    if method == "fixed_three_ratios":
        def get_params(row: pd.Series) -> dict:
            return {
                "k_ratio": row["k_ratio"],
                "value_0_ratio": row["value_0_ratio"],
                "Delta_ratio": row["Delta_ratio"],
            }
    elif method == "thresholded_expected_error":
        def get_params(row: pd.Series) -> dict:
            return {
                "threshold": row["threshold"],
                "alpha": row["alpha"],
            }
    else:
        print(f"[ERROR] Method '{method}' not recognized for get_params.", file=sys.stderr)
        sys.exit(1)

    try:
        computed_results["step"] = computed_results.apply(
            lambda row: get_step(df_trials, get_params(row)), axis=1
        )
    except (KeyError, AssertionError) as exc:
        print(f"[ERROR] Failed to map steps from results.csv: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n[INFO] Computed results with steps (head):")
    print(computed_results.head().to_string(index=False))

    # ------------------------------------------------------------------
    # 10. Plot (Cell 16)
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            matplotlib.use("Agg")  # non-interactive backend, safe for headless
            import seaborn as sns
            fig, ax = plt.subplots()
            sns.boxplot(data=computed_results, x="step", y="objective_value", ax=ax)
            baseline_mean = df_primal_integral[
                df_primal_integral["method"] == "baseline"
            ]["primal_integral"].mean()
            ax.axhline(y=baseline_mean, color="r", linestyle="--", label="Gurobi")
            conpas_rows = df_primal_integral[df_primal_integral["method"] == "conpas_paper"]
            if len(conpas_rows) != 0:
                ax.axhline(
                    y=conpas_rows["primal_integral"].mean(),
                    color="g",
                    linestyle="--",
                    label="ConPaS paper",
                )
            ax.set_yscale("log")
            ax.legend()
            fig.tight_layout()

            plot_path = args.plot_output
            if plot_path is None:
                plot_path = str(
                    backpas_network_method_paths / f"boxplot_{args.configuration}.png"
                )
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Box-plot saved to '{plot_path}'.")
        except ImportError:
            print("[WARNING] seaborn not installed — skipping plot.")
        except Exception as exc:
            print(f"[WARNING] Could not render plot: {exc}")

    # ------------------------------------------------------------------
    # 11. Save results (Cell 17)
    # ------------------------------------------------------------------
    if not args.no_save:
        conpas_rows = df_primal_integral[df_primal_integral["method"] == "conpas_paper"]
        if len(conpas_rows) != 0:
            print(
                "[ERROR] 'conpas_paper' method is not empty. "
                "Please check the results before saving.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            computed_results.to_csv(results_path, index=False)
            print(f"\n[INFO] Results saved to '{results_path}'.")
        except Exception as exc:
            print(f"[ERROR] Could not save results.csv: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n[INFO] --no-save flag set: results.csv was NOT overwritten.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
