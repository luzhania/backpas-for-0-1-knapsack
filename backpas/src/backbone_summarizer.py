#!/usr/bin/env python3
"""
backbone_summarizer.py

This script recursively traverses the 'backpas/dataset' directory to build 
a comprehensive CSV summarizing backbone extraction results.

Expected Directory Structure:
backpas/dataset/<subset_name>/
    ├── backbone/
    │   └── <instance_name>.opb.backbone
    └── backbone_extraction_log/
        └── <instance_name>.opb.log

Columns generated:
type, n, phi, backbone_literals_count, positive_literals_count, negative_literals_count, extraction_time
"""

import os
import re
import csv
from pathlib import Path

# Base repository directory (two levels up from backpas/src/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKPAS_DIR = REPO_ROOT / "backpas"

# Folder and output configuration
DATASET_DIR = BACKPAS_DIR / "dataset"
OUTPUT_CSV = DATASET_DIR / "backbone_summary.csv"

# Expresiones regulares
# Ejemplo: train_instance_1620_type14_n1000_phi69.opb.backbone
# Captures: group 1 (type), group 2 (n), group 3 (phi)
REGEX_FILENAME = re.compile(r"^.*_type(\d+)_n(\d+)_phi(\d+)\.opb\.backbone$")

# Ejemplo: TOTAL RUNTIME: 3440963 microseconds
REGEX_TIME = re.compile(r"TOTAL RUNTIME:\s+(\d+)\s+microseconds", re.IGNORECASE)


def parse_backbone_file(filepath: Path):
    """
    Counts the total number of literals, positives, and negatives in a .backbone file.
    """
    total = 0
    positives = 0
    negatives = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Backbone variables start with "b x..." or "b -x..."
                if line.startswith("b "):
                    var_str = line.split()[1] # E.g.: "x869" or "-x869"
                    total += 1
                    if var_str.startswith("-"):
                        negatives += 1
                    else:
                        positives += 1
    except Exception as e:
        print(f"  [ERROR] Processing backbone file {filepath.name}: {e}")
        return None, None, None

    return total, positives, negatives


def parse_log_file(filepath: Path):
    """
    Looks for the extraction time in the log corresponding to the instance.
    Looks for the line "TOTAL RUNTIME: <value> microseconds" at the end of the file.
    Returns the time as a string or None if not found/failed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read the file backwards or simply search all lines 
            # (since the data is at the end, we will read the last lines)
            lines = f.readlines()
            for line in reversed(lines):
                match = REGEX_TIME.search(line)
                if match:
                    return match.group(1)
    except FileNotFoundError:
        print(f"  [WARNING] Log not found at: {filepath}")
        return None
    except Exception as e:
        print(f"  [ERROR] Processing extraction log {filepath.name}: {e}")
        return None

    return None


def main():
    root_path = Path(DATASET_DIR)
    
    if not root_path.exists():
        print(f"[ERROR] The directory {DATASET_DIR} does not exist.")
        return

    csv_rows = []
    
    # Escribir la cabecera en nuestro arreglo en memoria
    header = [
        "type", 
        "n", 
        "phi", 
        "backbone_literals_count", 
        "positive_literals_count", 
        "negative_literals_count", 
        "backbone_percentage",
        "positive_backbone_percentage",
        "negative_backbone_percentage",
        "extraction_time"
    ]

    print(f"=== Starting backbone collector in {DATASET_DIR} ===")
    
    processed_count = 0
    failed_count = 0

    # Use rglob to dynamically find all .opb.backbone files
    for backbone_file in root_path.rglob("*.opb.backbone"):
        filename = backbone_file.name
        
        # 1. Get "type", "n", and "phi"
        match = REGEX_FILENAME.search(filename)
        if not match:
            print(f"  [COMPATIBILITY] Name {filename} does not match expected format. Skipping.")
            failed_count += 1
            continue
            
        tipo = match.group(1)
        n = match.group(2)
        phi = match.group(3)

        # 2. Contar literales
        total_back, pos_back, neg_back = parse_backbone_file(backbone_file)
        if total_back is None:
            failed_count += 1
            continue

        # 3. Get extraction time.
        # By structure we know the log will be in "../backbone_extraction_log/file.opb.log" relative to the "backbone" folder
        parent_dir = backbone_file.parent.parent # Go up from /backbone to /<subdirectory>
        expected_log_name = filename.replace(".opb.backbone", ".opb.log")
        log_file = parent_dir / "backbone_extraction_log" / expected_log_name

        tiempo = parse_log_file(log_file)
        if tiempo is None:
            # If there is no log with time, we mark as "NA" instead of failing the entire instance, 
            # assuming that the backbone was indeed generated
            tiempo = "NA"

        n_val = int(n)
        backbone_percentage = total_back / n_val if n_val > 0 else 0
        positive_backbone_percentage = pos_back / total_back if total_back > 0 else 0
        negative_backbone_percentage = neg_back / total_back if total_back > 0 else 0

        # Register successfully parsed row
        csv_rows.append([
            tipo, n, phi,
            total_back, pos_back, neg_back,
            backbone_percentage, positive_backbone_percentage, negative_backbone_percentage,
            tiempo
        ])
        processed_count += 1

    # Guardar en CSV solo si hemos recolectado algo
    if csv_rows:
        try:
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(csv_rows)
            print("\n=== CSV Generation Summary ===")
            print(f"- File successfully saved at: {OUTPUT_CSV}")
            print(f"- Instances processed: {processed_count}")
            print(f"- Instances skipped/failed: {failed_count}")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] There was a problem trying to save the CSV file: {e}")
    else:
        print("\nNo valid backbone records were found. The CSV file has not been generated.")


if __name__ == "__main__":
    main()
