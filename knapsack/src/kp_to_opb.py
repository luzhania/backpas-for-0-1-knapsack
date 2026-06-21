#!/usr/bin/env python3
"""
knapsack_to_opb.py

Convert a 0-1 knapsack instance in Pisinger .in format
to an OPB (Pseudo-Boolean Optimization) model file.

Example .in:
4
1 10 5
2 7 3
3 12 6
4 8 4
10

This will create an OPB with:
- A negated objective (minimization) suitable for OPB
- One constraint for capacity
"""

import sys
import os


def parse_pisinger_in(file_path):
    """
    Read a .in file and return (items, capacity)
    items: list of (index, profit, weight)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) < 2:
        raise ValueError("Not enough lines in input file.")

    # Number of items
    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError("First line must be an integer (number of items).")

    if len(lines) != n + 2:
        raise ValueError(
            f"Expected {n} item lines + capacity line, found {len(lines)-1} lines."
        )

    items = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        if len(parts) != 3:
            raise ValueError(f"Invalid item line: {lines[i]}")
        idx = int(parts[0])
        profit = float(parts[1])
        weight = float(parts[2])
        items.append((idx, profit, weight))

    # Last line is capacity
    try:
        capacity = float(lines[n + 1])
    except ValueError:
        raise ValueError("Final line must be numeric (capacity).")

    return items, capacity


def generate_opb(items, capacity):
    """
    Create the OPB model text from a knapsack instance.
    """
    n_vars = len(items)
    # Build negated objective: min: -profit1 x1 -profit2 x2 ...
    obj_terms = []
    for idx, profit, _ in items:
        # negated profit for minimization
        coeff = -int(profit)
        obj_terms.append(f"{coeff} x{idx}")

    obj_line = "min: " + " ".join(obj_terms) + " ;"

    # Build the capacity constraint
    # In OPB style, we convert sum(weights xi) <= capacity
    # to sum(-weights xi) >= -capacity
    lhs_terms = []
    for idx, _, weight in items:
        coeff = -int(weight)
        lhs_terms.append(f"{coeff} x{idx}")

    cons_line = " ".join(lhs_terms) + f" >= {-int(capacity)} ;"

    lines = []
    lines.append(f"* #variable= {n_vars} #constraint= 1")
    lines.append(obj_line)
    lines.append(cons_line)

    return "\n".join(lines)


def write_opb_file(text, output_path):
    with open(output_path, "w") as f:
        f.write(text)


def main():
    if len(sys.argv) != 3:
        print("Usage: python knapsack_to_opb.py input.in output.opb")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        items, cap = parse_pisinger_in(input_file)
        opb_text = generate_opb(items, cap)
        write_opb_file(opb_text, output_file)
        print(f"OPB model written to {output_file}")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()