#!/usr/bin/env python3
"""Checks that the Rust model (crates/core/src/model.rs, as ported by
port_model.py) predicts what fit_model.py fitted: runs `smallifier-lab
parity` on a lab file and compares with the Python predictions row by row.

Usage: parity.py LAB_BIN fit.json lab.json PRESET
"""
import json
import math
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_model as fm  # noqa: E402


def main():
    lab_bin, fit_path, lab_path, preset = sys.argv[1:5]
    fit = json.load(open(fit_path))
    fm.QD = fit.get("qd", 6.0)
    rows = fm.load([lab_path])
    rust = subprocess.run([lab_bin, "parity", lab_path, preset], capture_output=True, text=True, check=True).stdout.split("\n")
    worst_b = worst_p = 0.0
    n = 0
    for r, line in zip(rows, rust):
        _, _, _, rb, rp = line.split()
        pb = fm.predict(fit["rate"], r)
        pp = 10 * math.log10(65025 / fm.d_predict(fit["rate"], fit["dist"], r))
        if pb < 64:  # Rust floors tiny predictions at 64 bytes
            continue
        worst_b = max(worst_b, abs(float(rb) / pb - 1))
        worst_p = max(worst_p, abs(float(rp) - pp))
        n += 1
    print(f"{preset}: {n} rows, worst size difference {worst_b * 100:.4f}%, worst PSNR difference {worst_p:.4f} dB")
    sys.exit(0 if worst_b < 1e-3 and worst_p < 1e-3 else 1)


if __name__ == "__main__":
    main()
