#!/usr/bin/env python3
"""BD-rate of one preset against another from `lab gen --verify` files of
the same clips and sizes (matched by clip, size, fps, chunk and prefilter
rung), e.g. how much smaller x264 slower is than medium at equal PSNR.

Usage: preset_gain.py REF_DIR REF_PRESET TEST_DIR TEST_PRESET
  e.g. preset_gain.py target/lab3 medium target/lab6 slower
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svt_survey import bd_rate  # noqa: E402


def curves(path):
    d = json.load(open(path))
    out = collections.defaultdict(list)
    for r in d["results"]:
        if r["true_mse"] > 0:
            rung = d["rungs"][r["rung"]]
            out[(r["chunk"], rung["w"], rung["h"])].append((r["bytes"], 10 * math.log10(65025 / r["true_mse"])))
    return out


def main():
    ref_dir, ref_p, test_dir, test_p = sys.argv[1:5]
    per_cfg = []
    allv = []
    for tp in sorted(glob.glob(f"{test_dir}/{test_p}_*.json")):
        cfg = os.path.basename(tp)[len(test_p) + 1:]
        rp = f"{ref_dir}/{ref_p}_{cfg}"
        if not os.path.exists(rp):
            continue
        a, b = curves(rp), curves(tp)
        vals = []
        for k in sorted(set(a) & set(b)):
            if len(a[k]) >= 4 and len(b[k]) >= 4:
                v = bd_rate(sorted(a[k]), sorted(b[k]))
                if v is not None and abs(v) < 80:
                    vals.append(v)
        if vals:
            per_cfg.append((cfg, np.mean(vals), len(vals)))
            allv += vals
    for cfg, v, n in per_cfg:
        print(f"  {cfg:28} {v:+6.1f}%  ({n} chunk x rung curves)")
    print(f"{test_p} vs {ref_p}: BD-rate {np.mean(allv):+.1f}% (median {np.median(allv):+.1f}%, {len(allv)} curves)")


if __name__ == "__main__":
    main()
