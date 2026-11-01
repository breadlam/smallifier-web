#!/usr/bin/env python3
"""Summarise `lab svtsurvey` output: per SVT-AV1 preset and bit depth, the
BD-rate against a reference (default preset 8, 8-bit), over the whole CRF
range and at the low-rate end, and native encode time.

Usage: svt_survey.py [--ref P:D] files.jsonl...
"""
import collections
import json
import math
import sys

import numpy as np


def bd_rate(ref, test, lo=None, hi=None):
    """Mean log-rate difference (%) at equal PSNR, from (bytes, psnr) points
    (Bjontegaard, cubic fit of log-rate vs PSNR)."""
    r1 = np.log([b for b, _ in ref]); p1 = np.array([p for _, p in ref])
    r2 = np.log([b for b, _ in test]); p2 = np.array([p for _, p in test])
    a = max(p1.min(), p2.min()) if lo is None else lo
    b = min(p1.max(), p2.max()) if hi is None else hi
    if b <= a:
        return None
    c1 = np.polyfit(p1, r1, 3); c2 = np.polyfit(p2, r2, 3)
    i1 = np.polyval(np.polyint(c1), b) - np.polyval(np.polyint(c1), a)
    i2 = np.polyval(np.polyint(c2), b) - np.polyval(np.polyint(c2), a)
    return (math.exp((i2 - i1) / (b - a)) - 1) * 100


def main():
    args = sys.argv[1:]
    ref = (8, 8)
    if args and args[0] == "--ref":
        p, d = args[1].split(":")
        ref = (int(p), int(d))
        args = args[2:]
    rows = [json.loads(l) for f in args for l in open(f) if l.strip()]
    curves = collections.defaultdict(list)  # (clip, chunk, preset, depth) -> [(bytes, psnr)]
    secs = collections.defaultdict(float)
    frames = collections.defaultdict(int)
    sse_err = []
    for r in rows:
        psnr = 10 * math.log10(65025 / r["mse"])
        curves[(r["clip"], r["chunk"], r["preset"], r["depth"])].append((r["bytes"], psnr))
        secs[(r["preset"], r["depth"])] += r["secs"]
        frames[(r["preset"], r["depth"])] += r["frames"]
        sse_err.append(10 * math.log10(r["svt_mse"] / r["mse"]))
    print(f"SVT-AV1's own PSNR vs decoded: mean {np.mean(sse_err):+.3f} dB, max |{np.max(np.abs(sse_err)):.3f}| dB")
    keys = sorted({(p, d) for (_, _, p, d) in curves})
    chunks = sorted({(c, k) for (c, k, _, _) in curves})
    ref_fps = frames[ref] / secs[ref]
    print(f"{'preset':>6} {'bits':>4} {'BD-rate':>8} {'low end':>8} {'time x':>7}  (vs preset {ref[0]}, {ref[1]}-bit; "
          f"low end = PSNR below the reference's CRF-46 point)")
    for k in keys:
        all_, low = [], []
        for ch in chunks:
            a = sorted(curves.get((*ch, *ref), []))
            b = sorted(curves.get((*ch, *k), []))
            if len(a) < 4 or len(b) < 4:
                continue
            v = bd_rate(a, b)
            if v is not None:
                all_.append(v)
            # Low-rate end: the 3 lowest-rate reference points.
            lo_hi = a[2][1]
            v = bd_rate(a, b, lo=max(a[0][1], b[0][1]), hi=lo_hi)
            if v is not None:
                low.append(v)
        t = (frames[k] / secs[k]) if secs[k] else float("nan")
        print(f"{k[0]:>6} {k[1]:>4} {np.mean(all_):+7.1f}% {np.mean(low):+7.1f}% {ref_fps / t:6.2f}x"
              f"   ({len(all_)} chunks)")


if __name__ == "__main__":
    main()
