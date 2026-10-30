#!/usr/bin/env python3
"""Measure how VMAF relates to PSNR on our kind of content, so the planner
can maximise (predicted) mean VMAF like smallifier.py did.

Encodes each clip at 720p30 with x264 medium over a grid of prefilter rungs
and CRFs (ffmpeg's libx264 and lanczos, close enough for this purpose),
then measures per-frame luma PSNR and VMAF (vmaf_v0.6.1neg, as in
smallifier.py) against the 720p30 source, pooled over 2 s windows.

  vmaf_map.py OUT.json CLIP...      (then: vmaf_map.py --fit OUT.json)
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

W, H, FPS = 1280, 720, 30
RUNGS = [0.5, 0.6, 0.75, 1.0]
CRFS = [20, 26, 32, 38, 44]
WINDOW = 60


def ref_filter():
    return f"fps={FPS},scale={W}:{H}:flags=lanczos,format=yuv420p"


def job(clip, rung, crf, tmp):
    rw, rh = round(W * rung / 2) * 2, round(H * rung / 2) * 2
    pre = ref_filter() + (f",scale={rw}:{rh}:flags=lanczos,scale={W}:{H}:flags=lanczos" if rung < 1 else "")
    out = os.path.join(tmp, f"{os.path.basename(clip)}_{rung}_{crf}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", pre, "-an", "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-x264-params", "aq-mode=3:rc-lookahead=20", "-threads", "2", out], check=True)
    log = out + ".json"
    psnr_log = out + ".psnr"
    subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-i", clip, "-lavfi",
                    f"[1:v]{ref_filter()},split[r1][r2];[0:v]format=yuv420p,split[d1][d2];"
                    f"[d1][r1]libvmaf=model=version=vmaf_v0.6.1neg:n_threads=4:log_fmt=json:log_path={log};"
                    f"[d2][r2]psnr=stats_file={psnr_log}", "-f", "null", "-"], check=True)
    vm = [f["metrics"]["vmaf"] for f in json.load(open(log))["frames"]]
    ps = []
    for line in open(psnr_log):
        for tok in line.split():
            if tok.startswith("mse_y:"):
                ps.append(float(tok[6:]))
    n = min(len(vm), len(ps))
    rows = []
    for a in range(0, n - WINDOW // 2, WINDOW):
        b = min(n, a + WINDOW)
        mse = sum(ps[a:b]) / (b - a)
        rows.append({"clip": os.path.basename(clip), "rung": rung, "crf": crf, "start": a,
                     "psnr": 10 * math.log10(65025 / max(mse, 1e-6)), "vmaf": sum(vm[a:b]) / (b - a)})
    os.unlink(out)
    return rows


def fit(path):
    import numpy as np
    from scipy.optimize import least_squares
    rows = json.load(open(path))
    p = np.array([r["psnr"] for r in rows])
    v = np.array([r["vmaf"] for r in rows])
    f = lambda x, p: 100.0 / (1.0 + np.exp(-(p - x[0]) / x[1]))
    r = least_squares(lambda x: f(x, p) - v, [30.0, 3.0])
    err = f(r.x, p) - v
    print(f"VMAF ~= 100 / (1 + exp(-(PSNR - {r.x[0]:.3f}) / {r.x[1]:.3f}))   "
          f"n={len(rows)} rmse {np.sqrt(np.mean(err**2)):.2f} VMAF, median |err| {np.median(abs(err)):.2f}")
    for lo in range(20, 50, 4):
        sel = (p >= lo) & (p < lo + 4)
        if sel.any():
            print(f"  PSNR {lo}-{lo+4}: mean VMAF {v[sel].mean():5.1f} (fit {f(r.x, p[sel]).mean():5.1f}), n={sel.sum()}")


def main():
    if sys.argv[1] == "--fit":
        fit(sys.argv[2])
        return
    out, clips = sys.argv[1], sys.argv[2:]
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(48) as ex:
        futs = [ex.submit(job, c, r, q, tmp) for c in clips for r in RUNGS for q in CRFS]
        rows = [row for f in futs for row in f.result()]
    json.dump(rows, open(out, "w"))
    print(f"wrote {len(rows)} windows to {out}")
    fit(out)


if __name__ == "__main__":
    main()
