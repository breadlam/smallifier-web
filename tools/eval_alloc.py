#!/usr/bin/env python3
"""How much quality does planning from predictions cost, versus planning
from real encodes (what smallifier.py does with ~20 encodes per chunk)?

For each lab clip and a few budgets:
  oracle - greedy equal-slope allocation over *measured* (bytes, MSE)
           points, scene by scene, rung x CRF (as smallifier.py does).
  model  - the same allocation over *predicted* points; then, as the app
           does with two-pass, each scene gets its planned share of the
           budget at the planned rung, and its quality is read off the
           measured bytes->MSE curve of that rung.
Prints the duration-weighted mean of the VMAF proxy the planner maximises.

Usage: eval_alloc.py PARAMS_JSON files...   (PARAMS_JSON from fit_model.py --save)
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import fit_model as fm  # noqa: E402

def vmaf_proxy(p, factor=1.0):
    return 96.0 / (1.0 + math.exp(-(p - 26.883) / 3.798))


def psnr(mse):
    return 10 * math.log10(65025 / max(mse, 1e-6))


def hull(points):
    """points: (bytes, value, payload). Upper concave hull, bytes ascending."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    par = []
    for p in pts:
        if not par or p[1] > par[-1][1]:
            par.append(p)
    h = []
    for p in par:
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0:
                h.pop()
            else:
                break
        h.append(p)
    return h


def allocate(hulls, budget):
    choice = [0] * len(hulls)
    total = sum(h[0][0] for h in hulls)
    while True:
        best = None
        for s, h in enumerate(hulls):
            k = choice[s]
            if k + 1 >= len(h):
                continue
            db = h[k + 1][0] - h[k][0]
            if db <= 0 or total + db > budget:
                continue
            slope = (h[k + 1][1] - h[k][1]) / db
            if best is None or slope > best[1]:
                best = (s, slope)
        if best is None:
            break
        s = best[0]
        total += hulls[s][choice[s] + 1][0] - hulls[s][choice[s]][0]
        choice[s] += 1
    return choice, total


def measured_curve(rows, chunk_ids, rung):
    """Scene-level measured (bytes, frames-weighted SE) per CRF at a rung."""
    by_crf = {}
    for r in rows:
        if r["chunk_id"] in chunk_ids and r["rung"] == rung:
            b, se = by_crf.get(r["crf"], (0.0, 0.0))
            by_crf[r["crf"]] = (b + r["bytes"], se + r["true_mse"] * r["c"]["frames"])
    return sorted(by_crf.values())


def quality_at(curve, frames, bytes_, factor=1.0):
    """Interpolate PSNR at a byte count on a measured curve (log-bytes, log-MSE)."""
    bs = np.log([c[0] for c in curve])
    ms = np.log([c[1] / frames for c in curve])
    x = math.log(bytes_)
    if x <= bs[0]:
        m = ms[0] + (x - bs[0]) * (ms[1] - ms[0]) / (bs[1] - bs[0])
    elif x >= bs[-1]:
        m = ms[-1] + (x - bs[-1]) * (ms[-1] - ms[-2]) / (bs[-1] - bs[-2])
    else:
        m = np.interp(x, bs, ms)
    return vmaf_proxy(psnr(math.exp(m)), factor)


def main():
    params = json.load(open(sys.argv[1]))
    rate, dist = params["rate"], params["dist"]
    fm.QD = params.get("qd", 6.0)
    for path in sys.argv[2:]:
        rows = fm.load([path])
        d = json.load(open(path))
        for i, r in enumerate(rows):
            r["chunk_id"] = d["results"][i]["chunk"]
        chunks = d["chunks"]
        scenes = sorted({c["scene"] for c in chunks})
        nr = len(d["rungs"])
        crfs = sorted({r["crf"] for r in rows})
        # Measured and predicted points per scene.
        meas_h, pred_h, scene_frames, scene_chunks = [], [], [], []
        for s in scenes:
            ids = {i for i, c in enumerate(chunks) if c["scene"] == s}
            frames = sum(chunks[i]["end"] - chunks[i]["start"] for i in ids)
            scene_frames.append(frames)
            scene_chunks.append(ids)
            mp, pp = [], []
            for rg in range(nr):
                for crf in crfs:
                    sel = [r for r in rows if r["chunk_id"] in ids and r["rung"] == rg and r["crf"] == crf]
                    b = sum(r["bytes"] for r in sel)
                    se = sum(r["true_mse"] * r["c"]["frames"] for r in sel)
                    fac = d["rungs"][rg]["w"] / d["w"]
                    mp.append((b, vmaf_proxy(psnr(se / frames), fac) * frames, (rg, crf)))
                    pb = sum(fm.predict(rate, r) for r in sel)
                    pse = sum(fm.d_predict(rate, dist, r) * r["c"]["frames"] for r in sel)
                    pp.append((pb, vmaf_proxy(psnr(pse / frames), fac) * frames, (rg, crf)))
            meas_h.append(hull(mp))
            pred_h.append(hull(pp))
        total_frames = sum(scene_frames)
        name = path.split("/")[-1].replace(".json", "")
        # Budgets: what uniform CRFs at full res would cost, from generous to
        # tight in this file's CRF grid.
        for ref_crf in sorted({crfs[1], crfs[len(crfs) // 2], crfs[-3], crfs[-2]}):
            budget = sum(r["bytes"] for r in rows if r["rung"] == nr - 1 and r["crf"] == ref_crf)
            oc, _ = allocate(meas_h, budget)
            q_oracle = sum(meas_h[s][oc[s]][1] for s in range(len(scenes))) / total_frames
            pc, ptotal = allocate(pred_h, budget)
            # Two-pass gives each scene its planned share of the real budget.
            q_model = 0.0
            for s in range(len(scenes)):
                pb, _, (rg, _) = pred_h[s][pc[s]]
                share = max(budget * pb / max(ptotal, 1.0), 1.0)
                q_model += quality_at(measured_curve(rows, scene_chunks[s], rg), scene_frames[s], share,
                                      d["rungs"][rg]["w"] / d["w"]) * scene_frames[s]
            q_model /= total_frames
            # Baseline: one uniform CRF at full resolution (single 2-pass encode).
            q_flat = quality_at(measured_curve(rows, set(range(len(chunks))), nr - 1), total_frames, budget)
            print(f"{name:28s} budget@crf{ref_crf}: oracle {q_oracle:6.2f}  model {q_model:6.2f} "
                  f"(gap {q_oracle - q_model:+.2f})  plain 2-pass {q_flat:6.2f}  (mean predicted VMAF)")


if __name__ == "__main__":
    main()
