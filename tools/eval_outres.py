#!/usr/bin/env python3
"""Which output resolution should a budget use? Compares oracle (measured)
allocations restricted to one output size, all scored against the same
reference (lower sizes were decoded and upscaled by the lab's --ref).

  eval_outres.py REF.json LOWER_ref.json [LOWER2_ref.json ...]
Prints, per budget, the best quality reachable at each output size and the
bits per pixel (at the reference size) of that budget.
"""
import json
import math
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from eval_alloc import allocate, hull, psnr  # noqa: E402


def options(path):
    d = json.load(open(path))
    chunks = d["chunks"]
    scenes = sorted({c["scene"] for c in chunks})
    out = {}
    for s in scenes:
        ids = [i for i, c in enumerate(chunks) if c["scene"] == s]
        frames = sum(chunks[i]["end"] - chunks[i]["start"] for i in ids)
        pts = []
        for rg in range(len(d["rungs"])):
            for crf in sorted({r["crf"] for r in d["results"]}):
                rs = [r for r in d["results"] if r["chunk"] in ids and r["rung"] == rg and r["crf"] == crf]
                b = sum(r["bytes"] for r in rs)
                se = sum(r["true_mse"] * (chunks[r["chunk"]]["end"] - chunks[r["chunk"]]["start"]) for r in rs)
                pts.append((b, psnr(se / frames) * frames, (rg, crf)))
        out[(chunks[ids[0]]["start"], chunks[ids[-1]]["end"])] = (hull(pts), frames)
    return d, out


def main():
    ref_d, ref = options(sys.argv[1])
    others = [options(p) for p in sys.argv[2:]]
    keys = sorted(ref)
    for d, o in others:
        if sorted(o) != keys:
            print(f"scene boundaries differ for {d['w']}x{d['h']}: {sorted(o)} vs {keys}")
            return
    total_frames = sum(ref[k][1] for k in keys)
    fps = ref_d["fps"]
    secs = total_frames / fps
    print(f"{ref_d['clip']} reference {ref_d['w']}x{ref_d['h']}")
    for mbps in (0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0):
        budget = mbps * 1e6 / 8 * secs
        row = []
        for d, o in [(ref_d, ref)] + others:
            hs = [o[k][0] for k in keys]
            choice, _ = allocate(hs, budget)
            q = sum(hs[i][choice[i]][1] for i in range(len(keys))) / total_frames
            row.append(f"{d['w']}x{d['h']}: {q:5.2f}")
        bpp = mbps * 1e6 / (ref_d["w"] * ref_d["h"] * fps)
        print(f"  {mbps:4.1f} Mbps (bpp@ref {bpp:.3f}): " + "  ".join(row))


if __name__ == "__main__":
    main()
