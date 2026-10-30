#!/usr/bin/env python3
"""Summarise the lab's per-chunk rate-control experiments (1-pass ABR and
2-pass) against CRF encodes of the same size."""
import json, math, statistics as st, sys

for f in sys.argv[1:]:
    d = json.load(open(f))
    for label, bkey, mkey in [("1-pass ABR", "bytes", "true_mse"), ("2-pass", "bytes_2pass", "true_mse_2pass")]:
        errs, dps, short = [], [], []
        for a in d["abr"]:
            c = d["chunks"][a["chunk"]]
            e = (a[bkey] / a["target"] - 1) * 100
            errs.append(e)
            dps.append(10 * math.log10(a["crf_true_mse"] / a[mkey]))
            if c["end"] - c["start"] < 120:
                short.append(abs(e))
        ae = sorted(abs(e) for e in errs)
        print(f"{f.split('/')[-1]:22s} {label:10s} size err median {st.median(ae):4.1f}%  p90 {ae[int(.9*len(ae))]:5.1f}%  "
              f"max {ae[-1]:5.1f}%  mean {st.mean(errs):+5.1f}%  dPSNR vs CRF {st.mean(dps):+.2f} dB"
              + (f"  | <4s chunks median {st.median(short):.1f}%" if short else ""))
    p1 = sum(a["t_pass1"] for a in d["abr"]); p2 = sum(a["t_pass2"] for a in d["abr"])
    print(f"{'':22s} first pass costs {p1/p2*100:.0f}% of the second pass")
