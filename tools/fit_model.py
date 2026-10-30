#!/usr/bin/env python3
"""Fit the size/quality model to lab ground truth (target/lab/*.json).

The model works from the analysis histograms only (no encodes):

  Qstep(q)     = 0.625 * 2^(q/6)                    H.264 quantizer step
  x            = |coef| * gain_rung(pos) / Qstep(q)  normalised level
  nz(q)        = #coefs with x >= tau
  lv(q)        = sum over those of log2(x / tau)
  bits_frame   = a * nz + b * lv + c * mbs            (separately for I and P)
  bytes(chunk) = [bits_I(crf + dI) + bits_P(crf + dP)] / 8

Usage: fit_model.py [--holdout CLIP] files...
Prints fitted constants (to paste into crates/core/src/model.rs) and the
prediction error, overall and on any held-out clip.
"""
import json
import math
import sys

import numpy as np
from scipy.optimize import least_squares

NBINS = 52


def bin_mag():
    i = np.arange(NBINS, dtype=np.float64)
    m = 2.0 ** (-1.0 + (i - 0.5) / 4.0)
    m[0] = 0.25
    return m


MAG = bin_mag()


def load(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        gains = []
        factors = []
        for r in d["rungs"]:
            factors.append(r["w"] / d["w"])
            gx = np.array(r["gx"])
            gy = np.array(r["gy"])
            gx[0] = gy[0] = 1.0
            gains.append(np.outer(gy, gx).reshape(64))  # pos = v*8+u
        chunks = []
        for c in d["chunks"]:
            chunks.append({
                "I": np.array(c["i_hist"], dtype=np.float64).reshape(64, NBINS),
                "P": np.array(c["p_hist"], dtype=np.float64).reshape(64, NBINS),
                "P2": np.array(c["p2_hist"], dtype=np.float64).reshape(64, NBINS),
                "P4": np.array(c["p4_hist"], dtype=np.float64).reshape(64, NBINS),
                "pf": c["p_frames"], "p2f": c["p2_frames"], "p4f": c["p4_frames"],
                "src": np.array(c["src_energy"]),
                "frames": c["end"] - c["start"],
                "mbs_frame": c["i_pixels"] / 256.0,
                "pixels": c["i_pixels"],
                "intra_frac": c["intra_mbs"] / max(1, c["mbs"]),
                "mv_cost": c["mv_cost"],
            })
        name = d["clip"].split("/")[-1].split(".")[0] + f"_{d['h']}p{d['fps']}"
        for r in d["results"]:
            rows.append({
                "clip": name, "c": chunks[r["chunk"]], "gain": gains[r["rung"]],
                "rung": r["rung"], "gain_factor": factors[r["rung"]], "crf": r["crf"], "bytes": r["bytes"],
                "true_mse": r["true_mse"], "scale_mse": r["scale_mse"],
                "fps": d["fps"], "h": d["h"],
            })
    return rows


def qstep(q):
    return 0.625 * 2.0 ** (q / 6.0)


FREQ = np.array([(k % 8) + (k // 8) for k in range(64)], dtype=np.float64) / 14.0


def feats(hist, gain, q, tau, tau_hf=0.0):
    """nz and lv for a (64, NBINS) histogram at quantizer q through gain.
    tau_hf raises the survival threshold with frequency (trellis/psy-rd
    zero high-frequency coefficients more readily than a flat deadzone)."""
    x = (MAG[None, :] * gain[:, None]) / qstep(q)
    t = (tau * (1.0 + tau_hf * FREQ))[:, None]
    on = x >= t
    nz = (hist * on).sum()
    lv = (hist * on * np.log2(np.maximum(x, 1e-9) / t)).sum()
    return nz, lv


NAMES = ["aI", "bI", "cI", "aP", "bP", "cP", "dI", "dP", "tau", "tau_hf", "mvP"]
X0 = [3.0, 1.0, 0.5, 3.0, 1.0, 0.05, -3.0, 2.0, 0.7, 0.3, 0.5]
LO = [0, 0, 0, 0, 0, 0, -15, -10, 0.3, -0.5, 0]
HI = [50, 50, 50, 50, 50, 50, 10, 15, 1.5, 5.0, 20]


def predict(params, r, which="P"):
    aI, bI, cI, aP, bP, cP, dI, dP, tau, tau_hf, mvP = params
    c = r["c"]
    nzI, lvI = feats(c["I"], r["gain"], r["crf"] + dI, tau, tau_hf)
    h = c[which]
    scale = c["pf"] / max(1, c[{"P": "pf", "P2": "p2f", "P4": "p4f"}[which]])
    nzP, lvP = feats(h, r["gain"], r["crf"] + dP, tau, tau_hf)
    # MV side info shrinks as QP grows (x264 weighs MV cost by lambda).
    mv = mvP * c["mv_cost"] * qstep(30.0) / qstep(r["crf"])
    bits = (aI * nzI + bI * lvI + cI * c["mbs_frame"] + scale * (aP * nzP + bP * lvP)
            + cP * c["mbs_frame"] * c["pf"] + mv)
    return max(bits / 8.0, 1.0)


def residuals(params, rows, which="P"):
    return np.array([np.log(predict(params, r, which) / r["bytes"]) for r in rows])


def report(label, params, rows, which="P"):
    e = residuals(params, rows, which)
    pct = (np.exp(np.abs(e)) - 1) * 100
    print(f"{label:28s} n={len(rows):4d}  median |err| {np.median(pct):5.1f}%  "
          f"p90 {np.percentile(pct, 90):5.1f}%  max {pct.max():6.1f}%  bias {np.mean(e)*100:+5.1f}%")


def main():
    args = sys.argv[1:]
    save = None
    if args and args[0] == "--save":
        save = args[1]
        args = args[2:]
    holdout = None
    if args and args[0] == "--holdout":
        holdout = args[1]
        args = args[2:]
    rows = load(args)
    train = [r for r in rows if not (holdout and r["clip"].startswith(holdout))]
    test = [r for r in rows if holdout and r["clip"].startswith(holdout)]
    fit = least_squares(residuals, X0, bounds=(LO, HI), args=(train,), loss="soft_l1", f_scale=0.1)
    p = fit.x
    print("params:", ", ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, p)))
    report("train", p, train)
    for clip in sorted({r["clip"] for r in rows}):
        sub = [r for r in rows if r["clip"] == clip]
        report(("HOLDOUT " if holdout and clip.startswith(holdout) else "") + clip, p, sub)
    for which in ["P2", "P4"]:
        report(f"all, P sampled ({which})", p, rows, which)
    dp = fit_distortion(p, train)
    if save:
        json.dump({"rate": list(p), "dist": list(dp), "rate_names": NAMES, "dist_names": D_NAMES}, open(save, "w"), indent=1)
        print("saved", save)
    if test:
        e = d_residuals(dp, p, [r for r in test if r["true_mse"] > 0])
        print(f"  HOLDOUT PSNR error: median {np.median(np.abs(e)):.2f} dB, p90 {np.percentile(np.abs(e), 90):.2f} dB")


# ---------------------------------------------------------------- distortion
#
#   D_scale = fix(rung) * sum_k E_src[k] (1 - g_k)^2 / pixels   detail the rung removes
#   D_quant = exp(lnA) * Qstep(crf)^b * V^g                        V = residual energy per
#                                                                  pixel after the rung's gain
#   MSE     = D_scale + D_quant
# Fitted on log(MSE), i.e. directly in PSNR terms. fix() corrects the
# spectral estimate for the 0.6x / 0.75x rungs, whose cut-offs fall between
# DCT basis frequencies (measured: x1.16 / x1.19; 0.5x is exact).

D_NAMES = ["lnA", "b", "g"]
SCALE_FIX = {0.5: 1.0, 0.6: 1.16, 0.75: 1.19, 1.0: 1.0}


def scale_fix(r):
    f = round(r["gain_factor"], 2)
    return SCALE_FIX.get(f, 1.0)


def d_scale(r):
    c = r["c"]
    return float((c["src"] * (1.0 - r["gain"]) ** 2).sum() / (c["frames"] * c["pixels"]))


def residual_energy(r):
    c = r["c"]
    m = MAG[None, :] * r["gain"][:, None]
    return float((c["P"] * m * m).sum() / (c["pf"] * c["pixels"]))


def d_quant(params_rate, dparams, r):
    ln_a, b, g = dparams
    return math.exp(ln_a) * qstep(r["crf"]) ** b * max(residual_energy(r), 1e-6) ** g


def d_predict(params_rate, dparams, r):
    return scale_fix(r) * d_scale(r) + d_quant(params_rate, dparams, r)


def d_residuals(dparams, params_rate, rows):
    return np.array([10 * np.log10(d_predict(params_rate, dparams, r) / r["true_mse"]) for r in rows])


def fit_distortion(params_rate, rows):
    rows = [r for r in rows if r["true_mse"] > 0]
    fit = least_squares(d_residuals, [-3.6, 1.24, 0.85], args=(params_rate, rows), loss="soft_l1", f_scale=0.5)
    e = d_residuals(fit.x, params_rate, rows)
    print("distortion params:", ", ".join(f"{n}={v:.4f}" for n, v in zip(D_NAMES, fit.x)))
    print(f"  PSNR error: median {np.median(np.abs(e)):.2f} dB, p90 {np.percentile(np.abs(e), 90):.2f} dB, "
          f"bias {np.mean(e):+.2f} dB")
    e2 = np.array([10 * np.log10(scale_fix(r) * d_scale(r) / r["scale_mse"]) for r in rows if r["scale_mse"] > 0])
    print(f"  D_scale (spectral) vs measured prefilter MSE: median {np.median(np.abs(e2)):.2f} dB, "
          f"bias {np.mean(e2):+.2f} dB")
    return fit.x


if __name__ == "__main__":
    main()
