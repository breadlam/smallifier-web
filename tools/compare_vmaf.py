#!/usr/bin/env python3
"""Fair quality comparison of a Smallifier output against plain x264
two-pass at the same video bitrate, both scored on the same frames.

  compare_vmaf.py APP_OUTPUT.mp4 SOURCE [--workdir DIR]

The reference is the source decoded by ffmpeg, scaled to the output size
with lanczos (identical to the app's scaler to ~59 dB) and decimated by
frame index to the output rate (the app's frame selection for integer
ratios). The baseline encodes exactly those reference frames with x264
medium two-pass. Prints mean/p5/min VMAF (vmaf_v0.6.1neg) and luma PSNR.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np


def probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={entries}",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def score(dist_yuv, ref_yuv, w, h, fps, log):
    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", str(fps)]
    subprocess.run(["ffmpeg", "-v", "error", *raw, "-i", dist_yuv, *raw, "-i", ref_yuv, "-lavfi",
                    f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1neg:n_threads=48:log_fmt=json:log_path={log}",
                    "-f", "null", "-"], check=True)
    v = np.array([f["metrics"]["vmaf"] for f in json.load(open(log))["frames"]])
    fs, ys = w * h * 3 // 2, w * h
    a = np.memmap(dist_yuv, np.uint8, "r").reshape(-1, fs)
    r = np.memmap(ref_yuv, np.uint8, "r").reshape(-1, fs)
    mse = np.mean([np.mean((a[k, :ys].astype(np.float32) - r[k, :ys].astype(np.float32)) ** 2) for k in range(len(a))])
    return v, 10 * math.log10(65025 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("app")
    ap.add_argument("source")
    ap.add_argument("--workdir", default="target/compare")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    wd = lambda n: os.path.join(args.workdir, n)

    o = probe(args.app, "width,height,r_frame_rate,bit_rate,nb_frames")
    s = probe(args.source, "r_frame_rate")
    w, h = o["width"], o["height"]
    on, od = map(int, o["r_frame_rate"].split("/"))
    sn, sd = map(int, s["r_frame_rate"].split("/"))
    step = round((sn / sd) / (on / od))
    fps = on / od
    kbps = int(o["bit_rate"]) // 1000

    src_all = wd("src_all.yuv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.source, "-vf", f"scale={w}:{h}:flags=lanczos",
                    "-pix_fmt", "yuv420p", "-f", "rawvideo", src_all], check=True)
    fs = w * h * 3 // 2
    allf = np.memmap(src_all, np.uint8, "r").reshape(-1, fs)
    n = int(o["nb_frames"])
    ref = wd("ref.yuv")
    np.ascontiguousarray(allf[::step][:n]).tofile(ref)
    os.unlink(src_all)

    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", f"{on}/{od}"]
    x264 = ["-c:v", "libx264", "-preset", "medium", "-x264-params", "aq-mode=3:rc-lookahead=20", "-b:v", f"{kbps}k"]
    log = wd("x264pass")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *raw, "-i", ref, *x264, "-pass", "1", "-passlogfile", log, "-f", "null", "-"], check=True)
    base = wd("baseline.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *raw, "-i", ref, *x264, "-pass", "2", "-passlogfile", log, base], check=True)

    for name, path in [("smallifier", args.app), ("x264 2-pass", base)]:
        yuv = wd(os.path.basename(path) + ".yuv")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0:v", "-pix_fmt", "yuv420p", "-f", "rawvideo", yuv], check=True)
        v, psnr = score(yuv, ref, w, h, fps, yuv + ".vmaf.json")
        vb = int(probe(path, "bit_rate")["bit_rate"]) // 1000
        print(f"{os.path.basename(args.source):16s} {name:12s} video {vb:5d} kbps  VMAF mean {v.mean():5.2f}  p5 {np.percentile(v, 5):5.1f}  "
              f"min {v.min():5.1f}  PSNR {psnr:5.2f} dB")
        os.unlink(yuv)
    os.unlink(ref)


if __name__ == "__main__":
    main()
