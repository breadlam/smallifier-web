# Design notes

## 1. What smallifier.py does, and what it costs

Per scene (PySceneDetect cuts), it encodes SVT-AV1 preset 2 at 4 CRFs for
each rung of a resolution ladder (each rung = downscale + upscale back to
the output size), scores every encode with VMAF, keeps the Pareto front,
and allocates the budget by greedy equal slope of ΔVMAF·duration/Δbytes.
That is up to ~20 encodes plus ~20 VMAF runs per scene. The allocation idea
is sound; the cost is in *measuring* every candidate.

## 2. Survey (Oct 2026)

| Project | Approach | Take-away |
|---|---|---|
| compress.lol | WebCodecs H.264, ffmpeg.wasm fallback, WASM AAC | Settings from a fixed bitrate ladder with a 5–7% margin; no content analysis. Notes native AAC encoders ignore low bitrates. |
| VidShift / OpenReplay / Kommodo / RedPanda (comparison write-up) | WebCodecs hardware vs. x264-in-WASM | Hardware encoders are 5–6× faster but ~10 VMAF worse at equal size (no B-frames, no lookahead, 2 s GOPs). |
| WebCodecs `bitrateMode: "quantizer"` | per-frame QP | Would allow external rate control, but support differs per browser/OS/GPU, and so would quality. |
| Av1an, ab-av1 (Rust) | per-scene target quality / CRF search by probe encodes | Same "measure candidates" cost as smallifier.py. |
| VCA, SVT-AV1 bitrate estimation from motion search (arXiv 2407.05900) | content features → bitrate | Supports predicting size from cheap analysis. |
| ρ-domain rate models (He & Mitra) | bits ∝ non-zero quantised coefficients | The basis of the size model here. |

Decision: the brief is OS/browser independence (and Firefox), so the
encoder is **x264 compiled to WebAssembly** (identical output everywhere,
the quality of a real x264), with WebCodecs used only to **decode**.
Output is H.264 + Opus in MP4.

## 3. Pipeline

```
file ──demux (wasm)──▶ samples ──▶ N workers: VideoDecoder ─▶ convert/scale (wasm) ─▶ analysis (wasm)
                                                                      │
              plan (wasm): scenes, chunks, rung + bytes per scene ◀───┘
                                     │
          N workers: decode ─▶ scale ─▶ prefilter ─▶ x264 pass 1 (fast) ─▶ target from controller ─▶ x264 pass 2
                                     │
                        mux (wasm) ──┴──▶ MP4 ──▶ over target? re-run pass 2 of 1–2 chunks
```

* Chunks are closed-GOP, `stitchable` x264 encodes (identical SPS/PPS), so
  they encode in parallel workers and concatenate without re-muxing tricks.
  No SharedArrayBuffer, so no special HTTP headers are needed (GitHub Pages
  can't set them).
* Frames are scaled in Rust (Lanczos-3, matches ffmpeg's to ~59 dB PSNR) so
  encoder input is the same in every browser.
* The output frame rate is the source rate divided by an integer; frame k
  shows the latest source frame with pts ≤ t_k + half a source frame
  (nearest-slot rounding wobbles on microsecond timestamps).

## 4. The prediction model

**Analysis.** For sampled frames: a coarse-to-fine block motion search
against the previous output frame (with an MV-cost penalty), a per-MB
inter/intra decision, and an orthonormal 8×8 DCT of the chosen residual.
Coefficient magnitudes go into 52 log-spaced bins per frequency position;
source energy per position is kept too. Every 4th frame is enough (size
error barely changes vs. every frame).

**Prefilter rungs** are modelled as a per-frequency gain, measured by
running the real fixed-point scaler on cosines, so one analysis serves all
rungs.

**Size.** At quantiser step Qstep(CRF + δ), a coefficient survives if its
normalised level exceeds a threshold that rises with frequency (trellis /
psy-rd zero high frequencies more readily); bits are linear in survivors,
their log-levels, macroblock count and an MV term, separately for the I
frame and the P frames. 11 parameters per x264 preset.

**Quality.** MSE = (detail the rung removes, from the source spectrum) +
A·Qstep^b·V^g (V = residual energy after the rung). VMAF is predicted from
PSNR by a logistic fitted to measured VMAF (vmaf_v0.6.1neg, as
smallifier.py), with a small rung term. The planner maximises
duration-weighted mean predicted VMAF — maximising mean *PSNR in dB*
instead was tried first and was wrong: it happily starved a hard scene to
25 dB to lift easy ones to 43 dB.

**Fitting data** (`crates/lab`): 4 public-domain game recordings
(CSGO, DOTA2, GTA V, StarCraft 2 from media.xiph.org, 60 s 1080p60) at
360p–1080p, 30 and 60 fps; every chunk × 4 rungs × 8 CRFs encoded with the
same x264 build and decoded to measure true PSNR — ~3,500 encodes per preset.

## 5. Measurements

Size prediction alone, per chunk, across clips: median error ~16%
(clip-level bias up to ±25%). That is why sizes are not left to the model:

| Per-chunk rate control (8 s chunks) | median size error | p90 | quality vs CRF at equal size |
|---|---|---|---|
| x264 1-pass ABR | 7.5% | 18–35% (21% on chunks < 4 s) | −0.1 to −0.4 dB |
| x264 2-pass | 1.8–2.6% | 4–6% | ±0.2 dB |

The model's job is the *decisions*. Planning from predictions versus
planning from measured encodes (the oracle, what smallifier.py does by
brute force), over 11 clip/resolution/fps sets × 3 budgets, scored on
measured data: mean gap 0.06 dB PSNR (worst 0.4) with a PSNR objective;
0.46 VMAF-proxy points (worst 2.5; `veryfast`: 0.29 / 1.9) with the VMAF
objective. The oracle itself beats a uniform-CRF encode by +0.55 proxy
points on average (+5.4 at best, on GTA V, whose scenes differ most).
Distortion model error: median 0.41 dB, p90 1.3 dB.

Output resolution: on a common reference, native lower resolution only
beats prefiltered higher resolution below ~0.02–0.05 bits per pixel, and
never on StarCraft 2 (small UI text needs pixels), hence the 0.03 bpp
threshold for the output size, with per-scene rungs below it.

End to end in the browser (Chromium headless, 6 workers, 60 s 1080p60
source → 720p30), against plain x264 two-pass at the same video bitrate on
identical frames (`tools/compare_vmaf.py`):

| Clip (target) | wall time | size | Smallifier VMAF mean / p5 | x264 2-pass VMAF mean / p5 | PSNR (S / x264) |
|---|---|---|---|---|---|
| GTA V (8 MB) | 80 s | 7.98 MB | **80.0** / 67.4 | 78.7 / 67.5 | 33.58 / 33.66 dB |
| CS:GO (10 MB) | 101 s | 9.92 MB | **68.5** / 53.7 | 67.6 / 58.4 | 29.05 / 29.10 dB |
| StarCraft 2 (6 MB) | 32 s | 5.79 MB | **90.6** / 88.1 | 90.2 / 87.4 | 36.71 / 36.44 dB |
| DOTA 2 (8 MB) | 85 s | 7.93 MB | **80.3** / 75.1 | 80.1 / 75.1 | 33.47 / 33.40 dB |

Every output is within the target, with exactly two passes per chunk (no
re-runs needed in these runs). Quality is on par with or slightly above a
single whole-clip x264 two-pass (+0.1 to +1.3 VMAF): x264's own lookahead
and MB-tree already capture much of what smallifier.py's brute-force search
bought; per-scene planning adds most on clips whose scenes differ a lot
(GTA V). Splitting into independent chunks costs essentially nothing
(measured: 11 chunked two-pass encodes vs. one continuous one, equal
quality at equal bytes). Firefox gives the same result (GTA V 8 MB:
7.98 MB, 33.2 dB) in ~130 s; the *Fast* preset takes ~50 s (33.2 dB).

For scale: smallifier.py on the same GTA V clip would encode roughly 16
scene-length candidates per scene with SVT-AV1 preset 2 and VMAF-score each
(~16× the clip in encodes, plus VMAF); here the clip is encoded ~1.3× with
x264 and never VMAF-scored. AV1 would still be ~30% smaller at equal
quality — the price of playing everywhere.

## 6. AV1 (optional)

**Encoder.** SVT-AV1 4.2 (what smallifier.py used), built from source like
x264. It normally runs a pipeline of ~16 threads, which in a browser would
need WASM threads, SharedArrayBuffer and cross-origin-isolation headers
(GitHub Pages can't set them). But since 4.x, at `lp=1` it registers its
kernels with a cooperative dispatcher on the calling thread
(`CONFIG_SINGLE_THREAD_KERNEL`), so the wasm build uses plain WASI: the
library's `svt_threads.c` is swapped for a non-blocking single-thread
stand-in (`crates/codecs/csrc/svt_threads_st.c`). AV1 therefore works
wherever the H.264 path does, including Firefox private windows. It ships
as a second module (`smallifier-av1.wasm`, 4.3 MB) fetched only when AV1 is
chosen. Settings: preset 8, smallifier.py's tuning (tune 0, no temporal
filtering, quant matrices, aq-mode 2), closed GOP, no overlays (multi-pass
doesn't allow them).

**Rate control.** Two-pass VBR per chunk, per chunk of the lab clips at
realistic budgets:

| SVT-AV1 two-pass | median size error | p90 | per-clip bias |
|---|---|---|---|
| default | 2.0% | 7.0% | −2 to −6% |
| `gop-constraint-rc=1` | **0.9%** | **2.7%** | within ±1% |

(the first pass costs ~20% of the second). With the GOP constraint it is
more accurate than x264's two-pass, so the controller uses no bias prior
and a 2% per-chunk uncertainty.

**Model.** Refitted on the same lab design with SVT-AV1 encodes (CRF
20…62, ~3,400 encodes); SVT-AV1's CRF doubles the quantiser step every
~10.6 units (H.264 QP: 6). Distortion prediction is better than for x264
(median 0.28 dB, p90 1.15 dB). Planning from predictions versus the
measured oracle: mean gap 0.17 VMAF-proxy points (worst 0.81) over 33
clip × budget cases; here the oracle barely beats a uniform CRF (+0.04),
so SVT-AV1's own allocation is already good and the planner's job is
mostly the rungs and the bytes.

**Result.** 60 s 1080p60 game clips in the browser (Chromium, 6 workers),
against plain x264 two-pass at the same video bitrate on identical frames
(`tools/compare_vmaf.py`):

| Clip (target) | wall time | size | AV1 VMAF mean / p5 | x264 2-pass VMAF mean / p5 | PSNR (AV1 / x264) |
|---|---|---|---|---|---|
| GTA V (8 MB) | 173 s | 7.96 MB | **89.2** / 81.5 | 78.7 / 67.6 | 36.20 / 33.65 dB |
| CS:GO (10 MB) | 231 s | 9.96 MB | **82.7** / 72.3 | 67.8 / 58.7 | 31.29 / 29.12 dB |
| StarCraft 2 (6 MB) | 69 s | 5.90 MB | **93.9** / 90.9 | 90.4 / 87.8 | 38.43 / 36.56 dB |
| DOTA 2 (8 MB) | 214 s | 7.96 MB | **88.5** / 82.5 | 80.3 / 75.4 | 35.94 / 33.43 dB |

Two passes per chunk everywhere (no re-runs), every chunk within ~2% of
its target; predicted PSNR within ~0.2 dB of measured (GTA V: 36.36 vs
36.20). Plays back in Chromium and Firefox. Firefox took 378 s for GTA V
(7.95 MB) while the machine was busy with other work. Encoder memory
~200 MB per worker at 720p, ~350 MB at 1080p; second-pass speed in wasm
~2.5 fps (720p) / 0.5 fps (1080p) per worker at preset 8, so 1080p
output is slow.

## 7. Things found along the way

* wasm-ld links a cdylib for WASI as a *command* unless `__wasm_call_ctors`
  is referenced: every export then runs libc destructors on return, whose
  `__stdio_exit` rewinds open input files by their buffered bytes. x264's
  second pass read its stats file misaligned ("MB-tree frametype 255").
  The module exports `_initialize` (reactor) instead.
* x264 only computes PSNR at log level ≥ INFO.
* SVT-AV1's two-pass is far off on short, scene-spanning test clips but
  accurate on real chunks; judge rate control on the chunks the app makes.
* Model constants must be ported at full precision (`tools/port_model.py`):
  rounding them to 4 decimals flipped one 5.8-million-coefficient histogram
  bin across a quantiser threshold and moved a prediction by 4%.
* ffmpeg's `gray` pixel format is full range: decoding limited-range video
  to `gray` for comparisons silently scales luma (−1.3 dB, or "29 dB
  between identical frames").

## 8. Limitations

* Sources with very long GOPs (some screen recorders write a single
  keyframe) make every analysis segment and chunk decode from the nearest
  keyframe: correct, but slower.
* 10-bit and HDR inputs are reduced to 8-bit without tone mapping.
* HEVC/AV1 input depends on the browser having a decoder for it.

## 9. Possible next steps

* Model-based output-size choice (native lower size ≈ 0.6× the bytes of the
  same-size prefilter rung at −0.5…−1.3 dB, consistently across clips).
* WASM-SIMD versions of x264's hottest C kernels (SATD 8×4, get_ref,
  mc_chroma, SAD 16×16 take ~45% of encode time).
* AAC audio option for old Apple devices.
* A slower, higher-quality AV1 preset (6); a lower bits-per-pixel
  threshold for AV1's output size (it uses H.264's 0.03 today).
* Count histogram bins fractionally at the quantiser threshold, so
  predicted size is smooth in CRF instead of slightly stepped.
