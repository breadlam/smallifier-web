# Smallifier (web)

Shrinks a video clip to a target file size, in the browser, keeping fast
action and on-screen text as clear as the budget allows. It re-implements
`smallifier.py` (kept here for reference) as a static site that can be
hosted on GitHub Pages.

Compared with `smallifier.py`:

| | smallifier.py | smallifier-web |
|---|---|---|
| Output | AV1 (SVT-AV1) + Opus in MP4 | **H.264** (x264) + Opus in MP4, which plays almost everywhere — or **AV1** (SVT-AV1) for newer devices |
| Finding settings | up to 4 CRFs × 5 resolutions encoded **and VMAF-scored** per scene, then pick | one cheap analysis pass **predicts** size and quality for every scene × resolution × CRF, then pick |
| Encodes per chunk | ~20 trial encodes + ~20 VMAF runs | a fast first pass (~30% of an encode) + the final pass; a final pass is re-run only if the file would miss the target |
| Runs on | a machine with ffmpeg/SVT-AV1/libvmaf/Python | any current Firefox, Chrome, Edge or Safari; nothing is uploaded |

## Using it

Open the page, pick a video (MP4, MOV, MKV or WebM with H.264, HEVC, VP8,
VP9 or AV1 — whatever your browser can decode), set the target size and
press **Smallify**. The result can be previewed and downloaded.

Settings:

* **Target size**, in MB (10⁶ bytes, what upload limits usually mean) or MiB.
  The output never exceeds it.
* **Maximum resolution**: automatic picks the largest of 1080p/720p/540p/…
  that the budget supports (≥ 0.03 bits per pixel); each scene can still be
  softened further (see below).
* **Frame rate**: up to 30 fps (default, as smallifier.py), or keep the
  source rate up to 60. High rates are divided by an integer (60 → 30,
  59.94 → 29.97) so motion stays even.
* **Audio**: Opus, by default 5% of the budget clamped to 24–96 kbps, as
  smallifier.py.
* **Video codec**: *H.264* plays on practically everything. *AV1* (SVT-AV1
  preset 8, smallifier.py's tuning) looks far better at the same size —
  +3.5 to +15 VMAF over x264 on our test clips — but takes about 2–4× as long and only
  plays where AV1 decoding exists: current Chrome, Edge, Firefox, Android;
  on Apple devices only those with an AV1 hardware decoder (iPhone 15 Pro,
  M3 Macs and later).
* **Encoder speed**: for H.264, *Balanced* (x264 `medium`) or *Fast* (x264
  `veryfast`, about 3× quicker, slightly softer).

Browser notes: decoding uses WebCodecs (Firefox 130+, Chrome/Edge 94+,
Safari 16.4+). Firefox on Linux decodes H.264 through the system's FFmpeg
libraries (`libavcodec`), which most desktop installs have. Encoding is
x264 or SVT-AV1 compiled to WebAssembly, so the output does not depend on
the browser, OS or GPU. The AV1 encoder (4.3 MB) is only downloaded when
AV1 is chosen. Opus-in-MP4 plays on iOS 18.4+ / macOS 15.4+ and in all
current desktop browsers.

## How it works

See [docs/DESIGN.md](docs/DESIGN.md) for the survey, the model and the
measurements. In short:

1. **Demux** (Rust/wasm): MP4/MOV (including fragmented) and Matroska/WebM,
   reading only the index structures, never the whole file.
2. **Analysis** (workers, WebCodecs decode + Rust/wasm): frames at the output
   rate and size; scene cuts from thumbnails; for every 4th frame a block
   motion search and 8×8 DCT of the residual an encoder would code, kept as
   per-frequency magnitude histograms.
3. **Plan**: from the histograms, a model fitted against thousands of real
   x264 encodes predicts each scene's size and quality at every prefilter
   *rung* (0.5/0.6/0.75/1.0 × output size, downscale + upscale, like
   smallifier.py's per-scene resolution ladder) and CRF. A greedy
   equal-slope allocation over each scene's convex hull maximises predicted
   mean VMAF within the budget — the same allocation smallifier.py ran on
   measured points.
4. **Encode** (workers, x264 or SVT-AV1 in wasm): each chunk (≤ 8 s, cut at
   scene changes) is an independent closed-GOP **two-pass** encode, which
   lands on its byte target within a few percent (SVT-AV1 within ~1%). Each chunk's final target is set
   when its first pass finishes, from the budget actually left.
5. **Mux** (Rust/wasm) H.264 + Opus into a fast-start MP4; if the real file
   is over the target, the second pass of one or two chunks is re-run.

## Development

```sh
tools/fetch-deps.sh      # x264, libopus (pinned) and wasi-sdk into third_party/
tools/build.sh           # builds web/smallifier.wasm
python3 -m http.server -d web 8000   # then open http://localhost:8000
```

Needs Rust with the `wasm32-wasip1` target (`rustup target add wasm32-wasip1`).
Pushing to `main` builds and deploys to GitHub Pages
(`.github/workflows/pages.yml`; enable Pages with "GitHub Actions" as the
source).

Layout:

* `crates/core` — demuxers, muxer, scaling, analysis, model, planner (pure Rust, unit tested: `cargo test -p smallifier-core`)
* `crates/codecs` — x264, libopus and (feature `av1`) SVT-AV1 built from source by `build.rs` (no configure or
  CMake), with small C shims
* `crates/wasm` — the WebAssembly module's C-ABI entry points
* `crates/lab` — native tools that make ground-truth encodes with the same x264 build, for fitting and checking the model
* `web/` — the page, its worker, and a minimal WASI shim with an in-memory filesystem (for x264's two-pass stats)
* `tools/` — model fitting (`fit_model.py`), allocation and resolution evaluation (`eval_alloc.py`, `eval_outres.py`),
  VMAF calibration (`vmaf_map.py`), browser end-to-end tests (`browser_test.py`, Playwright) and
  the VMAF comparison against plain two-pass (`compare_vmaf.py`)

The Python tools need `numpy` and `scipy` (`playwright` for the browser
tests, an ffmpeg with libx264 and libvmaf for the comparisons).

Refitting the model after changing encoder settings:

```sh
cargo build -p smallifier-lab --release
target/release/smallifier-lab gen CLIP.mp4 1280x720 30 target/lab/clip.json --verify --preset medium   # or svt8
python3 tools/fit_model.py --save target/fit/medium.json target/lab/*.json
python3 tools/port_model.py MEDIUM target/fit/medium.json     # writes crates/core/src/model.rs
target/release/smallifier-lab parity target/lab/clip.json medium   # must match fit_model.py's predictions
python3 tools/eval_alloc.py target/fit/medium.json target/lab/*.json
```

## Licence

x264 is GPL-2.0-or-later, so the built site is distributed under the GPL
(the crates declare `GPL-2.0-or-later`); libopus is BSD; SVT-AV1 is
BSD-3-Clause-Clear with the Alliance for Open Media patent licence.
