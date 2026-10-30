// Smallifier main thread: UI, demuxing, planning and muxing (wasm), and
// the worker pool that decodes, analyses and encodes.

import { WASI } from "./wasi.js";

const $ = (id) => document.getElementById(id);
const te = new TextEncoder();
const td = new TextDecoder();

// Output resolution ladder (short side) and prefilter rungs (fractions of
// the output size; the model was fitted on exactly these).
const LADDER = [1080, 720, 540, 432, 360, 270];
const RUNG_FACTORS = [0.5, 0.6, 0.75, 1.0];
// Lowest average bits per pixel at which a ladder size is worth encoding;
// below it the next size down is chosen. The per-scene prefilter rungs
// handle finer trade-offs within the chosen size.
const BPP_MIN = 0.03;
const AUDIO_FRACTION = 0.05;
const AUDIO_MIN = 24000;
const AUDIO_MAX = 96000;
const MAX_CHUNK_SEC = 8;
const MIN_SCENE_SEC = 1;
// Rounding allowance only: the encode controller holds back its own margin
// for two-pass size error, sized to the uncertainty that remains.
const SAFETY = 0.002;
const KNOWN_FPS = [[24000, 1001], [24, 1], [25, 1], [30000, 1001], [30, 1], [48, 1], [50, 1], [60000, 1001], [60, 1],
                   [72, 1], [90, 1], [100, 1], [120, 1], [144, 1], [165, 1], [240, 1]];

// ----------------------------------------------------------------- wasm

class Module {
  constructor(exports) {
    this.ex = exports;
  }
  get mem() {
    return new Uint8Array(this.ex.memory.buffer);
  }
  put(bytes) {
    const p = this.ex.sm_alloc(bytes.length);
    this.mem.set(bytes, p);
    return [p, bytes.length];
  }
  result() {
    const p = this.ex.sm_result_ptr();
    return this.mem.slice(p, p + this.ex.sm_result_len());
  }
  check(status, what) {
    if (status < 0) throw new Error(`${what}: ${td.decode(this.result())}`);
    return status;
  }
  json() {
    return JSON.parse(td.decode(this.result()));
  }
  /** Calls fn(ptr, len, ptr, len, ...) with the given byte arrays. */
  withBytes(fn, ...arrays) {
    const held = arrays.map((a) => this.put(a));
    try {
      return fn(...held.flat());
    } finally {
      held.forEach(([p, l]) => this.ex.sm_free(p, l));
    }
  }
}

let wasmModule;
let wasm;

async function loadWasm() {
  const url = new URL("smallifier.wasm", import.meta.url);
  try {
    wasmModule = await WebAssembly.compileStreaming(fetch(url));
  } catch {
    wasmModule = await WebAssembly.compile(await (await fetch(url)).arrayBuffer());
  }
  const wasi = new WASI({ stderr: (m) => log(`[wasm] ${m}`) });
  const inst = await WebAssembly.instantiate(wasmModule, wasi.imports);
  wasi.attach(inst);
  inst.exports._initialize();
  wasm = new Module(inst.exports);
}

// ------------------------------------------------------------------- UI

const t0 = performance.now();
function log(msg) {
  const line = `[${((performance.now() - t0) / 1000).toFixed(1)}s] ${msg}`;
  $("log").textContent += line + "\n";
  console.log(line);
}

function fmtBytes(b) {
  if (b >= 1e6) return `${(b / 1e6).toFixed(2)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} kB`;
  return `${b} B`;
}

function fmtTime(s) {
  if (!isFinite(s)) return "–";
  const m = Math.floor(s / 60);
  return m ? `${m}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
}

function showError(msg) {
  $("error").hidden = false;
  $("error").textContent = msg;
  log(`ERROR: ${msg}`);
}

const phases = ["Reading", "Audio", "Analysis", "Planning", "Encoding", "Finishing"];
function setPhase(name, detail = "") {
  const ol = $("phases");
  if (!ol.children.length) phases.forEach((p) => ol.append(Object.assign(document.createElement("li"), { textContent: p })));
  const idx = phases.indexOf(name);
  [...ol.children].forEach((li, i) => {
    li.className = i < idx ? "done" : i === idx ? "active" : "";
    li.innerHTML = phases[i] + (i === idx && detail ? ` <span class="detail">— ${detail}</span>` : "");
  });
}
function setProgress(frac, text) {
  $("bar").style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  if (text !== undefined) $("status").textContent = text;
}

// -------------------------------------------------------------- demuxing

async function readBytes(file, start, end) {
  return new Uint8Array(await file.slice(start, end).arrayBuffer());
}

async function demux(file) {
  const head = await readBytes(file, 0, 16);
  if (head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3) return demuxMkv(file);
  return demuxMp4(file);
}

async function demuxMp4(file) {
  let pos = 0;
  let moov = null;
  const moofs = [];
  while (pos + 8 <= file.size) {
    const h = await readBytes(file, pos, Math.min(file.size, pos + 16));
    const dv = new DataView(h.buffer);
    let size = dv.getUint32(0);
    const type = td.decode(h.subarray(4, 8));
    let hdr = 8;
    if (size === 1) {
      size = Number(dv.getBigUint64(8));
      hdr = 16;
    } else if (size === 0) {
      size = file.size - pos;
    }
    if (size < hdr) break;
    if (type === "moov") moov = await readBytes(file, pos + hdr, pos + size);
    else if (type === "moof") moofs.push([pos, await readBytes(file, pos + hdr, pos + size)]);
    pos += size;
  }
  if (!moov) throw new Error("Not an MP4/MOV file I can read (no moov box).");
  // moofs blob: [u64 offset][u32 len][payload]...
  const total = moofs.reduce((n, [, b]) => n + 12 + b.length, 0);
  const blob = new Uint8Array(total);
  const bdv = new DataView(blob.buffer);
  let o = 0;
  for (const [off, b] of moofs) {
    bdv.setBigUint64(o, BigInt(off), true);
    bdv.setUint32(o + 8, b.length, true);
    blob.set(b, o + 12);
    o += 12 + b.length;
  }
  wasm.check(wasm.withBytes((a, b, c, d) => wasm.ex.sm_mp4_parse(a, b, c, d), moov, blob), "MP4");
  return finishDemux(wasm.json().tracks);
}

async function demuxMkv(file) {
  wasm.ex.sm_mkv_new();
  for (;;) {
    const at = wasm.ex.sm_mkv_want();
    const end = Math.min(file.size, at + 4 * 1024 * 1024);
    const buf = at < file.size ? await readBytes(file, at, end) : new Uint8Array(0);
    const eof = end >= file.size ? 1 : 0;
    const done = wasm.withBytes((p, l) => wasm.ex.sm_mkv_feed(at, p, l, eof), buf);
    if (done) break;
    setProgress(at / file.size, `Indexing ${Math.round((100 * at) / file.size)}%`);
  }
  return finishDemux(wasm.json().tracks);
}

function finishDemux(tracks) {
  tracks.forEach((t, i) => {
    wasm.check(wasm.ex.sm_track_samples(i), "samples");
    const r = wasm.result();
    t.samples = new Float64Array(r.buffer, r.byteOffset, r.byteLength / 8);
  });
  return tracks;
}

// ----------------------------------------------------------- input model

function analyseVideoTrack(v) {
  const n = v.samples.length / 5;
  const pts = new Float64Array(n);
  const keys = [];
  for (let i = 0; i < n; i++) {
    pts[i] = v.samples[5 * i + 2];
    if (v.samples[5 * i + 4]) keys.push({ i, pts: pts[i] });
  }
  const sorted = Float64Array.from(pts).sort();
  // Mean frame interval over the whole stream: robust to millisecond
  // timestamps (Matroska: 60 fps deltas alternate 16/17 ms, so a median
  // would say 58.8 fps), and the average rate for variable-rate sources.
  const frameDur = n > 1 ? (sorted[n - 1] - sorted[0]) / (n - 1) : 1 / 30;
  const start = sorted[0];
  const duration = sorted[n - 1] - start + frameDur;
  keys.sort((a, b) => a.pts - b.pts);
  return { n, start, duration, srcFps: 1 / frameDur, keys };
}

function snapFps(fps) {
  let best = null;
  for (const [n, d] of KNOWN_FPS) {
    const err = Math.abs(n / d - fps) / fps;
    if (err < 0.015 && (!best || err < best.err)) best = { n, d, err };
  }
  return best ? [best.n, best.d] : [Math.max(1, Math.round(fps)), 1];
}

/** Output frame rate: the source rate, divided by an integer to stay <= cap. */
function outputFps(srcFps, cap) {
  const [n, d] = snapFps(srcFps);
  const div = Math.max(1, Math.ceil(n / d / cap - 1e-9));
  return [n, d * div];
}

const even = (x) => Math.max(2, Math.round(x / 2) * 2);

function scaleToShort(w, h, s) {
  const k = s / Math.min(w, h);
  return { w: even(w * k), h: even(h * k) };
}

function chooseOutput(w, h, maxShort, videoBps, fps) {
  const short = Math.min(w, h);
  const sizes = [...new Set([Math.min(short, maxShort), ...LADDER.filter((s) => s < Math.min(short, maxShort))])];
  for (const s of sizes) {
    const o = scaleToShort(w, h, s);
    if (videoBps / (o.w * o.h * fps) >= BPP_MIN) return o;
  }
  return scaleToShort(w, h, sizes[sizes.length - 1]);
}

function rungsFor(out) {
  return RUNG_FACTORS.map((f) => (f === 1 ? [out.w, out.h] : [even(out.w * f), even(out.h * f)]));
}

/** Decode-order sample range covering output frames [a, b). */
function sampleRange(video, vt, fpsN, fpsD, a, b) {
  const ta = vt.start + (a * fpsD) / fpsN;
  const tb = vt.start + (b * fpsD) / fpsN;
  let key = vt.keys[0];
  for (const k of vt.keys) {
    if (k.pts <= ta + 1e-6) key = k;
    else break;
  }
  const n = video.samples.length / 5;
  let end = key.i;
  // Include everything that decodes before the range ends (+1 s for B-frame reordering).
  while (end < n && video.samples[5 * end + 3] <= tb + 1.0) end++;
  return video.samples.slice(5 * key.i, 5 * end);
}

// ---------------------------------------------------------------- workers

class Worker2 {
  constructor() {
    this.w = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
    this.ready = new Promise((resolve, reject) => {
      this.w.onmessage = (e) => (e.data.type === "ready" ? resolve() : reject(new Error(e.data.message)));
      this.w.onerror = (e) => reject(new Error(e.message || "worker failed to start"));
    });
    this.w.postMessage({ type: "init", module: wasmModule });
  }
  /** Runs one job; `on` handles intermediate messages. */
  run(type, job, on = () => {}) {
    return new Promise((resolve, reject) => {
      this.reject = reject;
      this.w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "done") resolve(m.result);
        else if (m.type === "error") reject(new Error(m.message));
        else on(m, this);
      };
      this.w.onerror = (e) => reject(new Error(e.message || "worker error"));
      this.w.postMessage({ type, job });
    });
  }
  post(m) {
    this.w.postMessage(m);
  }
  terminate() {
    this.w.terminate();
    this.reject?.(new Error("Cancelled."));
  }
}

async function makePool(n) {
  const pool = Array.from({ length: n }, () => new Worker2());
  await Promise.all(pool.map((w) => w.ready));
  return pool;
}

/** Runs tasks (functions taking a worker) on the pool, at most one per worker. */
async function runOnPool(pool, tasks) {
  const queue = [...tasks];
  await Promise.all(pool.map(async (w) => {
    while (queue.length) await queue.shift()(w);
  }));
}

// ------------------------------------------------------------------ main

let file = null;
let tracks = null;
let cancelled = false;
let activePool = null;

function fillWorkerOptions() {
  const cores = navigator.hardwareConcurrency || 4;
  const sel = $("workers");
  const auto = Math.max(1, Math.min(cores - 1, 8));
  sel.innerHTML = `<option value="auto">Auto (${auto})</option>`;
  for (let i = 1; i <= Math.max(16, cores); i++) sel.innerHTML += `<option value="${i}">${i}</option>`;
}

function workerCount(out) {
  const v = $("workers").value;
  if (v !== "auto") return +v;
  const cores = navigator.hardwareConcurrency || 4;
  // x264 needs ~350 MB per encoder at 1080p, ~170 MB at 720p.
  const px = out.w * out.h;
  const memCap = px > 1.5e6 ? 4 : px > 0.6e6 ? 6 : 8;
  const devMem = navigator.deviceMemory; // Chrome only; GB, capped at 8
  const memCap2 = devMem ? Math.max(1, Math.floor((devMem * 1024 * 0.5) / (px > 1.5e6 ? 350 : 170))) : memCap;
  return Math.max(1, Math.min(cores - 1, memCap, memCap2));
}

async function onFile(f) {
  $("error").hidden = true;
  $("result").hidden = true;
  file = f;
  log(`file: ${f.name} (${fmtBytes(f.size)})`);
  try {
    tracks = await demux(f);
  } catch (e) {
    showError(`Couldn't read this file: ${e.message}`);
    return;
  }
  const video = tracks.find((t) => t.kind === "video");
  const audio = tracks.find((t) => t.kind === "audio");
  if (!video) {
    showError("No video track found.");
    return;
  }
  const vt = analyseVideoTrack(video);
  const dl = [
    ["File", `${f.name} — ${fmtBytes(f.size)}`],
    ["Duration", fmtTime(vt.duration)],
    ["Video", `${video.codec_name} ${video.width}×${video.height}, ${vt.srcFps.toFixed(2)} fps` +
      (video.rotation ? `, rotated ${video.rotation}°` : "")],
    ["Audio", audio ? `${audio.codec_name} ${audio.sample_rate} Hz, ${audio.channels} ch` : "none"],
  ];
  $("info").innerHTML = "<dl>" + dl.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + "</dl>";
  $("info").hidden = false;
  $("settings").hidden = false;
  log(`video ${video.codec} ${video.width}x${video.height} ${vt.n} samples, ${vt.keys.length} keyframes, ` +
      `${vt.srcFps.toFixed(3)} fps, ${vt.duration.toFixed(2)} s; audio ${audio ? audio.codec : "none"}`);
}

async function smallify() {
  cancelled = false;
  $("go").disabled = true;
  $("error").hidden = true;
  $("result").hidden = true;
  $("progress").hidden = false;
  $("phases").innerHTML = "";
  const started = performance.now();
  const timing = {};
  const mark = (k, t) => (timing[k] = (performance.now() - t) / 1000);
  let pool = null;
  try {
    const video = tracks.find((t) => t.kind === "video");
    const audioTrack = tracks.find((t) => t.kind === "audio");
    const vt = analyseVideoTrack(video);
    setPhase("Reading");

    // Decoder support (fail early with a clear message).
    const vcfg = { codec: video.codec, codedWidth: video.width, codedHeight: video.height };
    if (video.description) vcfg.description = Uint8Array.from(video.description.match(/../g), (b) => parseInt(b, 16));
    if (!video.codec || !(await VideoDecoder.isConfigSupported(vcfg)).supported) {
      throw new Error(`This browser can't decode the ${video.codec_name} video in this file.`);
    }

    // ---- targets
    const target = +$("target").value * +$("unit").value;
    const [fpsN, fpsD] = outputFps(vt.srcFps, +$("fps").value);
    const fps = fpsN / fpsD;
    const frames = Math.max(1, Math.round(vt.duration * fps));
    const duration = frames / fps;
    const preset = $("preset").value;
    let audioBps = $("audio").value === "auto"
      ? Math.round(Math.max(AUDIO_MIN, Math.min(AUDIO_MAX, ((target * 8) / duration) * AUDIO_FRACTION)))
      : +$("audio").value;
    const wantAudio = audioBps > 0 && audioTrack;
    const roughVideo = target * (1 - SAFETY) - (wantAudio ? (audioBps * duration) / 8 : 0);
    const maxShort = $("maxres").value === "auto" ? 1080 : +$("maxres").value;
    const out = chooseOutput(video.width, video.height, maxShort, (roughVideo * 8) / duration, fps);
    const rungs = rungsFor(out);
    const nWorkers = workerCount(out);
    log(`target ${fmtBytes(target)}; output ${out.w}x${out.h} @ ${fps.toFixed(3)} fps (${frames} frames), ` +
        `audio ${wantAudio ? audioBps / 1000 + " kbps" : "none"}, preset ${preset}, ${nWorkers} workers`);
    if (roughVideo <= 0) throw new Error("The target is too small to hold even the audio. Lower the audio bitrate or raise the target.");

    pool = await makePool(nWorkers);
    activePool = pool;
    const base = { file, track: video, t0: vt.start, fpsNum: fpsN, fpsDen: fpsD, srcW: video.width, srcH: video.height,
                   srcFrameDur: 1 / vt.srcFps };

    // ---- audio (runs alongside the analysis on one worker)
    setPhase("Audio");
    let audioResult = null;
    const audioJob = (async () => {
      if (!wantAudio) return;
      const t = performance.now();
      const acfg = { codec: audioTrack.codec, sampleRate: audioTrack.sample_rate, numberOfChannels: audioTrack.channels || 2 };
      if (audioTrack.description) acfg.description = Uint8Array.from(audioTrack.description.match(/../g), (b) => parseInt(b, 16));
      const supported = audioTrack.codec && (await AudioDecoder.isConfigSupported(acfg).catch(() => ({}))).supported;
      const w = pool[pool.length - 1];
      if (supported) {
        audioResult = await w.run("audio", {
          file, track: audioTrack, samples: audioTrack.samples, t0: vt.start, duration, bitrate: audioBps,
        });
      } else {
        // Let the browser's media stack demux and decode it (main thread only).
        log(`audio: ${audioTrack.codec_name} not decodable by WebCodecs here, trying decodeAudioData`);
        try {
          const ctx = new OfflineAudioContext(2, 1, 48000);
          const buf = await ctx.decodeAudioData(await file.arrayBuffer());
          const skip = Math.max(0, Math.round((vt.start - 0) * 48000));
          const len = Math.round(duration * 48000);
          const planes = [];
          for (let c = 0; c < Math.min(2, buf.numberOfChannels); c++) {
            const p = new Float32Array(len);
            p.set(buf.getChannelData(c).subarray(skip, skip + len));
            planes.push(p);
          }
          audioResult = await w.run("audioPcm", { planes, bitrate: audioBps });
        } catch (e) {
          log(`audio dropped: ${e.message}`);
        }
      }
      mark("audio", t);
      if (audioResult) log(`audio: ${fmtBytes(audioResult.data.length)} Opus (${audioResult.head.packets.length} packets)`);
    })();

    // ---- analysis, split at source keyframes where possible
    setPhase("Analysis", `${out.w}×${out.h}`);
    const tA = performance.now();
    const every = frames > 6000 ? 8 : 4;
    // Segments must be long enough to contain analysed frames.
    const nSeg = Math.max(1, Math.min(Math.floor(frames / (4 * every)), Math.max(nWorkers * 2, Math.ceil(frames / 900))));
    const keyFrames = vt.keys.map((k) => Math.round((k.pts - vt.start) * fps)).filter((k) => k > 0 && k < frames);
    const bounds = [0];
    for (let s = 1; s < nSeg; s++) {
      const ideal = Math.round((frames * s) / nSeg);
      const near = keyFrames.reduce((b, k) => (Math.abs(k - ideal) < Math.abs(b - ideal) ? k : b), -1e9);
      const pick = Math.abs(near - ideal) <= frames / nSeg / 3 ? near : ideal;
      if (pick > bounds[bounds.length - 1]) bounds.push(pick);
    }
    bounds.push(frames);
    const segs = bounds.slice(0, -1).map((a, i) => ({ a, b: bounds[i + 1] }));
    const analysed = new Array(segs.length);
    let anaFrames = 0;
    await runOnPool(pool.slice(0, wantAudio && nWorkers > 1 ? -1 : undefined), segs.map((s, i) => async (w) => {
      if (cancelled) return;
      analysed[i] = await w.run("analyze", {
        ...base, a: s.a, b: s.b, every, out, samples: sampleRange(video, vt, fpsN, fpsD, s.a, s.b),
      }, (m) => {
        if (m.frames) {
          anaFrames += m.frames;
          setProgress(anaFrames / frames, `Analysed ${anaFrames} of ${frames} frames`);
        }
      });
    }));
    if (cancelled) throw new Error("Cancelled.");
    const diffs = new Float32Array(frames);
    const index = [];
    const statParts = [];
    analysed.forEach((r, i) => {
      diffs.set(r.diffs, segs[i].a);
      if (i > 0) {
        const prev = analysed[i - 1].lastThumb;
        diffs[segs[i].a] = wasm.withBytes((a, al, b) => wasm.ex.sm_thumb_diff(a, b, al), prev, r.firstThumb);
      }
      index.push(...r.index);
      statParts.push(r.stats);
    });
    mark("analysis", tA);
    log(`analysis: ${index.length} frames sampled in ${timing.analysis.toFixed(1)} s`);

    await audioJob;
    if (cancelled) throw new Error("Cancelled.");

    // ---- plan
    setPhase("Planning");
    const audioBytes = audioResult ? audioResult.data.length : 0;
    const overhead = wasm.ex.sm_mux_overhead(frames, fps, audioResult ? audioResult.head.packets.length : 0, audioResult ? 1 : 0);
    const budget = Math.floor(target * (1 - SAFETY) - audioBytes - overhead);
    if (budget <= frames * 50) throw new Error(`The target leaves only ${fmtBytes(Math.max(0, budget))} for video; that's too little.`);
    const stats = new Uint8Array(statParts.reduce((n, s) => n + s.length, 0));
    let so = 0;
    for (const s of statParts) {
      stats.set(s, so);
      so += s.length;
    }
    const planIn = {
      frames, fps, out_w: out.w, out_h: out.h, rungs, diffs: Array.from(diffs), sampled: index, budget,
      max_chunk_frames: Math.round(MAX_CHUNK_SEC * fps), min_scene_sec: MIN_SCENE_SEC, preset,
    };
    wasm.check(wasm.withBytes((a, b, c, d) => wasm.ex.sm_plan(a, b, c, d), te.encode(JSON.stringify(planIn)), stats), "plan");
    const { chunks, plan } = wasm.json();
    log(`plan: ${plan.scenes.length} scenes, ${chunks.length} chunks, video budget ${fmtBytes(budget)}, ` +
        `predicted PSNR ${plan.predicted_psnr.toFixed(2)} dB, VMAF ~${plan.predicted_vmaf.toFixed(1)}` +
        `${plan.feasible ? "" : " (budget below the cheapest option!)"}`);
    plan.scenes.forEach((s, i) => log(`  scene ${i}: ${s.frames} frames, rung ${rungs[s.rung].join("x")}, CRF~${s.crf}, ` +
                                      `${fmtBytes(s.bytes)}, ${s.psnr.toFixed(1)} dB`));

    // ---- encode
    setPhase("Encoding", `${chunks.length} chunk${chunks.length === 1 ? "" : "s"}`);
    const tE = performance.now();
    const results = new Array(chunks.length);
    let passFrames = 0;
    let encodes = 0;
    const totalPassFrames = frames * 2;
    const chunkJob = (c) => ({
      ...base, chunk: c, start: chunks[c].start, end: chunks[c].end, out, preset,
      rung: { w: rungs[plan.scenes[chunks[c].scene].rung][0], h: rungs[plan.scenes[chunks[c].scene].rung][1] },
      planned: plan.chunk_bytes[c], samples: sampleRange(video, vt, fpsN, fpsD, chunks[c].start, chunks[c].end),
    });
    const requested = new Array(chunks.length).fill(0);
    const onMsg = (c) => (m, w) => {
      if (m.type === "pass1done") {
        const bytes = wasm.ex.sm_ctl_target(c);
        requested[c] = bytes;
        w.post({ type: "target", bytes });
      } else if (m.frames) {
        passFrames += m.frames;
        const el = (performance.now() - tE) / 1000;
        const eta = (el / passFrames) * (totalPassFrames - passFrames);
        setProgress(passFrames / totalPassFrames, `Encoded ${Math.round(passFrames / 2)} of ${frames} frames ` +
                    `(two passes) — about ${fmtTime(eta)} left`);
      }
    };
    await Promise.all(pool.map(async (w) => {
      for (;;) {
        if (cancelled) return;
        const c = wasm.ex.sm_ctl_next();
        if (c < 0) return;
        const r = await w.run("encode", chunkJob(c), onMsg(c));
        encodes += 2;
        results[c] = r;
        wasm.ex.sm_ctl_finished(c, r.head.bytes);
        log(`  chunk ${c} (${chunks[c].end - chunks[c].start} frames): planned ${fmtBytes(plan.chunk_bytes[c])}, ` +
            `asked ${fmtBytes(requested[c])}, got ${fmtBytes(r.head.bytes)} (${((r.head.bytes / requested[c] - 1) * 100).toFixed(1)}%)`);
      }
    }));
    if (cancelled) throw new Error("Cancelled.");
    log(`encoded: ${fmtBytes(wasm.ex.sm_ctl_total())} of video for a ${fmtBytes(budget)} budget`);

    // ---- mux, then fix up against the real file size
    setPhase("Finishing");
    const mux = () => {
      const sps = results[0].head.sps;
      const pps = results[0].head.pps;
      for (const r of results) {
        if (r.head.sps !== sps || r.head.pps !== pps) throw new Error("chunk parameter sets differ (internal error)");
      }
      const vrows = [];
      const parts = [];
      results.forEach((r, c) => {
        for (const [size, pts, key] of r.head.packets) vrows.push([size, pts + chunks[c].start, key]);
        parts.push(r.data);
      });
      const muxIn = {
        width: out.w, height: out.h, fps_num: fpsN, fps_den: fpsD, rotation: video.rotation || 0, sps, pps, video: vrows,
      };
      if (audioResult) {
        parts.push(audioResult.data);
        muxIn.audio = {
          channels: audioResult.head.channels, pre_skip: audioResult.head.pre_skip, input_rate: audioTrack.sample_rate,
          samples: audioResult.head.samples, packets: audioResult.head.packets,
        };
      }
      const data = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
      let off = 0;
      for (const p of parts) {
        data.set(p, off);
        off += p.length;
      }
      wasm.check(wasm.withBytes((a, b, c, d) => wasm.ex.sm_mux(a, b, c, d), te.encode(JSON.stringify(muxIn)), data), "mux");
      return wasm.result();
    };
    let mp4 = mux();
    // Over the target: always fix (aim 0.3% under). Under by more than 6%:
    // worth one re-run to use the room.
    for (let round = 0; round < 3; round++) {
      let excess = 0;
      if (mp4.length > target) excess = mp4.length - target * 0.997;
      else if (round === 0 && mp4.length < target * 0.94) excess = mp4.length - target * 0.99;
      if (!excess) break;
      wasm.check(wasm.ex.sm_ctl_fixups(excess, pool.length), "fixups");
      const fixes = wasm.json();
      if (!fixes.length) break;
      log(`${excess > 0 ? "over" : "under"} target by ${fmtBytes(Math.abs(mp4.length - target))}: ` +
          `re-running the second pass of ${fixes.length} chunk(s)`);
      setPhase("Encoding", `adjusting ${fixes.length} chunk(s)`);
      await runOnPool(pool, fixes.map((f) => async (w) => {
        const job = { ...chunkJob(f.chunk), stats: results[f.chunk].stats, target: f.bytes };
        const r = await w.run("encode", job, onMsg(f.chunk));
        encodes += 1;
        log(`  re-run chunk ${f.chunk}: ${fmtBytes(results[f.chunk].head.bytes)} -> asked ${fmtBytes(f.bytes)}, ` +
            `got ${fmtBytes(r.head.bytes)}`);
        results[f.chunk] = r;
        wasm.ex.sm_ctl_finished(f.chunk, r.head.bytes);
      }));
      setPhase("Finishing");
      mp4 = mux();
    }
    mark("encode", tE);
    const elapsed = (performance.now() - started) / 1000;

    // ---- report
    let se = 0;
    let n = 0;
    results.forEach((r, c) => {
      const f = chunks[c].end - chunks[c].start;
      se += (r.head.mse + r.head.scale_mse) * f;
      n += f;
    });
    const psnr = 10 * Math.log10(65025 / (se / n));
    showResult(mp4, {
      target, out, fps, frames, plan, chunks, rungs, results, psnr, elapsed, timing, encodes, audioBytes, preset,
      srcSize: file.size,
    });
    log(`done: ${fmtBytes(mp4.length)} (target ${fmtBytes(target)}), measured PSNR ${psnr.toFixed(2)} dB, ` +
        `${encodes} chunk passes for ${chunks.length} chunks, ${elapsed.toFixed(1)} s`);
  } catch (e) {
    showError(e.message);
    console.error(e);
  } finally {
    pool?.forEach((w) => w.terminate());
    activePool = null;
    $("go").disabled = false;
    $("progress").hidden = true;
  }
}

function showResult(mp4, r) {
  const blob = new Blob([mp4], { type: "video/mp4" });
  const url = URL.createObjectURL(blob);
  $("preview").src = url;
  const a = $("download");
  a.href = url;
  a.download = file.name.replace(/\.[^.]+$/, "") + "_small.mp4";
  a.textContent = `Download ${a.download} (${fmtBytes(mp4.length)})`;
  const ok = mp4.length <= r.target;
  const rows = [
    ["Size", `${fmtBytes(mp4.length)} of ${fmtBytes(r.target)} target (${((mp4.length / r.target) * 100).toFixed(1)}%)` +
      (ok ? "" : " — over target!")],
    ["Video", `H.264 ${r.out.w}×${r.out.h} @ ${r.fps.toFixed(2)} fps, ${fmtBytes(mp4.length - r.audioBytes)}`],
    ["Audio", r.audioBytes ? `Opus, ${fmtBytes(r.audioBytes)}` : "none"],
    ["Quality", `${r.psnr.toFixed(2)} dB PSNR vs. source at output size (predicted ${r.plan.predicted_psnr.toFixed(2)} dB; ` +
      `predicted VMAF ~${r.plan.predicted_vmaf.toFixed(0)})`],
    ["Work", `${r.chunks.length} chunks, ${r.encodes} encoder passes (fast first pass + final pass per chunk` +
      (r.encodes > 2 * r.chunks.length ? `, ${r.encodes - 2 * r.chunks.length} re-run(s)` : "") + ")"],
    ["Time", `${fmtTime(r.elapsed)} total — analysis ${fmtTime(r.timing.analysis)}, encoding ${fmtTime(r.timing.encode)}`],
  ];
  $("summary").innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
  $("scenes").innerHTML = "<tr><th>Scene</th><th class=num>Frames</th><th>Prefilter</th><th class=num>Planned</th>" +
    "<th class=num>Actual</th><th class=num>PSNR</th></tr>" +
    r.plan.scenes.map((s, i) => {
      const cs = r.chunks.map((c, j) => [c, j]).filter(([c]) => c.scene === i);
      const actual = cs.reduce((t, [, j]) => t + r.results[j].head.bytes, 0);
      let se = 0;
      let n = 0;
      cs.forEach(([c, j]) => {
        se += (r.results[j].head.mse + r.results[j].head.scale_mse) * (c.end - c.start);
        n += c.end - c.start;
      });
      return `<tr><td>${i + 1}</td><td class=num>${s.frames}</td><td>${r.rungs[s.rung].join("×")}</td>` +
        `<td class=num>${fmtBytes(cs.reduce((t, [, j]) => t + r.plan.chunk_bytes[j], 0))}</td>` +
        `<td class=num>${fmtBytes(actual)}</td><td class=num>${(10 * Math.log10(65025 / (se / n))).toFixed(1)} dB</td></tr>`;
    }).join("");
  $("result").hidden = false;
  window.__smallifierResult = { size: mp4.length, target: r.target, psnr: r.psnr, chunks: r.chunks.length,
                                encodes: r.encodes, elapsed: r.elapsed, out: r.out, fps: r.fps,
                                perChunk: r.chunks.map((c, j) => ({ start: c.start, end: c.end, rung: r.rungs[r.plan.scenes[c.scene].rung],
                                  mse: r.results[j].head.mse, scale_mse: r.results[j].head.scale_mse, bytes: r.results[j].head.bytes })) };
}

// ---------------------------------------------------------------- wiring

async function boot() {
  fillWorkerOptions();
  if (!("VideoDecoder" in self)) {
    showError("This browser doesn't support WebCodecs, which Smallifier needs for decoding. " +
              "Use a current Firefox, Chrome, Edge or Safari (16.4+).");
    return;
  }
  try {
    await loadWasm();
  } catch (e) {
    showError(`Couldn't load the encoder: ${e.message}`);
    return;
  }
  const input = $("file");
  input.onchange = () => input.files[0] && onFile(input.files[0]);
  const drop = $("drop");
  drop.ondragover = (e) => {
    e.preventDefault();
    drop.classList.add("over");
  };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  };
  $("go").onclick = smallify;
  $("cancel").onclick = () => {
    cancelled = true;
    activePool?.forEach((w) => w.terminate());
  };
  log("ready");
  window.__smallifierReady = true;
  // Diagnostics for tools/browser_test.py --dump: decoded luma of output
  // frames [a, b) at the chosen output frame rate.
  window.__smallifierDump = async (a, b, fpsCap, raw = false) => {
    const video = tracks.find((t) => t.kind === "video");
    const vt = analyseVideoTrack(video);
    const [fpsN, fpsD] = outputFps(vt.srcFps, fpsCap);
    const [w] = await makePool(1);
    try {
      const r = await w.run("dump", {
        file, track: video, t0: vt.start, fpsNum: fpsN, fpsDen: fpsD, srcFrameDur: 1 / vt.srcFps, a, b, raw,
        samples: sampleRange(video, vt, fpsN, fpsD, a, b),
      });
      let s = "";
      for (let i = 0; i < r.luma.length; i += 0x8000) s += String.fromCharCode(...r.luma.subarray(i, i + 0x8000));
      return btoa(s);
    } finally {
      w.terminate();
    }
  };
}

boot();
