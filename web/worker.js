// Worker: decodes its part of the input with WebCodecs and runs analysis,
// two-pass x264 chunk encodes, or the audio encode in its own wasm instance.

import { WASI } from "./wasi.js";

let ex, wasi;
const te = new TextEncoder();
const td = new TextDecoder();
let frameBuf = { ptr: 0, size: 0 };
let fmtCache = new Map();

const mem = () => new Uint8Array(ex.memory.buffer);
function put(bytes) {
  const p = ex.sm_alloc(bytes.length);
  mem().set(bytes, p);
  return [p, bytes.length];
}
function result() {
  const p = ex.sm_result_ptr();
  return mem().slice(p, p + ex.sm_result_len());
}
function check(status, what) {
  if (status < 0) throw new Error(`${what}: ${td.decode(result())}`);
  return status;
}
function callJson(fn, obj) {
  const [p, l] = put(te.encode(JSON.stringify(obj)));
  const r = fn(p, l);
  ex.sm_free(p, l);
  return r;
}
function fmtPtr(fmt) {
  if (!fmtCache.has(fmt)) fmtCache.set(fmt, put(te.encode(fmt)));
  return fmtCache.get(fmt);
}
const hexToBytes = (h) => new Uint8Array(h.match(/../g)?.map((b) => parseInt(b, 16)) ?? []);

async function init(module) {
  wasi = new WASI({ stdout: (m) => console.log("[wasm]", m), stderr: (m) => console.warn("[wasm]", m) });
  const inst = await WebAssembly.instantiate(module, wasi.imports);
  wasi.attach(inst);
  ex = inst.exports;
  ex._initialize();
}

// ------------------------------------------------------------- frame input

const DIRECT = new Set(["I420", "I420A", "I422", "I444", "NV12", "I420P10", "I422P10", "I444P10", "I420P12",
                        "RGBA", "RGBX", "BGRA", "BGRX"]);
let canvas = null;

let loggedFormat = false;

/** Copies a VideoFrame into the module's current input frame. */
async function loadFrame(frame) {
  if (!loggedFormat) {
    loggedFormat = true;
    console.log(`[worker] decoded frames: format ${frame.format}, coded ${frame.codedWidth}x${frame.codedHeight}, ` +
                `visible ${JSON.stringify(frame.visibleRect)}, colorSpace ${JSON.stringify(frame.colorSpace)}`);
  }
  const w = frame.visibleRect?.width ?? frame.codedWidth;
  const h = frame.visibleRect?.height ?? frame.codedHeight;
  let fmt = frame.format;
  let layout;
  if (fmt && DIRECT.has(fmt)) {
    const size = frame.allocationSize();
    if (frameBuf.size < size) {
      if (frameBuf.ptr) ex.sm_free(frameBuf.ptr, frameBuf.size);
      frameBuf = { ptr: ex.sm_alloc(size), size };
    }
    layout = await frame.copyTo(new Uint8Array(ex.memory.buffer, frameBuf.ptr, size));
  } else {
    // Opaque (GPU) frame: go through a canvas.
    if (!canvas || canvas.width !== w || canvas.height !== h) {
      canvas = new OffscreenCanvas(w, h);
    }
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(frame, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h).data;
    const size = img.length;
    if (frameBuf.size < size) {
      if (frameBuf.ptr) ex.sm_free(frameBuf.ptr, frameBuf.size);
      frameBuf = { ptr: ex.sm_alloc(size), size };
    }
    mem().set(img, frameBuf.ptr);
    fmt = "RGBA";
    layout = [{ offset: 0, stride: w * 4 }];
  }
  const lay = new Uint32Array(layout.length * 2);
  layout.forEach((l, i) => {
    lay[2 * i] = l.offset;
    lay[2 * i + 1] = l.stride;
  });
  const [lp, ll] = put(new Uint8Array(lay.buffer));
  const [fp, fl] = fmtPtr(fmt);
  const r = ex.sm_frame(fp, fl, w, h, frameBuf.ptr, frameBuf.size, lp, layout.length);
  ex.sm_free(lp, ll);
  check(r, "frame");
  return { w, h };
}

// ---------------------------------------------------------------- decoding

/**
 * Decodes job.samples (decode order, 5 f64 each: offset, size, pts, dts,
 * key; starting at a keyframe) and calls onFrame(k, frame) for each output
 * frame index k in [from, to), at the output rate: frame k shows the
 * latest source frame with pts <= t_k + half a source frame. (Rounding to
 * the nearest slot is not stable: microsecond timestamps put 60 -> 30 fps
 * exactly on .5 ties and the phase would wobble from frame to frame.)
 */
async function decodeRange(job, from, to, onFrame) {
  const { file, track, samples, t0, fpsNum, fpsDen, srcFrameDur } = job;
  const fps = fpsNum / fpsDen;
  // First output slot a frame is shown in.
  const slotOf = (us) => Math.ceil((us / 1e6 - t0 - 0.5 * srcFrameDur) * fps - 1e-6);
  const out = [];
  let failure = null;
  const decoder = new VideoDecoder({ output: (f) => out.push(f), error: (e) => (failure = e) });
  const config = { codec: track.codec, codedWidth: track.width, codedHeight: track.height, optimizeForLatency: false };
  if (track.description) config.description = hexToBytes(track.description);
  decoder.configure(config);

  let pending = null;
  let k = from;
  const emit = async (final) => {
    while (out.length) {
      const f = out.shift();
      if (k >= to) {
        f.close();
        continue;
      }
      const s = slotOf(f.timestamp);
      if (pending) {
        while (k < Math.min(s, to)) await onFrame(k++, pending);
        pending.close();
      }
      pending = f;
    }
    if (final && pending) {
      while (k < to) await onFrame(k++, pending);
      pending.close();
      pending = null;
    }
  };

  const n = samples.length / 5;
  let i = 0;
  // Read the file in runs of up to 256 samples / 8 MB (a run may span
  // interleaved audio; reading a little extra beats many small reads).
  while (i < n && k < to) {
    if (failure) throw failure;
    const start = samples[5 * i];
    let end = start + samples[5 * i + 1];
    let j = i + 1;
    while (j < n && j - i < 256) {
      const o = samples[5 * j];
      const e = o + samples[5 * j + 1];
      if (o < start || e - start > 8e6) break;
      end = Math.max(end, e);
      j++;
    }
    const buf = new Uint8Array(await file.slice(start, end).arrayBuffer());
    for (; i < j && k < to; i++) {
      const off = samples[5 * i] - start;
      decoder.decode(new EncodedVideoChunk({
        type: samples[5 * i + 4] ? "key" : "delta",
        timestamp: Math.round(samples[5 * i + 2] * 1e6),
        data: buf.subarray(off, off + samples[5 * i + 1]),
      }));
      while (decoder.decodeQueueSize > 6 || out.length > 3) {
        await emit(false);
        if (failure) throw failure;
        if (decoder.decodeQueueSize > 6) await new Promise((r) => setTimeout(r, 1));
      }
      await emit(false);
    }
  }
  if (failure) throw failure;
  if (k < to && decoder.state === "configured") {
    await decoder.flush();
    await emit(true);
  }
  pending?.close();
  out.forEach((f) => f.close());
  decoder.close();
  if (k < to) throw new Error(`decoded only up to frame ${k} of ${to}`);
}

// ---------------------------------------------------------------- analysis

async function analyze(job) {
  const { a, b, every, out } = job;
  ex.sm_ana_begin(job.srcW, job.srcH, out.w, out.h);
  // Frame k (> a) is analysed when (k - a) % every == every - 1, against
  // frame k - 1, which is therefore kept. Frame a has no predecessor here;
  // a segment too short for that still analyses its last frame.
  const last = b - 1;
  const sampled = (k) => k > a && ((k - a) % every === (every - 1) % every || (k === last && last - a < every));
  let done = 0;
  await decodeRange(job, a, b, async (k, frame) => {
    await loadFrame(frame);
    let flags = 2;
    if (sampled(k)) flags |= 1;
    if (sampled(k + 1)) flags |= 4;
    check(ex.sm_ana_push(flags), "analysis");
    if (++done % 10 === 0) postMessage({ type: "progress", frames: 10 });
  });
  postMessage({ type: "progress", frames: done % 10 });
  check(ex.sm_ana_end(), "analysis end");
  const r = result();
  const dv = new DataView(r.buffer);
  const n = dv.getUint32(0, true);
  const diffs = new Float32Array(r.buffer.slice(4, 4 + 4 * n));
  let p = 4 + 4 * n;
  const m = dv.getUint32(p, true);
  p += 4;
  const sz = ex.sm_stats_size();
  const index = new Uint32Array(m);
  const stats = new Uint8Array(m * sz);
  for (let i = 0; i < m; i++) {
    index[i] = dv.getUint32(p, true) + a;
    stats.set(r.subarray(p + 4, p + 4 + sz), i * sz);
    p += 4 + sz;
  }
  const t = dv.getUint32(p, true);
  const firstThumb = r.slice(p + 4, p + 4 + t);
  const lastThumb = r.slice(p + 4 + t, p + 4 + 2 * t);
  return { diffs, index, stats, firstThumb, lastThumb,
           transfer: [diffs.buffer, index.buffer, stats.buffer, firstThumb.buffer, lastThumb.buffer] };
}

// ---------------------------------------------------------------- encoding

function encConfig(job, rc, firstPass) {
  return {
    src_w: job.srcW, src_h: job.srcH, out_w: job.out.w, out_h: job.out.h, rung_w: job.rung.w, rung_h: job.rung.h,
    fps_num: job.fpsNum, fps_den: job.fpsDen, preset: job.preset, rc, first_pass: firstPass,
  };
}

async function runPass(job, rc, firstPass) {
  check(callJson(ex.sm_enc_begin, encConfig(job, rc, firstPass)), "encoder");
  let scaleMse = 0;
  let scaleN = 0;
  const filtered = job.rung.w !== job.out.w || job.rung.h !== job.out.h;
  await decodeRange(job, job.start, job.end, async (k, frame) => {
    await loadFrame(frame);
    // Sample what the prefilter removes (every 8th frame), so the reported
    // quality is against the real source, not the encoder's input.
    if (!firstPass && filtered && (k - job.start) % 8 === 0) {
      scaleMse += ex.sm_prefilter_mse(job.srcW, job.srcH, job.out.w, job.out.h, job.rung.w, job.rung.h);
      scaleN++;
    }
    check(ex.sm_enc_push(), "encode");
    if ((k - job.start) % 10 === 9) postMessage({ type: "progress", frames: 10, pass: firstPass ? 1 : 2 });
  });
  check(ex.sm_enc_end(), "encoder end");
  const r = result();
  const hl = new DataView(r.buffer).getUint32(0, true);
  const head = JSON.parse(td.decode(r.subarray(4, 4 + hl)));
  head.scale_mse = scaleN ? scaleMse / scaleN : 0;
  return { head, data: r.slice(4 + hl) };
}

const kbpsFor = (bytes, frames, job) => Math.max(1, Math.round((bytes * 8) / (frames * job.fpsDen / job.fpsNum) / 1000));

let awaitingTarget = null;

async function encode(job) {
  const frames = job.end - job.start;
  const stats = `/c${job.chunk}`;
  if (job.stats) {
    // Re-run of the second pass with stats from an earlier first pass.
    wasi.fs.write(stats.slice(1), job.stats.log);
    wasi.fs.write(stats.slice(1) + ".mbtree", job.stats.mbtree);
  } else {
    await runPass(job, `bitrate=${kbpsFor(job.planned, frames, job)}:pass=1:stats=${stats}`, true);
    postMessage({ type: "pass1done", chunk: job.chunk });
    job.target = await new Promise((r) => (awaitingTarget = r));
  }
  const res = await runPass(job, `bitrate=${kbpsFor(job.target, frames, job)}:pass=2:stats=${stats}`, false);
  const log = wasi.fs.read(stats.slice(1));
  const mbtree = wasi.fs.read(stats.slice(1) + ".mbtree");
  wasi.fs.remove(stats.slice(1));
  wasi.fs.remove(stats.slice(1) + ".mbtree");
  return {
    head: res.head, data: res.data, stats: { log, mbtree },
    transfer: [res.data.buffer, log.buffer, mbtree.buffer],
  };
}

// ------------------------------------------------------------------- audio

async function audio(job) {
  const { file, track, samples, t0, duration, bitrate } = job;
  const rate = track.sample_rate;
  const channels = track.channels || 2;
  const la = check(ex.sm_opus_begin(channels, rate, bitrate), "opus");
  const total = Math.round(duration * rate);
  let pos = 0; // samples pushed so far (relative to t0)
  let failure = null;
  const queue = [];
  const decoder = new AudioDecoder({ output: (d) => queue.push(d), error: (e) => (failure = e) });
  const config = { codec: track.codec, sampleRate: rate, numberOfChannels: channels };
  if (track.description) config.description = hexToBytes(track.description);
  decoder.configure(config);

  let scratch = { ptr: 0, size: 0 };
  const pushPlanar = (planes, frames) => {
    const size = planes.length * frames * 4;
    if (scratch.size < size) {
      if (scratch.ptr) ex.sm_free(scratch.ptr, scratch.size);
      scratch = { ptr: ex.sm_alloc(size), size };
    }
    const f = new Float32Array(ex.memory.buffer, scratch.ptr, planes.length * frames);
    planes.forEach((p, c) => f.set(p.subarray(0, frames), c * frames));
    check(ex.sm_opus_push(scratch.ptr, planes.length, frames), "opus");
  };
  const drain = () => {
    while (queue.length) {
      const d = queue.shift();
      const ch = d.numberOfChannels;
      const n = d.numberOfFrames;
      // Align to the video start: pad gaps with silence, drop overlap.
      const at = Math.round((d.timestamp / 1e6 - t0) * rate);
      if (at > pos && pos < total) {
        const gap = Math.min(at - pos, total - pos);
        pushPlanar(Array.from({ length: ch }, () => new Float32Array(gap)), gap);
        pos += gap;
      }
      const skip = Math.max(0, pos - at);
      const take = Math.min(n - skip, total - pos);
      if (take > 0) {
        const planes = [];
        for (let c = 0; c < ch; c++) {
          const p = new Float32Array(n);
          d.copyTo(p, { planeIndex: c, format: "f32-planar" });
          planes.push(p.subarray(skip, skip + take));
        }
        pushPlanar(planes, take);
        pos += take;
      }
      d.close();
    }
  };

  const n = samples.length / 5;
  for (let i = 0; i < n; ) {
    if (failure) throw failure;
    let j = i;
    const start = samples[5 * i];
    let end = start;
    while (j < n && samples[5 * j] >= start && samples[5 * j] + samples[5 * j + 1] - start < 4e6 && j - i < 512) {
      end = Math.max(end, samples[5 * j] + samples[5 * j + 1]);
      j++;
    }
    const buf = new Uint8Array(await file.slice(start, end).arrayBuffer());
    for (; i < j; i++) {
      const off = samples[5 * i] - start;
      decoder.decode(new EncodedAudioChunk({
        type: "key", timestamp: Math.round(samples[5 * i + 2] * 1e6), data: buf.subarray(off, off + samples[5 * i + 1]),
      }));
    }
    while (decoder.decodeQueueSize > 32) await new Promise((r) => setTimeout(r, 1));
    drain();
    postMessage({ type: "progress", audio: i / n });
    if (pos >= total) break;
  }
  if (failure) throw failure;
  await decoder.flush();
  drain();
  decoder.close();
  if (pos < total) pushPlanar(Array.from({ length: channels }, () => new Float32Array(total - pos)), total - pos);
  return opusResult(rate, la);
}

/** Opus from PCM decoded elsewhere (decodeAudioData fallback), 48 kHz planar. */
async function audioPcm(job) {
  const la = check(ex.sm_opus_begin(job.planes.length, 48000, job.bitrate), "opus");
  const step = 48000;
  const len = job.planes[0].length;
  for (let s = 0; s < len; s += step) {
    const n = Math.min(step, len - s);
    const size = job.planes.length * n * 4;
    const p = ex.sm_alloc(size);
    const f = new Float32Array(ex.memory.buffer, p, job.planes.length * n);
    job.planes.forEach((pl, c) => f.set(pl.subarray(s, s + n), c * n));
    check(ex.sm_opus_push(p, job.planes.length, n), "opus");
    ex.sm_free(p, size);
  }
  return opusResult(48000, la);
}

function opusResult(rate) {
  check(ex.sm_opus_end(rate), "opus end");
  const r = result();
  const hl = new DataView(r.buffer).getUint32(0, true);
  const head = JSON.parse(td.decode(r.subarray(4, 4 + hl)));
  const data = r.slice(4 + hl);
  return { head, data, transfer: [data.buffer] };
}

// ---------------------------------------------------------------- dispatch

/** Diagnostics: decoded source-resolution luma of output frames [a, b),
 *  via the wasm conversion or (job.raw) straight from copyTo. */
async function dump(job) {
  const planes = [];
  await decodeRange(job, job.a, job.b, async (k, frame) => {
    if (job.raw) {
      const buf = new Uint8Array(frame.allocationSize());
      const layout = await frame.copyTo(buf);
      const w = frame.visibleRect.width;
      const h = frame.visibleRect.height;
      const y = new Uint8Array(w * h);
      for (let r = 0; r < h; r++) y.set(buf.subarray(layout[0].offset + r * layout[0].stride, layout[0].offset + r * layout[0].stride + w), r * w);
      if (k === job.a) console.log(`[dump] format ${frame.format} layout ${JSON.stringify(layout)} size ${buf.length}`);
      planes.push(y);
      return;
    }
    await loadFrame(frame);
    check(ex.sm_input_luma(), "dump");
    planes.push(result());
  });
  const all = new Uint8Array(planes.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of planes) {
    all.set(p, o);
    o += p.length;
  }
  return { luma: all, transfer: [all.buffer] };
}

const handlers = { analyze, encode, audio, audioPcm, dump };

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "target") {
    awaitingTarget?.(msg.bytes);
    awaitingTarget = null;
    return;
  }
  try {
    if (msg.type === "init") {
      await init(msg.module);
      postMessage({ type: "ready" });
      return;
    }
    const r = await handlers[msg.type](msg.job);
    const { transfer = [], ...rest } = r;
    postMessage({ type: "done", result: rest }, transfer);
  } catch (err) {
    postMessage({ type: "error", message: String(err?.message ?? err), stack: err?.stack });
  }
};
