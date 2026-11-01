// Encoder speed of the wasm build per preset, as the app runs it (two-pass
// per chunk, one encoder per instance), for the app's time estimates.
// Usage: node tools/node_speed.mjs <module.wasm> <frames.yuv> W H <preset,...> [kbps]
//   presets: x264 names (medium, ...) or svtN / svtN-10
import fs from "node:fs";
import { WASI } from "../web/wasi.js";

const [wasmPath, yuvPath, W, H, presetList, KBPS] = process.argv.slice(2);
const w = +W, h = +H, fsize = w * h * 3 / 2;
const kbps = +(KBPS || 1000);
const yuv = fs.readFileSync(yuvPath);
const n = Math.floor(yuv.length / fsize);
const module = new WebAssembly.Module(fs.readFileSync(wasmPath));
const enc = new TextEncoder(), dec = new TextDecoder();

function instance() {
  const wasi = new WASI();
  const inst = new WebAssembly.Instance(module, wasi.imports);
  wasi.attach(inst);
  inst.exports._initialize();
  return inst.exports;
}

for (const preset of presetList.split(",")) {
  const ex = instance();
  const mem = () => new Uint8Array(ex.memory.buffer);
  const put = (bytes) => { const p = ex.sm_alloc(bytes.length); mem().set(bytes, p); return [p, bytes.length]; };
  const result = () => mem().slice(ex.sm_result_ptr(), ex.sm_result_ptr() + ex.sm_result_len());
  const [fmtP, fmtL] = put(enc.encode("I420"));
  const [layP] = put(new Uint8Array(new Uint32Array([0, w, w * h, w / 2, w * h + w * h / 4, w / 2]).buffer));
  const frameP = ex.sm_alloc(fsize);
  const av1 = preset.startsWith("svt");
  const run = (rc, firstPass) => {
    const cfg = { src_w: w, src_h: h, out_w: w, out_h: h, rung_w: w, rung_h: h, fps_num: 30, fps_den: 1,
                  preset, rc, first_pass: firstPass };
    const [cp, cl] = put(enc.encode(JSON.stringify(cfg)));
    if (ex.sm_enc_begin(cp, cl) < 0) throw new Error(dec.decode(result()));
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      mem().set(yuv.subarray(i * fsize, (i + 1) * fsize), frameP);
      if (ex.sm_frame(fmtP, fmtL, w, h, frameP, fsize, layP, 3) < 0) throw new Error(dec.decode(result()));
      if (ex.sm_enc_push() < 0) throw new Error("push failed");
    }
    if (ex.sm_enc_end() < 0) throw new Error(dec.decode(result()));
    const dt = (performance.now() - t0) / 1000;
    const r = result();
    const hl = new DataView(r.buffer).getUint32(0, true);
    const head = JSON.parse(dec.decode(r.subarray(4, 4 + hl)));
    const stats = r.subarray(r.length - head.stats_len);
    return { dt, head, stats };
  };
  const p1 = run(av1 ? `tbr=${kbps * 1000}b` : `bitrate=${kbps}:pass=1:stats=/s.log`, true);
  if (av1) {
    const [sp, sl] = put(p1.stats);
    ex.sm_set_stats(sp, sl);
  }
  const p2 = run(av1 ? `tbr=${kbps * 1000}b` : `bitrate=${kbps}:pass=2:stats=/s.log`, false);
  const target = kbps * 1000 / 8 * n / 30;
  const psnr = p2.head.mse > 0 ? (10 * Math.log10(65025 / p2.head.mse)).toFixed(2) : "n/a";
  console.log(JSON.stringify({
    preset, w, h, frames: n, pass1_fps: +(n / p1.dt).toFixed(2), pass2_fps: +(n / p2.dt).toFixed(2),
    bytes: p2.head.bytes, size_err: +((p2.head.bytes / target - 1) * 100).toFixed(1), psnr: +psnr,
    memory_mb: Math.round(ex.memory.buffer.byteLength / 1048576),
  }));
}
