// Smoke test of the wasm build under Node's WASI: two-pass x264 on raw
// I420 frames. Usage: node tools/node_smoke.mjs <module.wasm> <frames.yuv> W H
import fs from "node:fs";
import { WASI } from "../web/wasi.js";

const [wasmPath, yuvPath, W, H] = process.argv.slice(2);
const w = +W, h = +H, fsize = w * h * 3 / 2;
const wasi = new WASI();
const mod = new WebAssembly.Module(fs.readFileSync(wasmPath));
const inst = new WebAssembly.Instance(mod, wasi.imports);
wasi.attach(inst);
const ex = inst.exports;
ex._initialize();
const enc = new TextEncoder(), dec = new TextDecoder();
const mem = () => new Uint8Array(ex.memory.buffer);
const put = (bytes) => { const p = ex.sm_alloc(bytes.length); mem().set(bytes, p); return [p, bytes.length]; };
const result = () => mem().slice(ex.sm_result_ptr(), ex.sm_result_ptr() + ex.sm_result_len());

const yuv = fs.readFileSync(yuvPath);
const n = Math.floor(yuv.length / fsize);
const [fmtP, fmtL] = put(enc.encode("I420"));
const layout = new Uint32Array([0, w, w * h, w / 2, w * h + w * h / 4, w / 2]);
const [layP] = put(new Uint8Array(layout.buffer));
const frameP = ex.sm_alloc(fsize);

function run(rc, firstPass, rung) {
  const cfg = { src_w: w, src_h: h, out_w: w, out_h: h, rung_w: rung[0], rung_h: rung[1], fps_num: 30, fps_den: 1,
                preset: "medium", rc, first_pass: firstPass };
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
  return { dt, head, data: r.subarray(4 + hl) };
}

const kbps = +(process.env.KBPS || 1500);
for (const rung of [[w, h], [Math.round(w * 0.6 / 2) * 2, Math.round(h * 0.6 / 2) * 2]]) {
  const p1 = run(`bitrate=${kbps}:pass=1:stats=/s.log`, true, rung);
  const p2 = run(`bitrate=${kbps}:pass=2:stats=/s.log`, false, rung);
  const target = kbps * 1000 / 8 * n / 30;
  console.log(`rung ${rung.join("x")}: pass1 ${(n / p1.dt).toFixed(1)} fps, pass2 ${(n / p2.dt).toFixed(1)} fps; ` +
    `${p2.head.packets.length} packets, ${p2.head.bytes} bytes vs target ${target.toFixed(0)} ` +
    `(${((p2.head.bytes / target - 1) * 100).toFixed(1)}%), x264 PSNR ${(10 * Math.log10(65025 / p2.head.mse)).toFixed(2)} dB, ` +
    `stats files: ${[...wasi.fs.files.keys()].join(",")}`);
  if (rung[0] === w) {
    // Write an Annex-B stream for ffmpeg to check.
    const parts = [];
    const sc = Buffer.from([0, 0, 0, 1]);
    parts.push(sc, Buffer.from(p2.head.sps, "hex"), sc, Buffer.from(p2.head.pps, "hex"));
    let o = 0;
    for (const [size] of p2.head.packets) {
      const pk = p2.data.subarray(o, o + size); o += size;
      for (let i = 0; i < pk.length;) { const l = new DataView(pk.buffer, pk.byteOffset + i).getUint32(0); parts.push(sc, Buffer.from(pk.subarray(i + 4, i + 4 + l))); i += 4 + l; }
    }
    fs.writeFileSync(yuvPath + ".264", Buffer.concat(parts));
  }
}
