//! WebAssembly entry points (plain C ABI; loaded by web/wasi.js).
//!
//! The same module serves the main thread (demux, plan, control, mux) and
//! the workers (analysis, x264 encodes, Opus). Calls return an i32 status
//! (>= 0 ok, < 0 error); any payload, or the error text, is left in a result
//! buffer that JS reads with sm_result_ptr/len.

use codecs::{Opus, X264, X264Config};
use serde::Deserialize;
use serde_json::json;
use smallifier_core::analysis::{Analyzer, FrameStats, Thumb};
use smallifier_core::media::TrackInfo;
use smallifier_core::mkv::MkvParser;
use smallifier_core::model::{self, ChunkStats, Gain};
use smallifier_core::plan::{self, Controller, Rung};
use smallifier_core::yuv::{self, Frame, FrameScaler, PixelFormat, PlaneLayout, Prefilter};
use smallifier_core::{mp4, mux, scene};
use std::cell::RefCell;

#[derive(Default)]
struct State {
    result: Vec<u8>,
    tracks: Vec<TrackInfo>,
    mkv: Option<MkvParser>,
    controller: Option<Controller>,
    input: Option<Frame>,
    ana: Option<AnaSession>,
    enc: Option<EncSession>,
    stats: Vec<u8>,
    opus: Option<OpusSession>,
}

thread_local! {
    static ST: RefCell<State> = RefCell::new(State::default());
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    ST.with(|s| f(&mut s.borrow_mut()))
}

fn ok(bytes: Vec<u8>) -> i32 {
    with(|s| s.result = bytes);
    0
}

fn err(msg: impl Into<String>) -> i32 {
    with(|s| s.result = msg.into().into_bytes());
    -1
}

fn json_out(v: serde_json::Value) -> i32 {
    ok(serde_json::to_vec(&v).unwrap())
}

unsafe fn slice<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

// ------------------------------------------------------------- reactor init

unsafe extern "C" {
    fn __wasm_call_ctors();
}

/// WASI "reactor" initialisation: JS calls this once after instantiation.
///
/// Referencing `__wasm_call_ctors` here also stops wasm-ld from treating
/// the module as a "command" and wrapping every export with constructor
/// and destructor calls; those destructors run libc's `__stdio_exit` after
/// each call, which rewinds open input files (x264's two-pass stats) by
/// their buffered bytes every time and corrupts the reads.
#[unsafe(no_mangle)]
pub extern "C" fn _initialize() {
    unsafe { __wasm_call_ctors() }
}

// ------------------------------------------------------------------ memory

#[unsafe(no_mangle)]
pub extern "C" fn sm_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr`/`len` must come from `sm_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_result_ptr() -> *const u8 {
    with(|s| s.result.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_result_len() -> usize {
    with(|s| s.result.len())
}

// ----------------------------------------------------------------- demuxing

fn tracks_json(tracks: &[TrackInfo]) -> i32 {
    json_out(json!({ "tracks": tracks }))
}

/// `moofs`: repeated [u64 LE file offset][u32 LE length][payload].
///
/// # Safety
/// Pointers must reference `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_mp4_parse(moov: *const u8, moov_len: usize, moofs: *const u8, moofs_len: usize) -> i32 {
    let moov = unsafe { slice(moov, moov_len) };
    let blob = unsafe { slice(moofs, moofs_len) };
    let mut list = Vec::new();
    let mut p = 0;
    while p + 12 <= blob.len() {
        let off = u64::from_le_bytes(blob[p..p + 8].try_into().unwrap());
        let n = u32::from_le_bytes(blob[p + 8..p + 12].try_into().unwrap()) as usize;
        list.push((off, &blob[p + 12..p + 12 + n]));
        p += 12 + n;
    }
    match mp4::parse(moov, &list) {
        Ok(t) => {
            let r = tracks_json(&t);
            with(|s| s.tracks = t);
            r
        }
        Err(e) => err(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_mkv_new() {
    with(|s| s.mkv = Some(MkvParser::new()));
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_mkv_want() -> f64 {
    with(|s| s.mkv.as_ref().map(|m| m.want() as f64).unwrap_or(-1.0))
}

/// Returns 1 when indexing is complete (tracks JSON in the result).
///
/// # Safety
/// `ptr` must reference `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_mkv_feed(at: f64, ptr: *const u8, len: usize, eof: i32) -> i32 {
    let data = unsafe { slice(ptr, len) };
    let done = with(|s| {
        let m = s.mkv.as_mut().unwrap();
        m.feed(at as u64, data, eof != 0);
        m.done()
    });
    if !done {
        return 0;
    }
    let t = with(|s| s.mkv.take().unwrap().tracks());
    tracks_json(&t);
    with(|s| s.tracks = t);
    1
}

/// Packed sample table of track `i`: per sample 5 f64 (offset, size, pts,
/// dts, keyframe), in decode order.
#[unsafe(no_mangle)]
pub extern "C" fn sm_track_samples(i: usize) -> i32 {
    let out = with(|s| {
        s.tracks.get(i).map(|t| {
            let mut v = Vec::with_capacity(t.samples.len() * 40);
            for x in &t.samples {
                for f in [x.offset as f64, x.size as f64, x.pts, x.dts, x.keyframe as u8 as f64] {
                    v.extend_from_slice(&f.to_le_bytes());
                }
            }
            v
        })
    });
    match out {
        Some(v) => ok(v),
        None => err("no such track"),
    }
}

// ------------------------------------------------------------ frame input

/// Converts a VideoFrame.copyTo() buffer into the current input frame.
/// `layout` holds `nplanes` (offset, stride) u32 pairs.
///
/// # Safety
/// Pointers must reference the given number of readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_frame(fmt: *const u8, fmt_len: usize, w: usize, h: usize, buf: *const u8, buf_len: usize,
                                  layout: *const u32, nplanes: usize) -> i32 {
    let fmt = String::from_utf8_lossy(unsafe { slice(fmt, fmt_len) }).to_string();
    let Some(pf) = PixelFormat::parse(&fmt) else { return err(format!("unsupported pixel format {fmt}")) };
    let lay = unsafe { std::slice::from_raw_parts(layout, nplanes * 2) };
    let layout: Vec<PlaneLayout> =
        (0..nplanes).map(|i| PlaneLayout { offset: lay[2 * i] as usize, stride: lay[2 * i + 1] as usize }).collect();
    let buf = unsafe { slice(buf, buf_len) };
    let f = yuv::convert(pf, w, h, buf, &layout);
    with(|s| s.input = Some(f));
    0
}

/// Diagnostics: the current input frame's luma plane (as converted).
#[unsafe(no_mangle)]
pub extern "C" fn sm_input_luma() -> i32 {
    match with(|s| s.input.as_ref().map(|f| f.y.data.clone())) {
        Some(v) => ok(v),
        None => err("no input frame"),
    }
}

// ---------------------------------------------------------------- analysis

struct AnaSession {
    scaler: Option<FrameScaler>,
    analyzer: Analyzer,
    prev: Option<Frame>,
    first_thumb: Option<Thumb>,
    prev_thumb: Option<Thumb>,
    diffs: Vec<f32>,
    sampled: Vec<(u32, FrameStats)>,
    counted: u32,
}

fn to_output(scaler: &Option<FrameScaler>, f: Frame) -> Frame {
    match scaler {
        Some(s) => s.scale(&f),
        None => f,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_ana_begin(src_w: usize, src_h: usize, out_w: usize, out_h: usize) {
    let scaler = ((src_w, src_h) != (out_w, out_h)).then(|| FrameScaler::new(src_w, src_h, out_w, out_h));
    with(|s| {
        s.ana = Some(AnaSession {
            scaler,
            analyzer: Analyzer::new(out_w, out_h),
            prev: None,
            first_thumb: None,
            prev_thumb: None,
            diffs: Vec::new(),
            sampled: Vec::new(),
            counted: 0,
        })
    });
}

/// Feeds the current input frame. flags: 1 = analyse it (against the
/// previous kept frame), 2 = it belongs to the segment (record its
/// scene-cut difference), 4 = keep it, scaled, as the predecessor of the
/// next frame. A warm-up frame before the segment passes 4 only.
#[unsafe(no_mangle)]
pub extern "C" fn sm_ana_push(flags: u32) -> i32 {
    with(|s| {
        let Some(input) = s.input.take() else { return -1 };
        let a = s.ana.as_mut().unwrap();
        // Thumbnails are box averages, so the unscaled frame will do.
        let thumb = Thumb::of(&input);
        if flags & 2 != 0 {
            a.diffs.push(a.prev_thumb.as_ref().map(|p| thumb.diff(p)).unwrap_or(0.0));
            if a.first_thumb.is_none() {
                a.first_thumb = Some(thumb.clone());
            }
        }
        let scaled = (flags & 5 != 0).then(|| to_output(&a.scaler, input));
        if flags & 3 == 3 {
            let cur = scaled.as_ref().unwrap();
            let st = a.analyzer.analyze(a.prev.as_ref().map(|p| &p.y), &cur.y);
            a.sampled.push((a.counted, st));
        }
        if flags & 2 != 0 {
            a.counted += 1;
        }
        a.prev_thumb = Some(thumb);
        a.prev = if flags & 4 != 0 { scaled } else { None };
        0
    })
}

/// Result: [n u32][n f32 diffs][m u32][m x (u32 index, FrameStats)]
/// [t u32][first thumbnail, t bytes][last thumbnail, t bytes]. The
/// thumbnails let the caller compute the scene-cut difference across
/// segment boundaries.
#[unsafe(no_mangle)]
pub extern "C" fn sm_ana_end() -> i32 {
    let Some(a) = with(|s| s.ana.take()) else { return err("no analysis session") };
    let mut out = Vec::new();
    out.extend_from_slice(&(a.diffs.len() as u32).to_le_bytes());
    for d in &a.diffs {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&(a.sampled.len() as u32).to_le_bytes());
    for (i, st) in &a.sampled {
        out.extend_from_slice(&i.to_le_bytes());
        out.extend_from_slice(&st.to_bytes());
    }
    let empty = Thumb(Vec::new());
    let (first, last) = (a.first_thumb.as_ref().unwrap_or(&empty), a.prev_thumb.as_ref().unwrap_or(&empty));
    out.extend_from_slice(&(first.0.len() as u32).to_le_bytes());
    out.extend_from_slice(&first.0);
    out.extend_from_slice(&last.0);
    ok(out)
}

/// Scene-cut difference between two thumbnails from sm_ana_end.
///
/// # Safety
/// Pointers must reference `len` readable bytes each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_thumb_diff(a: *const u8, b: *const u8, len: usize) -> f32 {
    let (a, b) = unsafe { (slice(a, len), slice(b, len)) };
    if a.is_empty() {
        return 0.0;
    }
    Thumb(a.to_vec()).diff(&Thumb(b.to_vec()))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_stats_size() -> usize {
    FrameStats::BYTES
}

// ---------------------------------------------------------------- planning

#[derive(Deserialize)]
struct PlanIn {
    frames: usize,
    fps: f64,
    out_w: usize,
    out_h: usize,
    /// Prefilter rungs, ascending; the last must equal the output size.
    rungs: Vec<(usize, usize)>,
    diffs: Vec<f32>,
    /// Output frame index of each stats record.
    sampled: Vec<usize>,
    budget: f64,
    max_chunk_frames: usize,
    min_scene_sec: f64,
    #[serde(default)]
    preset: String,
}

/// # Safety
/// Pointers must reference the given number of readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_plan(json_ptr: *const u8, json_len: usize, stats_ptr: *const u8, stats_len: usize) -> i32 {
    let inp: PlanIn = match serde_json::from_slice(unsafe { slice(json_ptr, json_len) }) {
        Ok(v) => v,
        Err(e) => return err(format!("plan input: {e}")),
    };
    let blob = unsafe { slice(stats_ptr, stats_len) };
    if blob.len() != inp.sampled.len() * FrameStats::BYTES {
        return err("stats blob size mismatch");
    }
    if inp.sampled.is_empty() || inp.frames == 0 || inp.rungs.is_empty() {
        return err("nothing was analysed (clip too short?)");
    }
    let stats: Vec<FrameStats> =
        blob.chunks(FrameStats::BYTES).map(|b| FrameStats::from_bytes(b).unwrap()).collect();

    let cuts = scene::detect_cuts(&inp.diffs, inp.fps, inp.min_scene_sec);
    let chunks = scene::chunk_scenes(&cuts, inp.frames, inp.max_chunk_frames);
    let cuts_set: std::collections::HashSet<usize> = chunks.iter().map(|c| c.start).collect();

    let mut chunk_stats = Vec::with_capacity(chunks.len());
    for c in &chunks {
        let inside: Vec<usize> = (0..stats.len()).filter(|&i| inp.sampled[i] >= c.start && inp.sampled[i] < c.end).collect();
        // The I-frame proxy is the first sampled frame of the chunk (frames
        // in a shot look alike); P statistics exclude frames that sit on a
        // chunk start, whose inter residual spans a cut.
        let first = match inside.first() {
            Some(&i) => &stats[i],
            None => {
                // Chunk shorter than the sampling interval: nearest sample.
                let i = (0..stats.len()).min_by_key(|&i| inp.sampled[i].abs_diff(c.start)).unwrap();
                &stats[i]
            }
        };
        let rest: Vec<&FrameStats> =
            inside.iter().filter(|&&i| !cuts_set.contains(&inp.sampled[i])).map(|&i| &stats[i]).collect();
        chunk_stats.push(ChunkStats::build(c.len() as u32, first, &rest));
    }
    let rungs: Vec<Rung> = inp
        .rungs
        .iter()
        .map(|&(w, h)| Rung {
            w,
            h,
            gain: if (w, h) == (inp.out_w, inp.out_h) { Gain::identity() } else { Gain::for_rung(inp.out_w, inp.out_h, w, h) },
        })
        .collect();
    let m = model::model_for(&inp.preset);
    let p = plan::allocate(m, &chunks, &chunk_stats, &rungs, inp.budget);
    let frames: Vec<usize> = chunks.iter().map(|c| c.len()).collect();
    with(|s| s.controller = Some(Controller::new(inp.budget, p.chunk_bytes.clone(), frames, m)));
    json_out(json!({ "chunks": chunks, "plan": p, "cuts": cuts }))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_ctl_next() -> i32 {
    with(|s| s.controller.as_mut().and_then(|c| c.next_chunk()).map(|i| i as i32).unwrap_or(-1))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_ctl_target(i: usize) -> f64 {
    with(|s| s.controller.as_mut().unwrap().second_pass_target(i))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_ctl_finished(i: usize, bytes: f64) {
    with(|s| s.controller.as_mut().unwrap().finished(i, bytes));
}

/// Re-runs that remove `excess` bytes from the output (negative: add).
#[unsafe(no_mangle)]
pub extern "C" fn sm_ctl_fixups(excess: f64, workers: usize) -> i32 {
    let f = with(|s| s.controller.as_mut().unwrap().fixups(excess, workers));
    json_out(json!(f))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_ctl_total() -> f64 {
    with(|s| s.controller.as_ref().map(|c| c.total()).unwrap_or(0.0))
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_mux_overhead(frames: usize, fps: f64, audio_packets: usize, has_audio: i32) -> f64 {
    mux::overhead_estimate(frames, fps, audio_packets, has_audio != 0) as f64
}

// ----------------------------------------------------------------- encoding

#[derive(Deserialize)]
struct EncIn {
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
    rung_w: usize,
    rung_h: usize,
    fps_num: u32,
    fps_den: u32,
    /// x264 preset ("medium", ...) or "svtN" for SVT-AV1 preset N.
    preset: String,
    /// Rate control and pass options: x264 "bitrate=900:pass=1:stats=/s0";
    /// SVT-AV1 "tbr=900000b" (the pass comes from `first_pass`, and the
    /// second pass takes the statistics given to sm_set_stats).
    rc: String,
    first_pass: bool,
}

enum Encoder {
    X264(X264),
    #[cfg(feature = "av1")]
    Av1(codecs::av1::Svt),
}

struct OutPacket {
    data: Vec<u8>,
    pts: i64,
    key: bool,
    psnr_y: f64,
}

struct EncSession {
    enc: Encoder,
    scaler: Option<FrameScaler>,
    prefilter: Option<Prefilter>,
    packets: Vec<OutPacket>,
    frames: i64,
    keep: bool,
}

/// First-pass statistics for the next SVT-AV1 second pass.
///
/// # Safety
/// `ptr` must reference `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_set_stats(ptr: *const u8, len: usize) {
    let v = unsafe { slice(ptr, len) }.to_vec();
    with(|s| s.stats = v);
}

fn open_encoder(e: &EncIn) -> Result<Encoder, String> {
    if let Some(_p) = model::svt_preset(&e.preset) {
        #[cfg(feature = "av1")]
        {
            let stats = if e.first_pass { Vec::new() } else { with(|s| std::mem::take(&mut s.stats)) };
            let cfg = codecs::av1::SvtConfig {
                width: e.out_w as u32,
                height: e.out_h as u32,
                fps_num: e.fps_num,
                fps_den: e.fps_den,
                opts: model::svt_opts(_p, &format!("rc=1:{}:gop-constraint-rc=1", e.rc)),
                pass: if e.first_pass { 1 } else { 2 },
                stats,
            };
            return codecs::av1::Svt::open(&cfg).map(Encoder::Av1);
        }
        #[cfg(not(feature = "av1"))]
        return Err("this module was built without AV1".into());
    }
    let mut opts = format!("{}:{}", model::x264_opts(&e.preset), e.rc);
    if e.first_pass {
        opts.push_str(":fastfirstpass");
    }
    let cfg = X264Config {
        width: e.out_w as u32,
        height: e.out_h as u32,
        fps_num: e.fps_num,
        fps_den: e.fps_den,
        preset: e.preset.clone(),
        tune: String::new(),
        opts,
        stats: !e.first_pass,
    };
    X264::open(&cfg).map(Encoder::X264)
}

/// # Safety
/// `ptr` must reference `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_enc_begin(ptr: *const u8, len: usize) -> i32 {
    let e: EncIn = match serde_json::from_slice(unsafe { slice(ptr, len) }) {
        Ok(v) => v,
        Err(e) => return err(format!("encode input: {e}")),
    };
    let enc = match open_encoder(&e) {
        Ok(x) => x,
        Err(m) => return err(m),
    };
    let scaler = ((e.src_w, e.src_h) != (e.out_w, e.out_h)).then(|| FrameScaler::new(e.src_w, e.src_h, e.out_w, e.out_h));
    let prefilter = Prefilter::new(e.out_w, e.out_h, e.rung_w, e.rung_h);
    with(|s| s.enc = Some(EncSession { enc, scaler, prefilter, packets: Vec::new(), frames: 0, keep: !e.first_pass }));
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_enc_push() -> i32 {
    with(|s| {
        let Some(input) = s.input.take() else { return -1 };
        let e = s.enc.as_mut().unwrap();
        let f = to_output(&e.scaler, input);
        let f = match &e.prefilter {
            Some(p) => p.apply(&f),
            None => f,
        };
        let pts = e.frames;
        e.frames += 1;
        match &mut e.enc {
            Encoder::X264(x) => match x.encode(&f.y.data, &f.u.data, &f.v.data, f.y.w, f.u.w, pts, pts == 0) {
                Ok(Some(p)) => {
                    let data = if e.keep { p.data } else { Vec::new() };
                    e.packets.push(OutPacket { data, pts: p.pts, key: p.keyframe, psnr_y: p.psnr_y });
                    0
                }
                Ok(None) => 0,
                Err(_) => -1,
            },
            #[cfg(feature = "av1")]
            Encoder::Av1(a) => match a.send(&f.y.data, &f.u.data, &f.v.data, f.y.w, f.u.w, pts) {
                Ok(()) => 0,
                Err(_) => -1,
            },
        }
    })
}

/// Result: [json len u32][json][packet data...][first-pass stats...]; json
/// has the codec, its decoder configuration (H.264 sps/pps, or the AV1
/// sequence header) as hex, per-packet [size, pts, key], the mean luma MSE
/// the encoder measured against its (prefiltered) input (x264 only; -1 for
/// AV1) and the length of the trailing first-pass statistics (AV1 only).
#[unsafe(no_mangle)]
pub extern "C" fn sm_enc_end() -> i32 {
    let Some(mut e) = with(|s| s.enc.take()) else { return err("no encode session") };
    let hex = |v: &[u8]| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
    #[cfg_attr(not(feature = "av1"), allow(unused_mut))]
    let mut stats = Vec::new();
    let config = match &mut e.enc {
        Encoder::X264(x) => {
            loop {
                match x.flush() {
                    Ok(Some(p)) => {
                        let data = if e.keep { p.data } else { Vec::new() };
                        e.packets.push(OutPacket { data, pts: p.pts, key: p.keyframe, psnr_y: p.psnr_y });
                    }
                    Ok(None) => break,
                    Err(m) => return err(m),
                }
            }
            json!({ "codec": "h264", "sps": hex(&x.sps()), "pps": hex(&x.pps()) })
        }
        #[cfg(feature = "av1")]
        Encoder::Av1(a) => {
            match a.finish() {
                Ok(ps) => {
                    for p in ps {
                        let data = if e.keep { p.data } else { Vec::new() };
                        e.packets.push(OutPacket { data, pts: p.pts, key: p.keyframe, psnr_y: -1.0 });
                    }
                }
                Err(m) => return err(m),
            }
            if !e.keep {
                stats = a.stats();
            }
            json!({ "codec": "av1", "av1_header": hex(&a.header()) })
        }
    };
    let measured = e.packets.iter().all(|p| p.psnr_y > 0.0) && !e.packets.is_empty();
    let mse: f64 = if measured {
        e.packets.iter().map(|p| 65025.0 / 10f64.powf(p.psnr_y / 10.0)).sum::<f64>() / e.packets.len() as f64
    } else {
        -1.0
    };
    let table: Vec<[i64; 3]> = e.packets.iter().map(|p| [p.data.len() as i64, p.pts, p.key as i64]).collect();
    let bytes: usize = e.packets.iter().map(|p| p.data.len()).sum();
    let mut head = config;
    head["packets"] = json!(table);
    head["bytes"] = json!(bytes);
    head["mse"] = json!(mse);
    head["stats_len"] = json!(stats.len());
    let head = serde_json::to_vec(&head).unwrap();
    let mut out = Vec::with_capacity(4 + head.len() + bytes + stats.len());
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    out.extend_from_slice(&head);
    for p in &e.packets {
        out.extend_from_slice(&p.data);
    }
    out.extend_from_slice(&stats);
    ok(out)
}

/// Luma MSE between the current input frame scaled to (w, h) and the same
/// after the rung's prefilter (to report true quality = this + x264's MSE).
#[unsafe(no_mangle)]
pub extern "C" fn sm_prefilter_mse(src_w: usize, src_h: usize, w: usize, h: usize, rw: usize, rh: usize) -> f64 {
    with(|s| {
        let Some(input) = s.input.as_ref() else { return -1.0 };
        let f = if (src_w, src_h) != (w, h) { FrameScaler::new(src_w, src_h, w, h).scale(input) } else { input.clone() };
        let Some(p) = Prefilter::new(w, h, rw, rh) else { return 0.0 };
        let g = p.apply(&f);
        f.y.data.iter().zip(&g.y.data).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / f.y.data.len() as f64
    })
}

// ------------------------------------------------------------------- audio

struct OpusSession {
    enc: Opus,
    channels: usize,
    resampler: Resampler,
    pending: Vec<f32>,
    packets: Vec<Vec<u8>>,
    input_frames: u64,
}

/// Windowed-sinc resampler to 48 kHz for interleaved audio.
struct Resampler {
    ratio: f64,
    channels: usize,
    taps: usize,
    history: Vec<f32>,
    pos: f64,
}

impl Resampler {
    fn new(rate: u32, channels: usize) -> Self {
        Resampler { ratio: rate as f64 / 48000.0, channels, taps: 16, history: Vec::new(), pos: 0.0 }
    }

    fn kernel(&self, x: f64) -> f64 {
        let cutoff = (1.0 / self.ratio).min(1.0) * 0.97;
        let t = self.taps as f64;
        if x.abs() >= t {
            return 0.0;
        }
        // Low-pass impulse response cutoff * sinc(cutoff * x): unity DC gain.
        let sinc = if x.abs() < 1e-9 { cutoff } else { (std::f64::consts::PI * x * cutoff).sin() / (std::f64::consts::PI * x) };
        let w = 0.5 + 0.5 * (std::f64::consts::PI * x / t).cos();
        sinc * w
    }

    /// Consumes interleaved input, returns interleaved 48 kHz output.
    fn process(&mut self, input: &[f32], flush: bool) -> Vec<f32> {
        let ch = self.channels;
        if (self.ratio - 1.0).abs() < 1e-9 {
            return input.to_vec();
        }
        self.history.extend_from_slice(input);
        if flush {
            self.history.extend(std::iter::repeat_n(0.0, self.taps * 2 * ch));
        }
        let frames = self.history.len() / ch;
        let mut out = Vec::new();
        let t = self.taps as isize;
        loop {
            let centre = self.pos.floor() as isize;
            if centre + t >= frames as isize {
                break;
            }
            for c in 0..ch {
                let mut acc = 0.0;
                for k in (centre - t + 1)..=(centre + t) {
                    if k < 0 {
                        continue;
                    }
                    acc += self.history[k as usize * ch + c] as f64 * self.kernel(self.pos - k as f64);
                }
                out.push(acc as f32);
            }
            self.pos += self.ratio;
        }
        // Drop consumed history, keeping the filter span.
        let keep_from = (self.pos.floor() as isize - t).max(0) as usize;
        self.history.drain(..keep_from * ch);
        self.pos -= keep_from as f64;
        out
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sm_opus_begin(channels: usize, rate: u32, bitrate: u32) -> i32 {
    let out_ch = channels.clamp(1, 2);
    match Opus::open(out_ch, bitrate) {
        Ok(enc) => {
            let la = enc.lookahead() as i32;
            with(|s| {
                s.opus = Some(OpusSession {
                    enc,
                    channels: out_ch,
                    resampler: Resampler::new(rate, out_ch),
                    pending: Vec::new(),
                    packets: Vec::new(),
                    input_frames: 0,
                })
            });
            la
        }
        Err(m) => err(m),
    }
}

/// Planar f32 input (`channels` planes of `frames` samples), as AudioData
/// gives it; more than two channels are downmixed to stereo.
///
/// # Safety
/// `ptr` must reference channels * frames f32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_opus_push(ptr: *const f32, channels: usize, frames: usize) -> i32 {
    let planes = unsafe { std::slice::from_raw_parts(ptr, channels * frames) };
    with(|s| {
        let o = s.opus.as_mut().unwrap();
        let mut inter = Vec::with_capacity(frames * o.channels);
        for i in 0..frames {
            let at = |c: usize| planes[c * frames + i];
            match (channels, o.channels) {
                (_, 1) => inter.push((0..channels).map(at).sum::<f32>() / channels as f32),
                (1, 2) => {
                    inter.push(at(0));
                    inter.push(at(0));
                }
                (2, 2) => {
                    inter.push(at(0));
                    inter.push(at(1));
                }
                (n, _) => {
                    // L/R plus centre and surrounds at -3 dB (ITU-style downmix).
                    let (mut l, mut r) = (at(0), at(1));
                    if n >= 3 {
                        l += 0.707 * at(2);
                        r += 0.707 * at(2);
                    }
                    for c in 4..n {
                        if c % 2 == 0 { l += 0.707 * at(c) } else { r += 0.707 * at(c) }
                    }
                    inter.push(l * 0.6);
                    inter.push(r * 0.6);
                }
            }
        }
        o.input_frames += frames as u64;
        let res = o.resampler.process(&inter, false);
        o.pending.extend_from_slice(&res);
        encode_pending(o, false)
    })
}

fn encode_pending(o: &mut OpusSession, flush: bool) -> i32 {
    let step = Opus::FRAME * o.channels;
    if flush && o.pending.len() % step != 0 {
        let pad = step - o.pending.len() % step;
        o.pending.extend(std::iter::repeat_n(0.0, pad));
    }
    let mut used = 0;
    while o.pending.len() - used >= step {
        match o.enc.encode(&o.pending[used..used + step]) {
            Ok(p) => o.packets.push(p),
            Err(_) => return -1,
        }
        used += step;
    }
    o.pending.drain(..used);
    0
}

/// Result: [json len u32][json {packets:[sizes], pre_skip, samples}][data].
#[unsafe(no_mangle)]
pub extern "C" fn sm_opus_end(input_rate: u32) -> i32 {
    let Some(mut o) = with(|s| s.opus.take()) else { return err("no opus session") };
    let tail = o.resampler.process(&[], true);
    o.pending.extend_from_slice(&tail);
    // Push the encoder's lookahead out with silence.
    let la = o.enc.lookahead() as usize;
    o.pending.extend(std::iter::repeat_n(0.0, la * o.channels));
    if encode_pending(&mut o, true) < 0 {
        return err("opus encode failed");
    }
    let samples = (o.input_frames as f64 * 48000.0 / input_rate as f64).round() as u64;
    let sizes: Vec<usize> = o.packets.iter().map(|p| p.len()).collect();
    let head = serde_json::to_vec(&json!({
        "packets": sizes, "pre_skip": la, "samples": samples, "channels": o.channels
    }))
    .unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    out.extend_from_slice(&head);
    for p in &o.packets {
        out.extend_from_slice(p);
    }
    ok(out)
}

// ------------------------------------------------------------------- muxing

#[derive(Deserialize)]
struct MuxIn {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    rotation: u32,
    /// "h264" (sps/pps) or "av1" (av1_header).
    #[serde(default)]
    codec: String,
    #[serde(default)]
    sps: String,
    #[serde(default)]
    pps: String,
    #[serde(default)]
    av1_header: String,
    /// [size, pts, key] per video sample, decode order.
    video: Vec<[i64; 3]>,
    audio: Option<MuxAudio>,
}

#[derive(Deserialize)]
struct MuxAudio {
    channels: u32,
    pre_skip: u32,
    input_rate: u32,
    samples: u64,
    packets: Vec<usize>,
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}

/// `data`: video sample bytes in order, then audio packet bytes.
///
/// # Safety
/// Pointers must reference the given number of readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sm_mux(json_ptr: *const u8, json_len: usize, data: *const u8, data_len: usize) -> i32 {
    let m: MuxIn = match serde_json::from_slice(unsafe { slice(json_ptr, json_len) }) {
        Ok(v) => v,
        Err(e) => return err(format!("mux input: {e}")),
    };
    let data = unsafe { slice(data, data_len) };
    let (sps, pps, av1_header) = (unhex(&m.sps), unhex(&m.pps), unhex(&m.av1_header));
    let mut pos = 0usize;
    let mut samples = Vec::with_capacity(m.video.len());
    for v in &m.video {
        let n = v[0] as usize;
        if pos + n > data.len() {
            return err("mux: video data short");
        }
        samples.push(mux::VSample { data: &data[pos..pos + n], pts: v[1], keyframe: v[2] != 0 });
        pos += n;
    }
    let mut packets = Vec::new();
    if let Some(a) = &m.audio {
        for &n in &a.packets {
            if pos + n > data.len() {
                return err("mux: audio data short");
            }
            packets.push(&data[pos..pos + n]);
            pos += n;
        }
    }
    let v = mux::VideoTrack {
        width: m.width,
        height: m.height,
        fps_num: m.fps_num,
        fps_den: m.fps_den,
        codec: if m.codec == "av1" {
            mux::VideoCodec::Av1 { seq_header: &av1_header }
        } else {
            mux::VideoCodec::H264 { sps: &sps, pps: &pps }
        },
        rotation: m.rotation,
        samples,
    };
    let a = m.audio.as_ref().map(|a| mux::AudioTrack {
        channels: a.channels,
        pre_skip: a.pre_skip,
        input_rate: a.input_rate,
        packets: packets.clone(),
        valid_samples: a.samples,
    });
    ok(mux::write(&v, a.as_ref()))
}
