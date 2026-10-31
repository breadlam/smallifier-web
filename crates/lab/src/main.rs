//! Native lab tool: generates ground-truth encodes for fitting and checking
//! the size/quality model, using the exact x264 build the web app ships.
//!
//!   lab speed <clip> <WxH> [frames]
//!   lab probe <file>...                 demux like the web app, check against ffprobe
//!   lab muxtest <clip> <out.mp4>         two chunks + Opus through the muxer
//!   lab twopass <clip> <WxH> <kbps1> [kbps2]
//!   lab parity <lab.json> <preset>       Rust model predictions (vs tools/fit_model.py)
//!   lab scalecheck <clip>                our Lanczos vs ffmpeg's
//!   lab av1 <clip> <WxH> <kbps> <preset> [frames] [extra opts]   SVT-AV1 two-pass check
//!   lab av1rc <clip> <lab.json> <preset>  SVT-AV1 two-pass size accuracy per chunk
//!   (gen with --preset svtN makes SVT-AV1 preset N ground truth)
//!   lab gen <clip> <WxH> <fps> <out.json> [--crfs 18,24,...] [--verify]
//!           [--preset P] [--rc-experiments]   (also 1-pass ABR and 2-pass per chunk)
//!           [--ref WxH]   measure quality against the source at WxH (decode upscaled)

use codecs::{X264, X264Config};
use smallifier_core::analysis::{Analyzer, FrameStats, Thumb};
use smallifier_core::scene;
use smallifier_core::yuv::{Frame, FrameScaler, Prefilter, roundtrip_gain};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use smallifier_core::model::x264_opts;

fn probe(path: &str) -> (usize, usize) {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path])
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().split(',').map(|v| v.parse::<usize>().unwrap());
    (it.next().unwrap(), it.next().unwrap())
}

/// Decodes `path` at `fps` (ffmpeg's fps filter) and scales each frame to
/// (w, h) with our own scaler, as the web app does.
fn load_frames(path: &str, fps: u32, w: usize, h: usize, max_frames: usize) -> Vec<Frame> {
    let (sw, sh) = probe(path);
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i", path, "-vf", &format!("fps={fps}"), "-pix_fmt", "yuv420p",
               "-f", "rawvideo", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    let mut stdout = child.stdout.take().unwrap();
    let fsize = sw * sh + 2 * sw.div_ceil(2) * sh.div_ceil(2);
    let scaler = FrameScaler::new(sw, sh, w, h);
    let mut frames = Vec::new();
    let mut buf = vec![0u8; fsize];
    while frames.len() < max_frames && stdout.read_exact(&mut buf).is_ok() {
        let f = Frame::from_i420(sw, sh, &buf);
        frames.push(if (sw, sh) == (w, h) { f } else { scaler.scale(&f) });
    }
    drop(stdout);
    let _ = child.kill();
    let _ = child.wait();
    frames
}

fn par_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let items: Vec<Mutex<Option<T>>> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
    let results: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(n.max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let t = items[i].lock().unwrap().take().unwrap();
                *results[i].lock().unwrap() = Some(f(t));
            });
        }
    });
    results.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// `rc` is the rate-control part of the options, e.g. "crf=26" or "bitrate=900".
fn x264_cfg(w: usize, h: usize, fps: u32, preset: &str, rc: &str, stats: bool) -> X264Config {
    X264Config {
        width: w as u32,
        height: h as u32,
        fps_num: fps,
        fps_den: 1,
        preset: preset.into(),
        tune: String::new(),
        opts: format!("{}:{rc}", x264_opts(preset)),
        stats,
    }
}

struct EncodeOut {
    bytes: usize,
    psnr_x264: f64,
    /// Annex-B stream (only kept when verifying).
    annexb: Vec<u8>,
}

fn encode_chunk(frames: &[Frame], cfg: &X264Config, keep: bool) -> EncodeOut {
    let mut enc = X264::open(cfg).unwrap();
    let mut bytes = 0;
    let mut psnr = Vec::new();
    let mut annexb = Vec::new();
    if keep {
        for ps in [enc.sps(), enc.pps()] {
            annexb.extend_from_slice(&[0, 0, 0, 1]);
            annexb.extend_from_slice(&ps);
        }
    }
    let mut take = |p: codecs::Packet| {
        bytes += p.data.len();
        psnr.push(p.psnr_y);
        if keep {
            let mut i = 0;
            while i + 4 <= p.data.len() {
                let n = u32::from_be_bytes(p.data[i..i + 4].try_into().unwrap()) as usize;
                annexb.extend_from_slice(&[0, 0, 0, 1]);
                annexb.extend_from_slice(&p.data[i + 4..i + 4 + n]);
                i += 4 + n;
            }
        }
    };
    for (i, f) in frames.iter().enumerate() {
        if let Some(p) = enc.encode(&f.y.data, &f.u.data, &f.v.data, f.y.w, f.u.w, i as i64, i == 0).unwrap() {
            take(p);
        }
    }
    while let Some(p) = enc.flush().unwrap() {
        take(p);
    }
    // Average MSE, not average PSNR, so the number composes across frames.
    let mse: f64 = psnr.iter().map(|p| 255.0f64.powi(2) / 10f64.powf(p / 10.0)).sum::<f64>() / psnr.len() as f64;
    EncodeOut { bytes, psnr_x264: 10.0 * (255.0f64.powi(2) / mse).log10(), annexb }
}

/// Decodes an Annex-B stream with ffmpeg and returns per-pixel luma MSE
/// against `reference`.
/// Decodes an Annex-B stream of (w, h) frames with ffmpeg and returns per-
/// pixel luma MSE against `reference`, upscaling the decode (our Lanczos)
/// when the reference is larger, so different output sizes are comparable.
fn true_luma_mse(annexb: &[u8], reference: &[Frame], w: usize, h: usize) -> f64 {
    true_luma_mse_fmt(annexb, "h264", reference, w, h)
}

/// `fmt` is the ffmpeg demuxer for `stream` ("h264" or "ivf").
fn true_luma_mse_fmt(stream: &[u8], fmt: &str, reference: &[Frame], w: usize, h: usize) -> f64 {
    let annexb = stream;
    let (rw, rh) = (reference[0].y.w, reference[0].y.h);
    let up = ((rw, rh) != (w, h)).then(|| smallifier_core::yuv::Scaler::new(w, h, rw, rh));
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-f", fmt, "-i", "-", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let data = annexb.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let mut out = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
    writer.join().unwrap();
    child.wait().unwrap();
    let fsize = w * h * 3 / 2;
    let n = (out.len() / fsize).min(reference.len());
    assert!(n == reference.len(), "decoded {n} of {} frames", reference.len());
    let mut se = 0f64;
    for (i, r) in reference.iter().enumerate().take(n) {
        let y = &out[i * fsize..i * fsize + w * h];
        let scaled;
        let y: &[u8] = match &up {
            Some(s) => {
                scaled = s.scale_plane(&smallifier_core::yuv::Plane { w, h, data: y.to_vec() });
                &scaled.data
            }
            None => y,
        };
        se += y.iter().zip(&r.y.data).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>();
    }
    se / (n * rw * rh) as f64
}

fn luma_mse(a: &Frame, b: &Frame) -> f64 {
    a.y.data.iter().zip(&b.y.data).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.y.data.len() as f64
}

fn rungs_for(w: usize, h: usize) -> Vec<(usize, usize)> {
    [0.5, 0.6, 0.75, 1.0]
        .iter()
        .map(|f| (((w as f64 * f / 2.0).round() * 2.0) as usize, ((h as f64 * f / 2.0).round() * 2.0) as usize))
        .collect()
}

fn speed(clip: &str, w: usize, h: usize, n: usize) {
    let frames = load_frames(clip, 30, w, h, n);
    for preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"] {
        let t = Instant::now();
        let r = encode_chunk(&frames, &x264_cfg(w, h, 30, preset, "crf=30", false), false);
        let dt = t.elapsed().as_secs_f64();
        println!("{preset:10} {:7.1} fps  {:8} bytes", frames.len() as f64 / dt, r.bytes);
    }
}

fn hist_json(v: &[u32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn generate(clip: &str, w: usize, h: usize, fps: u32, out_path: &str, crfs: &[f64], verify: bool, preset: &str,
            rc_experiments: bool, ref_wh: Option<(usize, usize)>) {
    let t0 = Instant::now();
    let frames = load_frames(clip, fps, w, h, usize::MAX);
    // Quality is measured against `refs` (the source at the reference size).
    let ref_frames = ref_wh.map(|(rw, rh)| load_frames(clip, fps, rw, rh, usize::MAX));
    let refs: &[Frame] = ref_frames.as_deref().unwrap_or(&frames);
    eprintln!("{} frames loaded in {:.1}s", frames.len(), t0.elapsed().as_secs_f64());

    // Scene cuts from thumbnails, every frame.
    let thumbs: Vec<Thumb> = frames.iter().map(Thumb::of).collect();
    let mut diffs = vec![0f32; frames.len()];
    for t in 1..frames.len() {
        diffs[t] = thumbs[t].diff(&thumbs[t - 1]);
    }
    let cuts = scene::detect_cuts(&diffs, fps as f64, 1.0);
    let chunks = scene::chunk_scenes(&cuts, frames.len(), (fps * 8) as usize);
    eprintln!("cuts {:?} -> {} chunks", cuts, chunks.len());

    // Analysis of every frame (P-style vs previous, and I-style).
    let t1 = Instant::now();
    let an = Analyzer::new(w, h);
    let stats: Vec<FrameStats> = par_map((0..frames.len()).collect(), |t| {
        an.analyze(if t > 0 { Some(&frames[t - 1].y) } else { None }, &frames[t].y)
    });
    eprintln!("analysis {:.1}s ({:.1} ms/frame/thread)", t1.elapsed().as_secs_f64(),
              t1.elapsed().as_secs_f64() * 1000.0 * 256.0 / frames.len() as f64);

    let rungs = rungs_for(w, h);
    // Encode jobs grouped by (chunk, rung) so the prefilter runs once.
    // Prefilter once per (chunk, rung), then fan out one job per CRF.
    let t2 = Instant::now();
    let pairs: Vec<(usize, usize)> = (0..chunks.len()).flat_map(|c| (0..rungs.len()).map(move |r| (c, r))).collect();
    let filtered: Vec<(Vec<Frame>, f64)> = par_map(pairs.clone(), |(ci, ri)| {
        let ch = &chunks[ci];
        let (rw, rh) = rungs[ri];
        let src = &frames[ch.start..ch.end];
        let out: Vec<Frame> = match Prefilter::new(w, h, rw, rh) {
            Some(p) => src.iter().map(|f| p.apply(f)).collect(),
            None => src.to_vec(),
        };
        let scale_mse = src.iter().zip(&out).map(|(a, b)| luma_mse(a, b)).sum::<f64>() / src.len() as f64;
        (out, scale_mse)
    });
    // CRF encodes, plus ABR encodes aimed at the sizes of two of the CRF
    // encodes (to see how well x264's own rate control hits a chunk target
    // and what it costs in quality versus CRF at the same size).
    let jobs: Vec<(usize, f64)> = (0..pairs.len()).flat_map(|p| crfs.iter().map(move |&c| (p, c))).collect();
    let crf_out = par_map(jobs.clone(), |(pi, crf)| {
        let (ci, ri) = pairs[pi];
        let (ref filt, scale_mse) = filtered[pi];
        let src = &refs[chunks[ci].start..chunks[ci].end];
        // "svtN" presets are SVT-AV1 preset N; anything else is x264.
        let (bytes, psnr_enc, stream, fmt) = match preset.strip_prefix("svt") {
            Some(p) => {
                let o = av1_encode(filt, fps, &svt_opts(p.parse().unwrap(), &format!("crf={crf}")), 0, Vec::new());
                let ivf = if verify { ivf_bytes(w, h, fps, &o.packets) } else { Vec::new() };
                (o.bytes, 0.0, ivf, "ivf")
            }
            None => {
                let e = encode_chunk(filt, &x264_cfg(w, h, fps, preset, &format!("crf={crf}"), true), verify);
                (e.bytes, e.psnr_x264, e.annexb, "h264")
            }
        };
        let true_mse = if verify { true_luma_mse_fmt(&stream, fmt, src, w, h) } else { -1.0 };
        (format!(
            "{{\"chunk\":{ci},\"rung\":{ri},\"crf\":{crf},\"bytes\":{},\"psnr_x264\":{:.4},\"scale_mse\":{:.5},\"true_mse\":{:.5}}}",
            bytes, psnr_enc, scale_mse, true_mse
        ), bytes, true_mse)
    });
    let stats_dir = format!("{out_path}.stats");
    std::fs::create_dir_all(&stats_dir).unwrap();
    let abr_jobs: Vec<usize> = if rc_experiments {
        (0..jobs.len()).filter(|&j| jobs[j].1 == 26.0 || jobs[j].1 == 34.0).collect()
    } else {
        Vec::new()
    };
    let abr_out = par_map(abr_jobs, |j| {
        let (pi, crf) = jobs[j];
        let (ci, ri) = pairs[pi];
        let (ref filt, _) = filtered[pi];
        let src = &refs[chunks[ci].start..chunks[ci].end];
        let target = crf_out[j].1;
        let kbps = target as f64 * 8.0 / (filt.len() as f64 / fps as f64) / 1000.0;
        let br = kbps.round() as u64;
        let e = encode_chunk(filt, &x264_cfg(w, h, fps, preset, &format!("bitrate={br}"), true), verify);
        let true_mse = if verify { true_luma_mse(&e.annexb, src, w, h) } else { -1.0 };
        // Two-pass: fast first pass, then the real pass reading its stats.
        let stats = format!("{stats_dir}/{j}.log");
        let t = Instant::now();
        encode_chunk(filt, &x264_cfg(w, h, fps, preset, &format!("bitrate={br}:pass=1:stats={stats}:fastfirstpass"), false), false);
        let t_p1 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let e2 = encode_chunk(filt, &x264_cfg(w, h, fps, preset, &format!("bitrate={br}:pass=2:stats={stats}"), true), verify);
        let t_p2 = t.elapsed().as_secs_f64();
        let true_mse2 = if verify { true_luma_mse(&e2.annexb, src, w, h) } else { -1.0 };
        format!(
            "{{\"chunk\":{ci},\"rung\":{ri},\"crf_ref\":{crf},\"target\":{target},\"bytes\":{},\"true_mse\":{:.5},\"crf_true_mse\":{:.5},\"bytes_2pass\":{},\"true_mse_2pass\":{:.5},\"t_pass1\":{t_p1:.3},\"t_pass2\":{t_p2:.3}}}",
            e.bytes, true_mse, crf_out[j].2, e2.bytes, true_mse2
        )
    });
    let results: Vec<Vec<String>> = crf_out.into_iter().map(|r| vec![r.0]).collect();
    let _ = std::fs::remove_dir_all(&stats_dir);
    drop(filtered);
    eprintln!("encodes {:.1}s", t2.elapsed().as_secs_f64());

    // Per-chunk aggregated stats: frame 0 as I, the rest as P.
    let mut chunk_json = Vec::new();
    for ch in &chunks {
        let first = &stats[ch.start];
        let mut p = FrameStats::default();
        for s in &stats[ch.start + 1..ch.end] {
            p.add(s);
        }
        let mut src_e = vec![0f64; 64];
        for s in &stats[ch.start..ch.end] {
            for k in 0..64 {
                src_e[k] += s.src_energy[k];
            }
        }
        // Every-other-frame P stats, to test temporal subsampling later.
        let mut p2 = FrameStats::default();
        for s in stats[ch.start + 1..ch.end].iter().step_by(2) {
            p2.add(s);
        }
        let mut p4 = FrameStats::default();
        for s in stats[ch.start + 1..ch.end].iter().step_by(4) {
            p4.add(s);
        }
        let src_parts: Vec<String> = src_e.iter().map(|v| format!("{v:.1}")).collect();
        chunk_json.push(format!(
            "{{\"start\":{},\"end\":{},\"scene\":{},\"i_hist\":{},\"p_hist\":{},\"p2_hist\":{},\"p4_hist\":{},\"p_frames\":{},\"p2_frames\":{},\"p4_frames\":{},\"src_energy\":[{}],\"mbs\":{},\"intra_mbs\":{},\"mv_cost\":{:.2},\"inter_sad\":{:.4},\"i_pixels\":{}}}",
            ch.start, ch.end, ch.scene, hist_json(&first.intra), hist_json(&p.inter), hist_json(&p2.inter), hist_json(&p4.inter),
            ch.len() - 1, stats[ch.start + 1..ch.end].iter().step_by(2).count(),
            stats[ch.start + 1..ch.end].iter().step_by(4).count(),
            src_parts.join(","), p.mbs, p.intra_mbs, p.mv_cost, p.inter_sad, first.pixels
        ));
    }
    let gains: Vec<String> = rungs
        .iter()
        .map(|&(rw, rh)| {
            let gx: Vec<String> = (0..8).map(|u| format!("{:.5}", roundtrip_gain(w, rw, u as f64 / 16.0))).collect();
            let gy: Vec<String> = (0..8).map(|v| format!("{:.5}", roundtrip_gain(h, rh, v as f64 / 16.0))).collect();
            format!("{{\"w\":{rw},\"h\":{rh},\"gx\":[{}],\"gy\":[{}]}}", gx.join(","), gy.join(","))
        })
        .collect();
    let rows: Vec<String> = results.into_iter().flatten().collect();
    let json = format!(
        "{{\"clip\":\"{clip}\",\"w\":{w},\"h\":{h},\"fps\":{fps},\"preset\":\"{preset}\",\"opts\":\"{}\",\"rungs\":[{}],\"chunks\":[{}],\"results\":[{}],\"abr\":[{}]}}",
        match preset.strip_prefix("svt") { Some(p) => svt_opts(p.parse().unwrap(), "crf=N"), None => x264_opts(preset) },
        gains.join(","), chunk_json.join(","), rows.join(","), abr_out.join(",")
    );
    std::fs::write(out_path, json).unwrap();
    eprintln!("wrote {out_path} in {:.1}s total", t0.elapsed().as_secs_f64());
}

/// Reads a file's tracks the way the web app does: top-level box scan for
/// MP4 (only moov/moof payloads are read), incremental indexing for MKV.
pub fn demux(path: &str) -> Vec<smallifier_core::media::TrackInfo> {
    let data = std::fs::read(path).unwrap();
    if data.len() > 8 && &data[4..8] == b"ftyp" {
        let mut moov = None;
        let mut moofs = Vec::new();
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let size32 = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            let (hdr, size) = match size32 {
                1 => (16, u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize),
                0 => (8, data.len() - pos),
                s => (8, s),
            };
            let typ = &data[pos + 4..pos + 8];
            if typ == b"moov" {
                moov = Some(&data[pos + hdr..pos + size]);
            } else if typ == b"moof" {
                moofs.push((pos as u64, &data[pos + hdr..pos + size]));
            }
            if size < hdr {
                break;
            }
            pos += size;
        }
        smallifier_core::mp4::parse(moov.expect("no moov"), &moofs).unwrap()
    } else {
        let mut p = smallifier_core::mkv::MkvParser::new();
        while !p.done() {
            let at = p.want() as usize;
            if at >= data.len() {
                p.feed(at as u64, &[], true);
                break;
            }
            let end = (at + (4 << 20)).min(data.len());
            p.feed(at as u64, &data[at..end], end == data.len());
        }
        p.tracks()
    }
}

fn probe_cmd(path: &str) {
    let tracks = demux(path);
    // ffprobe's packet list for comparison: (stream index, pos, size)
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "packet=stream_index,pos,size,pts_time", "-of", "csv=p=0", path])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let pk: Vec<(usize, u64, u32)> = text
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Some((f[0].parse().ok()?, f[3].parse().ok()?, f[2].parse().ok()?))
        })
        .collect();
    for (ti, t) in tracks.iter().enumerate() {
        let keys = t.samples.iter().filter(|s| s.keyframe).count();
        let first = t.samples.iter().map(|s| s.pts).fold(f64::MAX, f64::min);
        // ffprobe reports Matroska packet positions at the block element,
        // not the payload, so compare sizes there (payload bytes are
        // checked against ffmpeg's demuxer below).
        let mkv = !path.ends_with(".mp4") && !path.ends_with(".mov");
        let key = |o: u64| if mkv { 0 } else { o };
        let mut ours: Vec<(u64, u32)> = t.samples.iter().map(|s| (key(s.offset), s.size)).collect();
        let mut theirs: Vec<(u64, u32)> = pk.iter().filter(|p| p.0 == ti).map(|p| (key(p.1), p.2)).collect();
        ours.sort();
        theirs.sort();
        println!(
            "  track {} {:?} {} [{}] {}x{} {}Hz {}ch: {} samples, {} keys, first pts {:.3}, dur {:.2}s, desc {}B, offsets/sizes {}",
            t.id, t.kind, t.codec_name, t.codec, t.width, t.height, t.sample_rate, t.channels, t.samples.len(), keys,
            first, t.duration, t.description.len(),
            if ours == theirs { "MATCH ffprobe".to_string() } else { format!("DIFFER (ffprobe has {})", theirs.len()) }
        );
        // Payload check: our bytes for the first 50 samples must equal what
        // ffmpeg's demuxer copies out.
        let data = std::fs::read(path).unwrap();
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-i", path, "-map", &format!("0:{ti}"), "-c", "copy", "-frames", "50",
                   "-bsf", "null", "-f", "data", "-"])
            .output()
            .unwrap();
        let mut mine = Vec::new();
        for s in t.samples.iter().take(50) {
            mine.extend_from_slice(&data[s.offset as usize..s.offset as usize + s.size as usize]);
        }
        if !out.stdout.is_empty() {
            let n = mine.len().min(out.stdout.len());
            println!("    payload bytes of first 50 samples: {}", if mine[..n] == out.stdout[..n] { "identical" } else { "DIFFERENT" });
        }
    }
}

/// Two independently encoded chunks + Opus, muxed: exercises chunk
/// concatenation (stitchable SPS/PPS), B-frame timing and the audio track.
fn muxtest(clip: &str, out: &str) {
    use smallifier_core::mux;
    let (w, h, fps) = (640usize, 360usize, 30u32);
    let frames = load_frames(clip, fps, w, h, 150);
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    let mut packets: Vec<codecs::Packet> = Vec::new();
    for (start, end) in [(0usize, 70usize), (70, frames.len())] {
        let mut enc = X264::open(&x264_cfg(w, h, fps, "medium", "crf=28", false)).unwrap();
        if sps.is_empty() {
            sps = enc.sps();
            pps = enc.pps();
        } else {
            assert_eq!(sps, enc.sps(), "SPS differs between chunks");
            assert_eq!(pps, enc.pps(), "PPS differs between chunks");
        }
        for (i, f) in frames[start..end].iter().enumerate() {
            if let Some(mut p) = enc.encode(&f.y.data, &f.u.data, &f.v.data, w, w / 2, i as i64, i == 0).unwrap() {
                p.pts += start as i64;
                packets.push(p);
            }
        }
        while let Some(mut p) = enc.flush().unwrap() {
            p.pts += start as i64;
            packets.push(p);
        }
    }
    // 5 s of a 440 Hz tone, stereo.
    let mut opus = codecs::Opus::open(2, 64000).unwrap();
    let pre = opus.lookahead();
    let total = 48000 * 5;
    let mut apk = Vec::new();
    let mut n = 0;
    while n < total + pre as usize {
        let pcm: Vec<f32> = (0..codecs::Opus::FRAME)
            .flat_map(|i| {
                let t = (n + i) as f32 / 48000.0;
                let v = (t * 440.0 * std::f32::consts::TAU).sin() * 0.3;
                [v, v]
            })
            .collect();
        apk.push(opus.encode(&pcm).unwrap());
        n += codecs::Opus::FRAME;
    }
    let v = mux::VideoTrack {
        width: w as u32,
        height: h as u32,
        fps_num: fps,
        fps_den: 1,
        codec: mux::VideoCodec::H264 { sps: &sps, pps: &pps },
        rotation: 0,
        samples: packets.iter().map(|p| mux::VSample { data: &p.data, pts: p.pts, keyframe: p.keyframe }).collect(),
    };
    let a = mux::AudioTrack {
        channels: 2,
        pre_skip: pre,
        input_rate: 48000,
        packets: apk.iter().map(|p| p.as_slice()).collect(),
        valid_samples: total as u64,
    };
    let bytes = mux::write(&v, Some(&a));
    let media: usize = packets.iter().map(|p| p.data.len()).sum::<usize>() + apk.iter().map(|p| p.len()).sum::<usize>();
    std::fs::write(out, &bytes).unwrap();
    println!("wrote {out}: {} bytes ({} media, {} container; estimate {})", bytes.len(), media, bytes.len() - media,
             mux::overhead_estimate(packets.len(), fps as f64, apk.len(), true));
}

/// Prints the Rust model's predictions for every row of a lab file, for
/// comparison with tools/fit_model.py (they must agree).
fn parity(path: &str, preset: &str) {
    use smallifier_core::model::{self, ChunkStats, Gain};
    let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (w, h) = (d["w"].as_u64().unwrap() as usize, d["h"].as_u64().unwrap() as usize);
    let m = model::model_for(preset);
    let hist = |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    let chunks: Vec<ChunkStats> = d["chunks"].as_array().unwrap().iter().map(|c| {
        let frames = (c["end"].as_u64().unwrap() - c["start"].as_u64().unwrap()) as u32;
        ChunkStats {
            i_hist: hist(&c["i_hist"]),
            p_hist: hist(&c["p_hist"]),
            p_sampled: c["p_frames"].as_u64().unwrap() as u32,
            frames,
            src_energy: c["src_energy"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
            src_frames: frames,
            mbs_per_frame: c["i_pixels"].as_f64().unwrap() / 256.0,
            pixels_per_frame: c["i_pixels"].as_f64().unwrap(),
            mv_cost: c["mv_cost"].as_f64().unwrap(),
        }
    }).collect();
    let gains: Vec<Gain> = d["rungs"].as_array().unwrap().iter().map(|r| {
        let (rw, rh) = (r["w"].as_u64().unwrap() as usize, r["h"].as_u64().unwrap() as usize);
        if (rw, rh) == (w, h) { Gain::identity() } else { Gain::for_rung(w, h, rw, rh) }
    }).collect();
    for r in d["results"].as_array().unwrap() {
        let c = &chunks[r["chunk"].as_u64().unwrap() as usize];
        let g = &gains[r["rung"].as_u64().unwrap() as usize];
        let crf = r["crf"].as_f64().unwrap();
        println!("{} {} {} {:.1} {:.4}", r["chunk"], r["rung"], crf, model::predict_bytes(m, c, g, crf),
                 model::psnr(model::predict_mse(m, c, g, crf)));
        if std::env::var("PARITY_TERMS").is_ok_and(|v| v.parse::<f64>().ok() == Some(crf)) {
            eprintln!("terms {} {}: {:?}", r["chunk"], r["rung"], model::predict_terms(m, c, g, crf).map(|t| (t / 8.0).round()));
        }
    }
}

use smallifier_core::model::svt_opts;

pub struct Av1Out {
    pub bytes: usize,
    pub header: Vec<u8>,
    pub packets: Vec<codecs::av1::Packet>,
    pub stats: Vec<u8>,
    /// Copy of the first-pass stats (for a second use).
    pub stats_for_reuse: Vec<u8>,
    pub secs: f64,
}

pub fn av1_encode(frames: &[Frame], fps: u32, opts: &str, pass: u8, stats: Vec<u8>) -> Av1Out {
    use codecs::av1::{Svt, SvtConfig};
    let (w, h) = (frames[0].y.w, frames[0].y.h);
    let t = Instant::now();
    let mut e = Svt::open(&SvtConfig { width: w as u32, height: h as u32, fps_num: fps, fps_den: 1, opts: opts.into(), pass, stats })
        .unwrap();
    for (i, f) in frames.iter().enumerate() {
        e.send(&f.y.data, &f.u.data, &f.v.data, f.y.w, f.u.w, i as i64).unwrap();
    }
    let packets = e.finish().unwrap();
    let stats = if pass == 1 { e.stats() } else { Vec::new() };
    Av1Out { bytes: packets.iter().map(|p| p.data.len()).sum(), header: e.header(), packets, stats_for_reuse: stats.clone(), stats,
             secs: t.elapsed().as_secs_f64() }
}

/// IVF file (temporal delimiters put back) for checking with ffmpeg.
pub fn write_ivf(path: &str, w: usize, h: usize, fps: u32, packets: &[codecs::av1::Packet]) {
    std::fs::write(path, ivf_bytes(w, h, fps, packets)).unwrap();
}

pub fn ivf_bytes(w: usize, h: usize, fps: u32, packets: &[codecs::av1::Packet]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(b"DKIF");
    o.extend_from_slice(&0u16.to_le_bytes());
    o.extend_from_slice(&32u16.to_le_bytes());
    o.extend_from_slice(b"AV01");
    o.extend_from_slice(&(w as u16).to_le_bytes());
    o.extend_from_slice(&(h as u16).to_le_bytes());
    o.extend_from_slice(&fps.to_le_bytes());
    o.extend_from_slice(&1u32.to_le_bytes());
    o.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes());
    for p in packets {
        o.extend_from_slice(&(p.data.len() as u32 + 2).to_le_bytes());
        o.extend_from_slice(&(p.pts as u64).to_le_bytes());
        o.extend_from_slice(&[0x12, 0x00]);
        o.extend_from_slice(&p.data);
    }
    o
}

fn parse_wh(s: &str) -> (usize, usize) {
    let (a, b) = s.split_once('x').expect("WxH");
    (a.parse().unwrap(), b.parse().unwrap())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("speed") => {
            let (w, h) = parse_wh(&args[3]);
            speed(&args[2], w, h, args.get(4).map(|v| v.parse().unwrap()).unwrap_or(120));
        }
        Some("muxtest") => muxtest(&args[2], &args[3]),
        Some("av1rc") => {
            // lab av1rc <clip> <lab.json> <preset> -- per-chunk two-pass VBR accuracy at the
            // sizes x264 reached at CRF 28 and 36 (rung 3) in a lab file of the same clip.
            let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
            let (w, h) = (d["w"].as_u64().unwrap() as usize, d["h"].as_u64().unwrap() as usize);
            let preset: u32 = args[4].parse().unwrap();
            let frames = load_frames(&args[2], 30, w, h, usize::MAX);
            let chunks: Vec<(usize, usize)> = d["chunks"].as_array().unwrap().iter()
                .map(|c| (c["start"].as_u64().unwrap() as usize, c["end"].as_u64().unwrap() as usize)).collect();
            let mut jobs = Vec::new();
            for r in d["results"].as_array().unwrap() {
                let crf = r["crf"].as_f64().unwrap();
                if r["rung"].as_u64() == Some(3) && (crf == 28.0 || crf == 36.0) {
                    for gc in [false, true] {
                        jobs.push((r["chunk"].as_u64().unwrap() as usize, r["bytes"].as_f64().unwrap(), crf, gc));
                    }
                }
            }
            let rows = par_map(jobs, |(ci, bytes, crf, gc)| {
                let (a, b) = chunks[ci];
                let secs = (b - a) as f64 / 30.0;
                let rc = format!("rc=1:tbr={}b{}", (bytes * 8.0 / secs) as u64, if gc { ":gop-constraint-rc=1" } else { "" });
                let p1 = av1_encode(&frames[a..b], 30, &svt_opts(preset, &rc), 1, Vec::new());
                let p2 = av1_encode(&frames[a..b], 30, &svt_opts(preset, &rc), 2, p1.stats);
                // And the response to asking for 10% less, reusing the stats.
                let rc2 = format!("rc=1:tbr={}b{}", (0.9 * bytes * 8.0 / secs) as u64, if gc { ":gop-constraint-rc=1" } else { "" });
                let p3 = av1_encode(&frames[a..b], 30, &svt_opts(preset, &rc2), 2, p1.stats_for_reuse.clone());
                format!("{ci} {} {crf} {gc} {bytes:.0} {} {} {:.2} {:.2}", b - a, p2.bytes, p3.bytes, p1.secs, p2.secs)
            });
            for r in rows {
                println!("{r}");
            }
        }
        Some("av1") => {
            // lab av1 <clip> <WxH> <kbps> <preset> [frames]
            let (w, h) = parse_wh(&args[3]);
            let kbps: f64 = args[4].parse().unwrap();
            let preset: u32 = args[5].parse().unwrap();
            let n = args.get(6).map(|v| v.parse().unwrap()).unwrap_or(120);
            let extra = args.get(7).cloned().unwrap_or_default();
            let frames = load_frames(&args[2], 30, w, h, n);
            let rc = format!("rc=1:tbr={}b{extra}", (kbps * 1000.0) as u64);
            let p1 = av1_encode(&frames, 30, &svt_opts(preset, &rc), 1, Vec::new());
            let p2 = av1_encode(&frames, 30, &svt_opts(preset, &rc), 2, p1.stats.clone());
            let target = kbps * 1000.0 / 8.0 * frames.len() as f64 / 30.0;
            println!("preset {preset}: pass1 {:.1} fps ({} B stats), pass2 {:.1} fps; {} packets, {} bytes vs target {:.0} ({:+.1}%), keyframes {}, header {} B",
                     frames.len() as f64 / p1.secs, p1.stats.len(), frames.len() as f64 / p2.secs, p2.packets.len(), p2.bytes, target,
                     (p2.bytes as f64 / target - 1.0) * 100.0, p2.packets.iter().filter(|p| p.keyframe).count(), p2.header.len());
            let pts: Vec<i64> = p2.packets.iter().map(|p| p.pts).collect();
            println!("pts monotonic: {}", pts.windows(2).all(|w| w[1] > w[0]));
            let path = std::env::temp_dir().join("smallifier-av1.ivf");
            write_ivf(path.to_str().unwrap(), w, h, 30, &p2.packets);
            println!("wrote {}", path.display());
        }
        Some("parity") => parity(&args[2], &args[3]),
        Some("scalecheck") => {
            // Our 1080p->720p Lanczos versus ffmpeg's, on real frames.
            let clip = &args[2];
            let ours = load_frames(clip, 30, 1280, 720, 30);
            let out = Command::new("ffmpeg")
                .args(["-v", "error", "-i", clip, "-frames:v", "30", "-vf", "fps=30,scale=1280:720:flags=lanczos", "-pix_fmt", "yuv420p",
                       "-f", "rawvideo", "-"])
                .output()
                .unwrap()
                .stdout;
            let fs = 1280 * 720 * 3 / 2;
            for (shift_x, shift_y) in [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                let mut se = 0f64;
                let mut n = 0usize;
                for (i, f) in ours.iter().enumerate() {
                    let theirs = &out[i * fs..i * fs + 1280 * 720];
                    for y in 8..712 {
                        for x in 8..1272 {
                            let a = f.y.data[y * 1280 + x] as f64;
                            let b = theirs[((y as i32 + shift_y) as usize) * 1280 + (x as i32 + shift_x) as usize] as f64;
                            se += (a - b).powi(2);
                            n += 1;
                        }
                    }
                }
                println!("shift ({shift_x},{shift_y}): PSNR ours vs ffmpeg lanczos {:.2} dB", 10.0 * (65025.0 / (se / n as f64)).log10());
            }
        }
        Some("twopass") => {
            // Same flow as tools/node_smoke.mjs, natively.
            let (w, h) = parse_wh(&args[3]);
            let kbps: u32 = args[4].parse().unwrap();
            let kbps2: u32 = args.get(5).map(|v| v.parse().unwrap()).unwrap_or(kbps);
            let opts_extra = args.get(6).cloned().unwrap_or_default();
            let frames = load_frames(&args[2], 30, w, h, 120);
            let stats_path = std::env::temp_dir().join("smallifier-twopass.log");
            let stats = stats_path.to_str().unwrap();
            let p1 = encode_chunk(&frames, &x264_cfg(w, h, 30, "medium", &format!("bitrate={kbps}:pass=1:stats={stats}:fastfirstpass{opts_extra}"), false), false);
            let p2 = encode_chunk(&frames, &x264_cfg(w, h, 30, "medium", &format!("bitrate={kbps2}:pass=2:stats={stats}{opts_extra}"), true), false);
            println!("pass1 {} bytes, pass2 {} bytes (target {})", p1.bytes, p2.bytes, kbps2 as usize * 1000 / 8 * frames.len() / 30);
        }
        Some("probe") => {
            for f in &args[2..] {
                println!("{f}");
                probe_cmd(f);
            }
        }
        Some("gen") => {
            let (w, h) = parse_wh(&args[3]);
            let fps: u32 = args[4].parse().unwrap();
            let mut crfs = vec![18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0];
            let mut verify = false;
            let mut rc_experiments = false;
            let mut ref_wh = None;
            let mut preset = smallifier_core::model::X264_PRESET.to_string();
            let mut i = 6;
            while i < args.len() {
                match args[i].as_str() {
                    "--crfs" => {
                        crfs = args[i + 1].split(',').map(|v| v.parse().unwrap()).collect();
                        i += 1;
                    }
                    "--verify" => verify = true,
                    "--rc-experiments" => rc_experiments = true,
                    "--ref" => {
                        ref_wh = Some(parse_wh(&args[i + 1]));
                        i += 1;
                    }
                    "--preset" => {
                        preset = args[i + 1].clone();
                        i += 1;
                    }
                    a => panic!("unknown arg {a}"),
                }
                i += 1;
            }
            generate(&args[2], w, h, fps, &args[5], &crfs, verify, &preset, rc_experiments, ref_wh);
        }
        _ => eprintln!("usage: lab speed <clip> <WxH> [frames] | lab gen <clip> <WxH> <fps> <out.json> [--crfs a,b] [--verify]"),
    }
}
