//! ISO-BMFF (MP4/MOV) reading: track descriptions and sample tables from a
//! `moov` box, plus any `moof` boxes of a fragmented file.
//!
//! The caller locates top-level boxes (cheap: one header read per box) and
//! hands over only `moov`/`moof` payloads, so the media data never has to
//! be held in memory.

use crate::media::*;

pub struct Boxes<'a> {
    data: &'a [u8],
    pos: usize,
}

/// Iterates (fourcc, payload) over consecutive boxes.
pub fn boxes(data: &[u8]) -> Boxes<'_> {
    Boxes { data, pos: 0 }
}

impl<'a> Iterator for Boxes<'a> {
    type Item = ([u8; 4], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let d = &self.data[self.pos..];
        if d.len() < 8 {
            return None;
        }
        let size32 = u32::from_be_bytes(d[0..4].try_into().unwrap()) as u64;
        let typ: [u8; 4] = d[4..8].try_into().unwrap();
        let (hdr, size) = match size32 {
            0 => (8, d.len() as u64),
            1 if d.len() >= 16 => (16, u64::from_be_bytes(d[8..16].try_into().unwrap())),
            1 => return None,
            s => (8, s),
        };
        if size < hdr as u64 || size > d.len() as u64 {
            return None;
        }
        self.pos += size as usize;
        Some((typ, &d[hdr..size as usize]))
    }
}

fn child<'a>(data: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(t, _)| t == name).map(|(_, p)| p)
}

fn path<'a>(data: &'a [u8], names: &[&[u8; 4]]) -> Option<&'a [u8]> {
    let mut d = data;
    for n in names {
        d = child(d, n)?;
    }
    Some(d)
}

#[derive(Clone, Copy)]
struct R<'a>(&'a [u8], usize);

impl<'a> R<'a> {
    fn new(d: &'a [u8]) -> Self {
        R(d, 0)
    }
    fn ok(&self, n: usize) -> bool {
        self.1 + n <= self.0.len()
    }
    fn u8(&mut self) -> u8 {
        let v = self.0.get(self.1).copied().unwrap_or(0);
        self.1 += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        (self.u8() as u16) << 8 | self.u8() as u16
    }
    fn u24(&mut self) -> u32 {
        (self.u16() as u32) << 8 | self.u8() as u32
    }
    fn u32(&mut self) -> u32 {
        (self.u16() as u32) << 16 | self.u16() as u32
    }
    fn u64(&mut self) -> u64 {
        (self.u32() as u64) << 32 | self.u32() as u64
    }
    fn skip(&mut self, n: usize) {
        self.1 += n;
    }
}

/// Reads up to `n` records of `each` bytes (stops early on truncation).
fn records<T>(r: &mut R, n: usize, each: usize, mut f: impl FnMut(&mut R) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        if !r.ok(each) {
            break;
        }
        out.push(f(r));
    }
    out
}

/// Version and flags of a full box.
fn vf(r: &mut R) -> (u8, u32) {
    let v = r.u8();
    (v, r.u24())
}

struct Trex {
    duration: u32,
    size: u32,
    flags: u32,
}

/// Per-track state needed to append fragment samples.
struct TrackCtx {
    timescale: u32,
    /// media_time of the first edit, in media timescale units.
    edit_shift: i64,
    /// Leading empty edit, seconds.
    edit_delay: f64,
    trex: Trex,
    next_dts: u64,
}

pub fn parse(moov: &[u8], moofs: &[(u64, &[u8])]) -> Result<Vec<TrackInfo>, String> {
    let mvhd_scale = child(moov, b"mvhd")
        .map(|m| {
            let mut r = R::new(m);
            let (v, _) = vf(&mut r);
            r.skip(if v == 1 { 16 } else { 8 });
            r.u32()
        })
        .unwrap_or(1000)
        .max(1);
    let mut tracks = Vec::new();
    let mut ctxs = Vec::new();
    for (t, trak) in boxes(moov) {
        if &t != b"trak" {
            continue;
        }
        if let Some((info, ctx)) = parse_trak(trak, mvhd_scale)? {
            tracks.push(info);
            ctxs.push(ctx);
        }
    }
    if let Some(mvex) = child(moov, b"mvex") {
        for (t, p) in boxes(mvex) {
            if &t != b"trex" {
                continue;
            }
            let mut r = R::new(p);
            vf(&mut r);
            let id = r.u32();
            r.skip(4);
            let trex = Trex { duration: r.u32(), size: r.u32(), flags: r.u32() };
            if let Some(i) = tracks.iter().position(|t| t.id == id) {
                ctxs[i].trex = trex;
            }
        }
    }
    for &(moof_off, moof) in moofs {
        parse_moof(moof_off, moof, &mut tracks, &mut ctxs);
    }
    for t in &mut tracks {
        t.samples.sort_by(|a, b| a.dts.partial_cmp(&b.dts).unwrap());
        if let (Some(a), Some(b)) = (t.samples.first(), t.samples.iter().map(|s| s.pts).reduce(f64::max)) {
            let last_dur = if t.samples.len() > 1 {
                (t.samples[t.samples.len() - 1].dts - t.samples[t.samples.len() - 2].dts).max(0.0)
            } else {
                0.0
            };
            t.duration = t.duration.max(b + last_dur - a.pts.min(0.0));
        }
    }
    tracks.retain(|t| !t.samples.is_empty());
    Ok(tracks)
}

fn parse_trak(trak: &[u8], mvhd_scale: u32) -> Result<Option<(TrackInfo, TrackCtx)>, String> {
    let tkhd = child(trak, b"tkhd").ok_or("trak without tkhd")?;
    let mut r = R::new(tkhd);
    let (v, _) = vf(&mut r);
    r.skip(if v == 1 { 16 } else { 8 });
    let id = r.u32();
    r.skip(4 + if v == 1 { 8 } else { 4 } + 8 + 8);
    let m: Vec<i32> = (0..9).map(|_| r.u32() as i32).collect();
    let rotation = {
        let deg = (m[1] as f64).atan2(m[0] as f64).to_degrees().round() as i32;
        (((deg % 360) + 360) % 360) as u32 / 90 * 90
    };

    let mdia = child(trak, b"mdia").ok_or("trak without mdia")?;
    let hdlr = child(mdia, b"hdlr").ok_or("mdia without hdlr")?;
    let kind = match &hdlr[8..12] {
        b"vide" => Kind::Video,
        b"soun" => Kind::Audio,
        _ => return Ok(None),
    };
    let mdhd = child(mdia, b"mdhd").ok_or("no mdhd")?;
    let mut r = R::new(mdhd);
    let (v, _) = vf(&mut r);
    r.skip(if v == 1 { 16 } else { 8 });
    let timescale = r.u32().max(1);
    let duration = if v == 1 { r.u64() } else { r.u32() as u64 };

    let mut info = TrackInfo::new(id, kind);
    info.rotation = rotation;
    info.duration = duration as f64 / timescale as f64;

    // Edit list: first edit only (leading empty edit = delay).
    let (mut edit_shift, mut edit_delay) = (0i64, 0f64);
    if let Some(elst) = path(trak, &[b"edts", b"elst"]) {
        let mut r = R::new(elst);
        let (v, _) = vf(&mut r);
        let n = r.u32();
        for _ in 0..n {
            let (dur, mt) = if v == 1 { (r.u64(), r.u64() as i64) } else { (r.u32() as u64, r.u32() as i32 as i64) };
            r.skip(4);
            if mt == -1 {
                edit_delay += dur as f64 / mvhd_scale as f64;
                continue;
            }
            edit_shift = mt;
            break;
        }
    }

    let stbl = path(mdia, &[b"minf", b"stbl"]).ok_or("no stbl")?;
    if let Some(stsd) = child(stbl, b"stsd") {
        parse_stsd(&stsd[8..], &mut info);
    }
    let ctx = TrackCtx { timescale, edit_shift, edit_delay, trex: Trex { duration: 0, size: 0, flags: 0 }, next_dts: 0 };
    let (samples, next_dts) = sample_table(stbl, &ctx)?;
    info.samples = samples;
    Ok(Some((info, TrackCtx { next_dts, ..ctx })))
}

fn parse_stsd(entries: &[u8], info: &mut TrackInfo) {
    let Some((fourcc, e)) = boxes(entries).next() else { return };
    let name = String::from_utf8_lossy(&fourcc).to_string();
    info.codec_name = name.clone();
    match info.kind {
        Kind::Video => {
            if e.len() < 78 {
                return;
            }
            info.width = u16::from_be_bytes([e[24], e[25]]) as u32;
            info.height = u16::from_be_bytes([e[26], e[27]]) as u32;
            let kids = &e[78..];
            match &fourcc {
                b"avc1" | b"avc3" => {
                    info.codec_name = "H.264".into();
                    if let Some(c) = child(kids, b"avcC") {
                        info.codec = avc_codec_string(c).unwrap_or_default();
                        info.description = c.to_vec();
                    }
                }
                b"hvc1" | b"hev1" => {
                    info.codec_name = "HEVC".into();
                    if let Some(c) = child(kids, b"hvcC") {
                        info.codec = hevc_codec_string(&name, c).unwrap_or_default();
                        info.description = c.to_vec();
                    }
                }
                b"vp09" => {
                    info.codec_name = "VP9".into();
                    info.codec = child(kids, b"vpcC").and_then(vp9_codec_string).unwrap_or("vp09.00.10.08".into());
                }
                b"vp08" => {
                    info.codec_name = "VP8".into();
                    info.codec = "vp8".into();
                }
                b"av01" => {
                    info.codec_name = "AV1".into();
                    if let Some(c) = child(kids, b"av1C") {
                        info.codec = av1_codec_string(c).unwrap_or_default();
                        info.description = c.to_vec();
                    }
                }
                _ => {}
            }
        }
        Kind::Audio => {
            if e.len() < 28 {
                return;
            }
            let version = u16::from_be_bytes([e[8], e[9]]);
            info.channels = u16::from_be_bytes([e[16], e[17]]) as u32;
            info.sample_rate = u16::from_be_bytes([e[24], e[25]]) as u32;
            // QuickTime sound description v1/v2 extend the entry.
            let ext = match version {
                1 => 16,
                2 => 36,
                _ => 0,
            };
            if version == 2 && e.len() >= 64 {
                let sr = f64::from_bits(u64::from_be_bytes(e[32..40].try_into().unwrap()));
                info.sample_rate = sr.round() as u32;
                info.channels = u32::from_be_bytes(e[40..44].try_into().unwrap());
            }
            let kids = e.get(28 + ext..).unwrap_or(&[]);
            match &fourcc {
                b"mp4a" => {
                    let esds = child(kids, b"esds").or_else(|| child(kids, b"wave").and_then(|w| child(w, b"esds")));
                    if let Some(esds) = esds {
                        let (oti, asc) = parse_esds(esds);
                        match oti {
                            0x40 | 0x66 | 0x67 | 0x68 => {
                                info.codec_name = "AAC".into();
                                info.codec = aac_codec_string(&asc);
                                info.description = asc;
                            }
                            0x69 | 0x6b => {
                                info.codec_name = "MP3".into();
                                info.codec = "mp3".into();
                            }
                            _ => {}
                        }
                    }
                }
                b"Opus" => {
                    info.codec_name = "Opus".into();
                    info.codec = "opus".into();
                    if let Some(d) = child(kids, b"dOps") {
                        info.description = opus_head_from_dops(d);
                    }
                }
                b"fLaC" => {
                    info.codec_name = "FLAC".into();
                    info.codec = "flac".into();
                    if let Some(d) = child(kids, b"dfLa") {
                        let mut desc = b"fLaC".to_vec();
                        desc.extend_from_slice(&d[4..]);
                        info.description = desc;
                    }
                }
                b".mp3" => {
                    info.codec_name = "MP3".into();
                    info.codec = "mp3".into();
                }
                b"ac-3" => info.codec_name = "AC-3".into(),
                b"ec-3" => info.codec_name = "E-AC-3".into(),
                _ => {}
            }
        }
    }
}

/// (objectTypeIndication, DecoderSpecificInfo) from an esds payload.
fn parse_esds(esds: &[u8]) -> (u8, Vec<u8>) {
    let mut r = R::new(esds);
    vf(&mut r);
    fn desc_len(r: &mut R) -> usize {
        let mut n = 0usize;
        for _ in 0..4 {
            let b = r.u8();
            n = n << 7 | (b & 0x7f) as usize;
            if b & 0x80 == 0 {
                break;
            }
        }
        n
    }
    let (mut oti, mut asc) = (0u8, Vec::new());
    while r.ok(2) {
        let tag = r.u8();
        let len = desc_len(&mut r);
        match tag {
            0x03 => {
                r.skip(2);
                let flags = r.u8();
                if flags & 0x80 != 0 {
                    r.skip(2);
                }
                if flags & 0x40 != 0 {
                    let n = r.u8() as usize;
                    r.skip(n);
                }
                if flags & 0x20 != 0 {
                    r.skip(2);
                }
            }
            0x04 => {
                oti = r.u8();
                r.skip(12);
            }
            0x05 => {
                let end = (r.1 + len).min(esds.len());
                asc = esds[r.1..end].to_vec();
                break;
            }
            _ => r.skip(len),
        }
    }
    (oti, asc)
}

fn sample_table(stbl: &[u8], ctx: &TrackCtx) -> Result<(Vec<Sample>, u64), String> {
    let runs = |name: &[u8; 4]| -> Vec<(u32, u32)> {
        child(stbl, name)
            .map(|b| {
                let mut r = R::new(b);
                vf(&mut r);
                let n = r.u32() as usize;
                records(&mut r, n, 8, |r| (r.u32(), r.u32()))
            })
            .unwrap_or_default()
    };
    let stts = runs(b"stts");
    let ctts = runs(b"ctts");
    let mut sizes: Vec<u32> = Vec::new();
    if let Some(b) = child(stbl, b"stsz") {
        let mut r = R::new(b);
        vf(&mut r);
        let fixed = r.u32();
        let n = r.u32() as usize;
        sizes = if fixed != 0 { vec![fixed; n] } else { records(&mut r, n, 4, |r| r.u32()) };
    } else if let Some(b) = child(stbl, b"stz2") {
        let mut r = R::new(b);
        vf(&mut r);
        r.skip(3);
        let field = r.u8();
        let n = r.u32() as usize;
        for i in 0..n {
            sizes.push(match field {
                4 => {
                    let byte = b.get(12 + i / 2).copied().unwrap_or(0);
                    if i % 2 == 0 { (byte >> 4) as u32 } else { (byte & 15) as u32 }
                }
                8 => r.u8() as u32,
                _ => r.u16() as u32,
            });
        }
    }
    let mut chunk_offsets: Vec<u64> = Vec::new();
    if let Some(b) = child(stbl, b"stco") {
        let mut r = R::new(b);
        vf(&mut r);
        let n = r.u32() as usize;
        chunk_offsets = records(&mut r, n, 4, |r| r.u32() as u64);
    } else if let Some(b) = child(stbl, b"co64") {
        let mut r = R::new(b);
        vf(&mut r);
        let n = r.u32() as usize;
        chunk_offsets = records(&mut r, n, 8, |r| r.u64());
    }
    let stsc: Vec<(u32, u32)> = child(stbl, b"stsc")
        .map(|b| {
            let mut r = R::new(b);
            vf(&mut r);
            let n = r.u32() as usize;
            records(&mut r, n, 12, |r| {
                let fc = r.u32();
                let spc = r.u32();
                r.skip(4);
                (fc, spc)
            })
        })
        .unwrap_or_default();
    let sync: Option<Vec<u32>> = child(stbl, b"stss").map(|b| {
        let mut r = R::new(b);
        vf(&mut r);
        let n = r.u32() as usize;
        records(&mut r, n, 4, |r| r.u32())
    });

    let n = sizes.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok((out, 0));
    }
    // Sample offsets from stsc + stco.
    let mut offsets = Vec::with_capacity(n);
    let mut si = 0usize;
    for (ci, &base) in chunk_offsets.iter().enumerate() {
        let chunk_no = ci as u32 + 1;
        let spc = stsc.iter().rev().find(|(fc, _)| *fc <= chunk_no).map(|e| e.1).unwrap_or(1);
        let mut off = base;
        for _ in 0..spc {
            if si >= n {
                break;
            }
            offsets.push(off);
            off += sizes[si] as u64;
            si += 1;
        }
    }
    if offsets.len() < n {
        return Err(format!("sample table covers {} of {n} samples", offsets.len()));
    }
    let ts = ctx.timescale as f64;
    let mut dts: u64 = 0;
    let mut stts_it = stts.iter().flat_map(|&(c, d)| std::iter::repeat_n(d, c as usize));
    let mut ctts_it = ctts.iter().flat_map(|&(c, o)| std::iter::repeat_n(o as i32 as i64, c as usize));
    let mut sync_it = sync.as_ref().map(|v| v.iter().peekable());
    for i in 0..n {
        let delta = stts_it.next().unwrap_or(0) as u64;
        let cto = ctts_it.next().unwrap_or(0);
        let keyframe = match &mut sync_it {
            None => true,
            Some(it) => {
                while it.peek().is_some_and(|&&s| s < i as u32 + 1) {
                    it.next();
                }
                it.peek().is_some_and(|&&s| s == i as u32 + 1)
            }
        };
        out.push(Sample {
            offset: offsets[i],
            size: sizes[i],
            pts: (dts as i64 + cto - ctx.edit_shift) as f64 / ts + ctx.edit_delay,
            dts: (dts as i64 - ctx.edit_shift) as f64 / ts + ctx.edit_delay,
            keyframe,
        });
        dts += delta;
    }
    Ok((out, dts))
}

fn parse_moof(moof_off: u64, moof: &[u8], tracks: &mut [TrackInfo], ctxs: &mut [TrackCtx]) {
    for (t, traf) in boxes(moof) {
        if &t != b"traf" {
            continue;
        }
        let Some(tfhd) = child(traf, b"tfhd") else { continue };
        let mut r = R::new(tfhd);
        let (_, fl) = vf(&mut r);
        let id = r.u32();
        let Some(ti) = tracks.iter().position(|t| t.id == id) else { continue };
        let ctx = &mut ctxs[ti];
        let mut base = moof_off;
        if fl & 0x1 != 0 {
            base = r.u64();
        }
        if fl & 0x2 != 0 {
            r.skip(4);
        }
        let def_dur = if fl & 0x8 != 0 { r.u32() } else { ctx.trex.duration };
        let def_size = if fl & 0x10 != 0 { r.u32() } else { ctx.trex.size };
        let def_flags = if fl & 0x20 != 0 { r.u32() } else { ctx.trex.flags };
        if let Some(tfdt) = child(traf, b"tfdt") {
            let mut r = R::new(tfdt);
            let (v, _) = vf(&mut r);
            ctx.next_dts = if v == 1 { r.u64() } else { r.u32() as u64 };
        }
        let ts = ctx.timescale as f64;
        for (t, trun) in boxes(traf) {
            if &t != b"trun" {
                continue;
            }
            let mut r = R::new(trun);
            let (v, tf) = vf(&mut r);
            let count = r.u32();
            let mut off = if tf & 0x1 != 0 { (base as i64 + r.u32() as i32 as i64) as u64 } else { base };
            let first_flags = if tf & 0x4 != 0 { Some(r.u32()) } else { None };
            for i in 0..count {
                if !r.ok(4) && tf & 0xf00 != 0 {
                    break;
                }
                let dur = if tf & 0x100 != 0 { r.u32() } else { def_dur };
                let size = if tf & 0x200 != 0 { r.u32() } else { def_size };
                let flags = if tf & 0x400 != 0 { r.u32() } else if i == 0 { first_flags.unwrap_or(def_flags) } else { def_flags };
                let cto = if tf & 0x800 != 0 { if v == 0 { r.u32() as i64 } else { r.u32() as i32 as i64 } } else { 0 };
                let dts = ctx.next_dts as i64 - ctx.edit_shift;
                tracks[ti].samples.push(Sample {
                    offset: off,
                    size,
                    pts: (dts + cto) as f64 / ts + ctx.edit_delay,
                    dts: dts as f64 / ts + ctx.edit_delay,
                    keyframe: flags & 0x10000 == 0,
                });
                off += size as u64;
                ctx.next_dts += dur as u64;
            }
            base = off;
        }
    }
}
