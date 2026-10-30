//! Matroska / WebM indexing.
//!
//! An incremental EBML walker: the caller feeds file bytes from `want()`
//! onwards; the parser descends into the elements it needs, reads block
//! headers, and skips everything else (including frame payloads), so the
//! caller only has to read a small part of a large file.

use crate::media::*;

const SEGMENT: u32 = 0x18538067;
const INFO: u32 = 0x1549A966;
const TIMECODE_SCALE: u32 = 0x2AD7B1;
const DURATION: u32 = 0x4489;
const TRACKS: u32 = 0x1654AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const DEFAULT_DURATION: u32 = 0x23E383;
const VIDEO: u32 = 0xE0;
const PIXEL_WIDTH: u32 = 0xB0;
const PIXEL_HEIGHT: u32 = 0xBA;
const AUDIO: u32 = 0xE1;
const SAMPLING_FREQ: u32 = 0xB5;
const CHANNELS: u32 = 0x9F;
const CONTENT_ENCODINGS: u32 = 0x6D80;
const CLUSTER: u32 = 0x1F43B675;
const CLUSTER_TIMECODE: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
const REFERENCE_BLOCK: u32 = 0xFB;

/// Elements that may legitimately follow an unknown-size Cluster.
const SEGMENT_CHILDREN: [u32; 8] =
    [CLUSTER, 0x1C53BB6B, 0x1254C367, 0x1043A770, 0x1941A469, 0x114D9B74, INFO, TRACKS];

fn is_master(id: u32) -> bool {
    matches!(id, SEGMENT | INFO | TRACKS | TRACK_ENTRY | VIDEO | AUDIO | CLUSTER | BLOCK_GROUP)
}

/// Leaf elements whose payload we need in full.
fn wanted_leaf(id: u32) -> bool {
    matches!(
        id,
        TIMECODE_SCALE | DURATION | TRACK_NUMBER | TRACK_TYPE | CODEC_ID | CODEC_PRIVATE | DEFAULT_DURATION
            | PIXEL_WIDTH | PIXEL_HEIGHT | SAMPLING_FREQ | CHANNELS | CLUSTER_TIMECODE | REFERENCE_BLOCK
            | CONTENT_ENCODINGS
    )
}

/// EBML variable-length integer: (value, length). `id` keeps the marker bit.
fn vint(d: &[u8], id: bool) -> Option<(u64, usize)> {
    let b = *d.first()?;
    if b == 0 {
        return None;
    }
    let len = b.leading_zeros() as usize + 1;
    if len > 8 || d.len() < len {
        return None;
    }
    let mut v = if id { b as u64 } else { (b as u64) & (0xff >> len) };
    for &x in &d[1..len] {
        v = v << 8 | x as u64;
    }
    Some((v, len))
}

const UNKNOWN: u64 = u64::MAX;

#[derive(Default)]
struct MkvTrack {
    number: u64,
    typ: u64,
    codec_id: String,
    private: Vec<u8>,
    default_duration_ns: u64,
    width: u32,
    height: u32,
    sample_rate: f64,
    channels: u32,
    encoded: bool,
}

struct PendingBlock {
    track: u64,
    tc: i64,
    frames: Vec<(u64, u32)>,
    /// SimpleBlocks carry a keyframe flag; Blocks are keyframes unless a
    /// ReferenceBlock accompanies them.
    simple: bool,
    simple_key: bool,
    has_ref: bool,
}

pub struct MkvParser {
    pos: u64,
    stack: Vec<(u32, u64)>,
    tracks: Vec<MkvTrack>,
    cur: Option<MkvTrack>,
    out: Vec<(u64, Vec<Sample>)>,
    scale_ns: u64,
    duration: f64,
    cluster_tc: i64,
    block: Option<PendingBlock>,
    pub error: Option<String>,
    done: bool,
}

impl Default for MkvParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MkvParser {
    pub fn new() -> Self {
        MkvParser {
            pos: 0,
            stack: Vec::new(),
            tracks: Vec::new(),
            cur: None,
            out: Vec::new(),
            scale_ns: 1_000_000,
            duration: 0.0,
            cluster_tc: 0,
            block: None,
            error: None,
            done: false,
        }
    }

    /// File offset the next `feed` must start at.
    pub fn want(&self) -> u64 {
        self.pos
    }

    pub fn done(&self) -> bool {
        self.done
    }

    fn close_until(&mut self, pos: u64) {
        while let Some(&(id, end)) = self.stack.last() {
            if end != UNKNOWN && end <= pos {
                self.stack.pop();
                self.closed(id);
            } else {
                break;
            }
        }
    }

    fn closed(&mut self, id: u32) {
        match id {
            TRACK_ENTRY => {
                if let Some(t) = self.cur.take() {
                    self.tracks.push(t);
                }
            }
            BLOCK_GROUP => self.flush_block(),
            _ => {}
        }
    }

    fn flush_block(&mut self) {
        let Some(b) = self.block.take() else { return };
        let Some(t) = self.tracks.iter().find(|t| t.number == b.track) else { return };
        let key = if b.simple { b.simple_key } else { !b.has_ref };
        let tick = self.scale_ns as f64 / 1e9;
        let base = (self.cluster_tc + b.tc) as f64 * tick;
        let step = t.default_duration_ns as f64 / 1e9;
        let list = match self.out.iter_mut().find(|(n, _)| *n == b.track) {
            Some((_, l)) => l,
            None => {
                self.out.push((b.track, Vec::new()));
                &mut self.out.last_mut().unwrap().1
            }
        };
        for (i, &(offset, size)) in b.frames.iter().enumerate() {
            let pts = base + i as f64 * step;
            list.push(Sample { offset, size, pts, dts: pts, keyframe: key });
        }
    }

    /// Parses as much of `data` (which starts at file offset `at` ==
    /// `want()`) as possible. With `eof`, the data reaches the end of file.
    pub fn feed(&mut self, at: u64, data: &[u8], eof: bool) {
        if self.done || at != self.pos {
            return;
        }
        let end_of_data = at + data.len() as u64;
        loop {
            self.close_until(self.pos);
            let rel = (self.pos - at) as usize;
            if rel >= data.len() {
                if eof {
                    self.finish_all();
                }
                return;
            }
            let d = &data[rel..];
            let Some((id, il)) = vint(d, true) else {
                if eof || d.len() >= 8 {
                    // Junk or truncated: stop indexing here.
                    self.finish_all();
                }
                return;
            };
            let Some((size, sl)) = vint(&d[il..], false) else {
                if eof {
                    self.finish_all();
                }
                return;
            };
            let id = id as u32;
            let hdr = (il + sl) as u64;
            let unknown = size == (1u64 << (7 * sl)) - 1;
            let payload_at = self.pos + hdr;

            // An unknown-size cluster ends where the next segment-level element begins.
            if SEGMENT_CHILDREN.contains(&id) {
                while let Some(&(cid, cend)) = self.stack.last() {
                    if cid == CLUSTER && cend == UNKNOWN {
                        self.stack.pop();
                        self.closed(CLUSTER);
                    } else {
                        break;
                    }
                }
            }

            if is_master(id) {
                let end = if unknown { UNKNOWN } else { payload_at + size };
                self.stack.push((id, end));
                match id {
                    TRACK_ENTRY => self.cur = Some(MkvTrack::default()),
                    BLOCK_GROUP => self.block = None,
                    _ => {}
                }
                self.pos = payload_at;
                continue;
            }
            if unknown {
                self.error = Some(format!("unknown-size element {id:x}"));
                self.finish_all();
                return;
            }
            let pend = payload_at + size;
            if id == SIMPLE_BLOCK || id == BLOCK {
                // Need the block header (and lacing table) only.
                let need = size.min(64 * 1024);
                if payload_at + need > end_of_data && !eof {
                    return;
                }
                let avail = (end_of_data.min(payload_at + need) - payload_at) as usize;
                let p = &data[(payload_at - at) as usize..][..avail];
                if let Some(b) = parse_block_header(p, payload_at, size, id == SIMPLE_BLOCK) {
                    if id == SIMPLE_BLOCK {
                        self.block = Some(b);
                        self.flush_block();
                    } else {
                        let has_ref = self.block.as_ref().map(|b| b.has_ref).unwrap_or(false);
                        self.block = Some(PendingBlock { has_ref, ..b });
                    }
                }
                self.pos = pend;
                continue;
            }
            if wanted_leaf(id) {
                if pend > end_of_data {
                    if eof {
                        self.finish_all();
                    }
                    return;
                }
                let p = &data[(payload_at - at) as usize..(pend - at) as usize];
                self.leaf(id, p);
            }
            self.pos = pend;
        }
    }

    fn leaf(&mut self, id: u32, p: &[u8]) {
        let uint = || p.iter().fold(0u64, |a, &b| a << 8 | b as u64);
        let float = || match p.len() {
            4 => f32::from_be_bytes(p.try_into().unwrap()) as f64,
            8 => f64::from_be_bytes(p.try_into().unwrap()),
            _ => 0.0,
        };
        match id {
            TIMECODE_SCALE => self.scale_ns = uint().max(1),
            DURATION => self.duration = float(),
            CLUSTER_TIMECODE => self.cluster_tc = uint() as i64,
            REFERENCE_BLOCK => {
                if let Some(b) = &mut self.block {
                    b.has_ref = true;
                } else {
                    self.block = Some(PendingBlock { track: 0, tc: 0, frames: vec![], simple: false, simple_key: false, has_ref: true });
                }
            }
            _ => {
                let Some(t) = &mut self.cur else { return };
                match id {
                    TRACK_NUMBER => t.number = uint(),
                    TRACK_TYPE => t.typ = uint(),
                    CODEC_ID => t.codec_id = String::from_utf8_lossy(p).trim_end_matches('\0').to_string(),
                    CODEC_PRIVATE => t.private = p.to_vec(),
                    DEFAULT_DURATION => t.default_duration_ns = uint(),
                    PIXEL_WIDTH => t.width = uint() as u32,
                    PIXEL_HEIGHT => t.height = uint() as u32,
                    SAMPLING_FREQ => t.sample_rate = float(),
                    CHANNELS => t.channels = uint() as u32,
                    CONTENT_ENCODINGS => t.encoded = true,
                    _ => {}
                }
            }
        }
    }

    fn finish_all(&mut self) {
        while let Some((id, _)) = self.stack.pop() {
            self.closed(id);
        }
        self.done = true;
    }

    pub fn tracks(&self) -> Vec<TrackInfo> {
        let mut out = Vec::new();
        for t in &self.tracks {
            let kind = match t.typ {
                1 => Kind::Video,
                2 => Kind::Audio,
                _ => continue,
            };
            let mut info = TrackInfo::new(t.number as u32, kind);
            info.width = t.width;
            info.height = t.height;
            info.sample_rate = t.sample_rate.round() as u32;
            info.channels = t.channels;
            info.duration = self.duration * self.scale_ns as f64 / 1e9;
            map_codec(t, &mut info);
            if t.encoded {
                // Header stripping / compressed tracks aren't supported.
                info.codec.clear();
                info.codec_name.push_str(" (ContentEncoding)");
            }
            info.samples = self.out.iter().find(|(n, _)| *n == t.number).map(|(_, l)| l.clone()).unwrap_or_default();
            if !info.samples.is_empty() {
                out.push(info);
            }
        }
        out
    }
}

fn parse_block_header(p: &[u8], payload_at: u64, size: u64, simple: bool) -> Option<PendingBlock> {
    let (track, tl) = vint(p, false)?;
    if p.len() < tl + 3 {
        return None;
    }
    let tc = i16::from_be_bytes([p[tl], p[tl + 1]]) as i64;
    let flags = p[tl + 2];
    let mut pos = tl + 3;
    let lacing = (flags >> 1) & 3;
    let total = size as usize;
    let mut frames = Vec::new();
    if lacing == 0 {
        frames.push((payload_at + pos as u64, (total - pos) as u32));
    } else {
        let n = *p.get(pos)? as usize + 1;
        pos += 1;
        let mut sizes = Vec::with_capacity(n);
        match lacing {
            1 => {
                for _ in 0..n - 1 {
                    let mut s = 0usize;
                    loop {
                        let b = *p.get(pos)? as usize;
                        pos += 1;
                        s += b;
                        if b != 255 {
                            break;
                        }
                    }
                    sizes.push(s);
                }
            }
            3 => {
                let (first, l) = vint(&p[pos..], false)?;
                pos += l;
                sizes.push(first as usize);
                let mut prev = first as i64;
                for _ in 1..n - 1 {
                    let (raw, l) = vint(&p[pos..], false)?;
                    pos += l;
                    let bias = (1i64 << (7 * l - 1)) - 1;
                    prev += raw as i64 - bias;
                    sizes.push(prev.max(0) as usize);
                }
            }
            _ => {
                let each = (total - pos) / n;
                sizes = vec![each; n - 1];
            }
        }
        let used: usize = sizes.iter().sum();
        sizes.push(total.checked_sub(pos + used)?);
        let mut off = payload_at + pos as u64;
        for s in sizes {
            frames.push((off, s as u32));
            off += s as u64;
        }
    }
    Some(PendingBlock { track, tc, frames, simple, simple_key: simple && flags & 0x80 != 0, has_ref: false })
}

fn map_codec(t: &MkvTrack, info: &mut TrackInfo) {
    let p = &t.private;
    let (name, codec, desc): (&str, String, Vec<u8>) = match t.codec_id.as_str() {
        "V_MPEG4/ISO/AVC" => ("H.264", avc_codec_string(p).unwrap_or_default(), p.clone()),
        "V_MPEGH/ISO/HEVC" => ("HEVC", hevc_codec_string("hvc1", p).unwrap_or_default(), p.clone()),
        "V_VP8" => ("VP8", "vp8".into(), vec![]),
        "V_VP9" => ("VP9", "vp09.00.10.08".into(), vec![]),
        "V_AV1" => ("AV1", av1_codec_string(p).unwrap_or_default(), p.clone()),
        "A_OPUS" => ("Opus", "opus".into(), p.clone()),
        "A_VORBIS" => ("Vorbis", "vorbis".into(), p.clone()),
        "A_FLAC" => ("FLAC", "flac".into(), p.clone()),
        "A_MPEG/L3" => ("MP3", "mp3".into(), vec![]),
        id if id.starts_with("A_AAC") => {
            let asc = if p.len() >= 2 { p.clone() } else { synth_asc(t.sample_rate as u32, t.channels) };
            ("AAC", aac_codec_string(&asc), asc)
        }
        id => (id, String::new(), vec![]),
    };
    info.codec_name = name.to_string();
    info.codec = codec;
    info.description = desc;
}

/// AAC-LC AudioSpecificConfig for legacy A_AAC/* tracks without one.
fn synth_asc(rate: u32, channels: u32) -> Vec<u8> {
    const RATES: [u32; 12] = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000];
    let idx = RATES.iter().position(|&r| r == rate).unwrap_or(4) as u16;
    let v: u16 = (2 << 11) | (idx << 7) | ((channels.min(7) as u16) << 3);
    v.to_be_bytes().to_vec()
}
