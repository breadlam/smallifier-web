//! MP4 writer for the output: one H.264 track (+ optional Opus track),
//! `moov` first (fast start), media interleaved in ~0.5 s runs.

pub struct VideoTrack<'a> {
    pub width: u32,
    pub height: u32,
    /// Frame rate as a rational.
    pub fps_num: u32,
    pub fps_den: u32,
    pub sps: &'a [u8],
    pub pps: &'a [u8],
    /// Clockwise rotation to signal (0/90/180/270).
    pub rotation: u32,
    /// In decode order. `pts` is in frames.
    pub samples: Vec<VSample<'a>>,
}

pub struct VSample<'a> {
    pub data: &'a [u8],
    pub pts: i64,
    pub keyframe: bool,
}

pub struct AudioTrack<'a> {
    pub channels: u32,
    pub pre_skip: u32,
    /// Original input rate, informational (dOps).
    pub input_rate: u32,
    /// 20 ms Opus packets (960 samples at 48 kHz).
    pub packets: Vec<&'a [u8]>,
    /// Valid samples after pre-skip (sets the edit list duration).
    pub valid_samples: u64,
}

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v)
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_be_bytes())
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes())
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_be_bytes())
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b)
    }
    fn zeros(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, 0)
    }
    /// Writes a box; `f` fills the payload.
    fn bx(&mut self, name: &[u8; 4], f: impl FnOnce(&mut W)) {
        let start = self.0.len();
        self.u32(0);
        self.bytes(name);
        f(self);
        let size = (self.0.len() - start) as u32;
        self.0[start..start + 4].copy_from_slice(&size.to_be_bytes());
    }
    fn full(&mut self, name: &[u8; 4], version: u8, flags: u32, f: impl FnOnce(&mut W)) {
        self.bx(name, |w| {
            w.u32((version as u32) << 24 | flags);
            f(w);
        })
    }
}

const MOVIE_SCALE: u32 = 1000;

fn matrix(w: &mut W, rotation: u32) {
    let (a, b, c, d): (i32, i32, i32, i32) = match rotation {
        90 => (0, 1, -1, 0),
        180 => (-1, 0, 0, -1),
        270 => (0, -1, 1, 0),
        _ => (1, 0, 0, 1),
    };
    for v in [a << 16, b << 16, 0, c << 16, d << 16, 0, 0, 0, 0x4000_0000] {
        w.u32(v as u32);
    }
}

/// One interleaving run: (is_audio, first sample index, count).
type Run = (bool, usize, usize);

fn plan_runs(v: &VideoTrack, a: Option<&AudioTrack>) -> Vec<Run> {
    let fps = v.fps_num as f64 / v.fps_den as f64;
    let per_v = ((fps * 0.5).round() as usize).max(1);
    let per_a = 25; // 0.5 s of 20 ms packets
    let mut runs = Vec::new();
    let (mut vi, mut ai) = (0, 0);
    let na = a.map(|a| a.packets.len()).unwrap_or(0);
    while vi < v.samples.len() || ai < na {
        if vi < v.samples.len() {
            let n = per_v.min(v.samples.len() - vi);
            runs.push((false, vi, n));
            vi += n;
        }
        if ai < na {
            let n = per_a.min(na - ai);
            runs.push((true, ai, n));
            ai += n;
        }
    }
    runs
}

pub fn write(v: &VideoTrack, a: Option<&AudioTrack>) -> Vec<u8> {
    let runs = plan_runs(v, a);
    // Pass 1 with zero offsets to learn the moov size, pass 2 for real.
    let mut ftyp = W(Vec::new());
    ftyp.bx(b"ftyp", |w| {
        w.bytes(b"isom");
        w.u32(0x200);
        for b in [b"isom", b"iso2", b"avc1", b"mp41"] {
            w.bytes(b);
        }
    });
    let mdat_payload: usize = v.samples.iter().map(|s| s.data.len()).sum::<usize>()
        + a.map(|a| a.packets.iter().map(|p| p.len()).sum::<usize>()).unwrap_or(0);
    let large = mdat_payload as u64 + 16 > u32::MAX as u64;
    let probe = moov(v, a, &runs, 0, large);
    let mdat_hdr = if large { 16 } else { 8 };
    let data_start = (ftyp.0.len() + probe.len() + mdat_hdr) as u64;
    let moov_bytes = moov(v, a, &runs, data_start, large);
    debug_assert_eq!(moov_bytes.len(), probe.len());

    let mut out = ftyp.0;
    out.extend_from_slice(&moov_bytes);
    if large {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(mdat_payload as u64 + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&(mdat_payload as u32 + 8).to_be_bytes());
        out.extend_from_slice(b"mdat");
    }
    for &(audio, first, n) in &runs {
        for i in first..first + n {
            if audio {
                out.extend_from_slice(a.unwrap().packets[i]);
            } else {
                out.extend_from_slice(v.samples[i].data);
            }
        }
    }
    out
}

fn moov(v: &VideoTrack, a: Option<&AudioTrack>, runs: &[Run], data_start: u64, large: bool) -> Vec<u8> {
    // Chunk offsets per track, following the run order.
    let (mut voff, mut aoff) = (Vec::new(), Vec::new());
    let mut pos = data_start;
    for &(audio, first, n) in runs {
        let size: u64 = if audio {
            a.unwrap().packets[first..first + n].iter().map(|p| p.len() as u64).sum()
        } else {
            v.samples[first..first + n].iter().map(|s| s.data.len() as u64).sum()
        };
        if audio {
            aoff.push((pos, n));
        } else {
            voff.push((pos, n));
        }
        pos += size;
    }

    let vscale = v.fps_num * if v.fps_num < 1000 { 1000 } else { 1 };
    let vdur = v.fps_den as u64 * if v.fps_num < 1000 { 1000 } else { 1 };
    let nframes = v.samples.len() as u64;
    // Composition offsets: dts = n, cts = pts + shift, shift = max B delay.
    let shift = v.samples.iter().enumerate().map(|(n, s)| n as i64 - s.pts).max().unwrap_or(0).max(0);
    let video_media_dur = nframes * vdur;
    let video_movie_dur = video_media_dur * MOVIE_SCALE as u64 / vscale as u64;
    let audio_movie_dur = a.map(|a| a.valid_samples * MOVIE_SCALE as u64 / 48000).unwrap_or(0);

    let mut w = W(Vec::new());
    w.bx(b"moov", |w| {
        w.full(b"mvhd", 0, 0, |w| {
            w.u32(0);
            w.u32(0);
            w.u32(MOVIE_SCALE);
            w.u32(video_movie_dur.max(audio_movie_dur) as u32);
            w.u32(0x0001_0000);
            w.u16(0x0100);
            w.zeros(10);
            matrix(w, 0);
            w.zeros(24);
            w.u32(if a.is_some() { 3 } else { 2 });
        });
        // ---- video trak
        w.bx(b"trak", |w| {
            w.full(b"tkhd", 0, 3, |w| {
                w.u32(0);
                w.u32(0);
                w.u32(1);
                w.u32(0);
                w.u32(video_movie_dur as u32);
                w.zeros(8);
                w.u16(0);
                w.u16(0);
                w.u16(0);
                w.u16(0);
                matrix(w, v.rotation);
                w.u32(v.width << 16);
                w.u32(v.height << 16);
            });
            w.bx(b"edts", |w| {
                w.full(b"elst", 0, 0, |w| {
                    w.u32(1);
                    w.u32(video_movie_dur as u32);
                    w.u32((shift as u64 * vdur) as u32);
                    w.u32(0x0001_0000);
                });
            });
            w.bx(b"mdia", |w| {
                w.full(b"mdhd", 0, 0, |w| {
                    w.u32(0);
                    w.u32(0);
                    w.u32(vscale);
                    w.u32(video_media_dur as u32);
                    w.u16(0x55c4); // 'und'
                    w.u16(0);
                });
                hdlr(w, b"vide", "VideoHandler");
                w.bx(b"minf", |w| {
                    w.full(b"vmhd", 0, 1, |w| w.zeros(8));
                    dinf(w);
                    w.bx(b"stbl", |w| {
                        w.full(b"stsd", 0, 0, |w| {
                            w.u32(1);
                            w.bx(b"avc1", |w| {
                                w.zeros(6);
                                w.u16(1);
                                w.zeros(16);
                                w.u16(v.width as u16);
                                w.u16(v.height as u16);
                                w.u32(0x0048_0000);
                                w.u32(0x0048_0000);
                                w.u32(0);
                                w.u16(1);
                                let mut name = [0u8; 32];
                                let label = b"smallifier";
                                name[0] = label.len() as u8;
                                name[1..1 + label.len()].copy_from_slice(label);
                                w.bytes(&name);
                                w.u16(0x18);
                                w.u16(0xffff);
                                w.bx(b"avcC", |w| {
                                    w.u8(1);
                                    w.u8(v.sps[1]);
                                    w.u8(v.sps[2]);
                                    w.u8(v.sps[3]);
                                    w.u8(0xff);
                                    w.u8(0xe1);
                                    w.u16(v.sps.len() as u16);
                                    w.bytes(v.sps);
                                    w.u8(1);
                                    w.u16(v.pps.len() as u16);
                                    w.bytes(v.pps);
                                    if matches!(v.sps[1], 100 | 110 | 122 | 144) {
                                        w.u8(0xfc | 1); // 4:2:0
                                        w.u8(0xf8);
                                        w.u8(0xf8);
                                        w.u8(0);
                                    }
                                });
                                w.bx(b"colr", |w| {
                                    w.bytes(b"nclx");
                                    w.u16(1);
                                    w.u16(1);
                                    w.u16(1);
                                    w.u8(0);
                                });
                            });
                        });
                        w.full(b"stts", 0, 0, |w| {
                            w.u32(1);
                            w.u32(nframes as u32);
                            w.u32(vdur as u32);
                        });
                        // ctts, run-length coded.
                        let mut ctts: Vec<(u32, u32)> = Vec::new();
                        for (n, s) in v.samples.iter().enumerate() {
                            let off = ((s.pts - n as i64 + shift) as u64 * vdur) as u32;
                            match ctts.last_mut() {
                                Some(e) if e.1 == off => e.0 += 1,
                                _ => ctts.push((1, off)),
                            }
                        }
                        if ctts.len() > 1 || ctts.first().is_some_and(|e| e.1 != 0) {
                            w.full(b"ctts", 0, 0, |w| {
                                w.u32(ctts.len() as u32);
                                for (c, o) in &ctts {
                                    w.u32(*c);
                                    w.u32(*o);
                                }
                            });
                        }
                        let keys: Vec<u32> =
                            v.samples.iter().enumerate().filter(|(_, s)| s.keyframe).map(|(i, _)| i as u32 + 1).collect();
                        w.full(b"stss", 0, 0, |w| {
                            w.u32(keys.len() as u32);
                            for k in &keys {
                                w.u32(*k);
                            }
                        });
                        stsc(w, &voff);
                        w.full(b"stsz", 0, 0, |w| {
                            w.u32(0);
                            w.u32(nframes as u32);
                            for s in &v.samples {
                                w.u32(s.data.len() as u32);
                            }
                        });
                        stco(w, &voff, large);
                    });
                });
            });
        });
        // ---- audio trak
        if let Some(a) = a {
            let media_dur = a.packets.len() as u64 * 960;
            w.bx(b"trak", |w| {
                w.full(b"tkhd", 0, 3, |w| {
                    w.u32(0);
                    w.u32(0);
                    w.u32(2);
                    w.u32(0);
                    w.u32(audio_movie_dur as u32);
                    w.zeros(8);
                    w.u16(0);
                    w.u16(1);
                    w.u16(0x0100);
                    w.u16(0);
                    matrix(w, 0);
                    w.u32(0);
                    w.u32(0);
                });
                w.bx(b"edts", |w| {
                    w.full(b"elst", 0, 0, |w| {
                        w.u32(1);
                        w.u32(audio_movie_dur as u32);
                        w.u32(a.pre_skip);
                        w.u32(0x0001_0000);
                    });
                });
                w.bx(b"mdia", |w| {
                    w.full(b"mdhd", 0, 0, |w| {
                        w.u32(0);
                        w.u32(0);
                        w.u32(48000);
                        w.u32(media_dur as u32);
                        w.u16(0x55c4);
                        w.u16(0);
                    });
                    hdlr(w, b"soun", "SoundHandler");
                    w.bx(b"minf", |w| {
                        w.full(b"smhd", 0, 0, |w| w.u32(0));
                        dinf(w);
                        w.bx(b"stbl", |w| {
                            w.full(b"stsd", 0, 0, |w| {
                                w.u32(1);
                                w.bx(b"Opus", |w| {
                                    w.zeros(6);
                                    w.u16(1);
                                    w.zeros(8);
                                    w.u16(a.channels as u16);
                                    w.u16(16);
                                    w.u16(0);
                                    w.u16(0);
                                    w.u32(48000 << 16);
                                    w.bx(b"dOps", |w| {
                                        w.u8(0);
                                        w.u8(a.channels as u8);
                                        w.u16(a.pre_skip as u16);
                                        w.u32(a.input_rate);
                                        w.u16(0);
                                        w.u8(0);
                                    });
                                });
                            });
                            w.full(b"stts", 0, 0, |w| {
                                w.u32(1);
                                w.u32(a.packets.len() as u32);
                                w.u32(960);
                            });
                            stsc(w, &aoff);
                            w.full(b"stsz", 0, 0, |w| {
                                w.u32(0);
                                w.u32(a.packets.len() as u32);
                                for p in &a.packets {
                                    w.u32(p.len() as u32);
                                }
                            });
                            stco(w, &aoff, large);
                        });
                    });
                });
            });
        }
    });
    w.0
}

fn hdlr(w: &mut W, typ: &[u8; 4], name: &str) {
    w.full(b"hdlr", 0, 0, |w| {
        w.u32(0);
        w.bytes(typ);
        w.zeros(12);
        w.bytes(name.as_bytes());
        w.u8(0);
    });
}

fn dinf(w: &mut W) {
    w.bx(b"dinf", |w| {
        w.full(b"dref", 0, 0, |w| {
            w.u32(1);
            w.full(b"url ", 0, 1, |_| {});
        });
    });
}

fn stsc(w: &mut W, chunks: &[(u64, usize)]) {
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for (i, &(_, n)) in chunks.iter().enumerate() {
        if entries.last().map(|e| e.1) != Some(n as u32) {
            entries.push((i as u32 + 1, n as u32));
        }
    }
    w.full(b"stsc", 0, 0, |w| {
        w.u32(entries.len() as u32);
        for (first, n) in entries {
            w.u32(first);
            w.u32(n);
            w.u32(1);
        }
    });
}

fn stco(w: &mut W, chunks: &[(u64, usize)], large: bool) {
    if large {
        w.full(b"co64", 0, 0, |w| {
            w.u32(chunks.len() as u32);
            for &(o, _) in chunks {
                w.u64(o);
            }
        });
    } else {
        w.full(b"stco", 0, 0, |w| {
            w.u32(chunks.len() as u32);
            for &(o, _) in chunks {
                w.u32(o as u32);
            }
        });
    }
}

/// Bytes the container adds on top of the media payload, for budgeting
/// before anything is encoded (exact up to the ctts run count, which is
/// bounded by the frame count).
pub fn overhead_estimate(frames: usize, fps: f64, audio_packets: usize, has_audio: bool) -> usize {
    let runs_v = (frames as f64 / (fps * 0.5).max(1.0)).ceil() as usize;
    let runs_a = audio_packets.div_ceil(25);
    let base = 1200 + if has_audio { 600 } else { 0 };
    // stsz 4/sample, ctts <= 8/frame (assume worst case), stss ~4 per 2 s,
    // stco 4 per run.
    base + frames * (4 + 8) + frames / 30 + audio_packets * 4 + (runs_v + runs_a) * 4
}
