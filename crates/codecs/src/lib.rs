//! Safe wrappers over the vendored x264 and libopus encoders.

use std::ffi::{CString, c_char, c_int, c_void};

unsafe extern "C" {
    fn sx_open(w: c_int, h: c_int, fps_num: c_int, fps_den: c_int, preset: *const c_char,
               tune: *const c_char, opts: *const c_char, want_stats: c_int) -> *mut c_void;
    fn sx_sps(e: *mut c_void, len: *mut c_int) -> *const u8;
    fn sx_pps(e: *mut c_void, len: *mut c_int) -> *const u8;
    fn sx_encode(e: *mut c_void, y: *const u8, u: *const u8, v: *const u8, stride_y: c_int,
                 stride_c: c_int, pts: i64, force_idr: c_int) -> c_int;
    fn sx_flush(e: *mut c_void) -> c_int;
    fn sx_data(e: *mut c_void) -> *const u8;
    fn sx_pts(e: *mut c_void) -> i64;
    fn sx_dts(e: *mut c_void) -> i64;
    fn sx_keyframe(e: *mut c_void) -> c_int;
    fn sx_psnr_y(e: *mut c_void) -> f64;
    fn sx_ssim(e: *mut c_void) -> f64;
    fn sx_close(e: *mut c_void);

    fn so_open(sample_rate: c_int, channels: c_int, bitrate: c_int, err: *mut c_int) -> *mut c_void;
    fn so_lookahead(e: *mut c_void) -> c_int;
    fn so_encode(e: *mut c_void, pcm: *const f32, frame_size: c_int, out: *mut u8, cap: c_int) -> c_int;
    fn so_close(e: *mut c_void);
}

#[derive(Clone, Debug)]
pub struct X264Config {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub preset: String,
    pub tune: String,
    /// x264 options as "key=value:key=value", applied after the preset.
    pub opts: String,
    /// Ask x264 to measure per-frame PSNR/SSIM against its input.
    pub stats: bool,
}

/// One encoded access unit: length-prefixed NAL units, SEI removed.
#[derive(Clone, Debug)]
pub struct Packet {
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub keyframe: bool,
    pub psnr_y: f64,
    pub ssim: f64,
}

pub struct X264 {
    raw: *mut c_void,
}

// The encoder is single-threaded (built without pthreads) and owned.
unsafe impl Send for X264 {}

impl X264 {
    pub fn open(cfg: &X264Config) -> Result<Self, String> {
        let preset = CString::new(cfg.preset.as_str()).unwrap();
        let tune = CString::new(cfg.tune.as_str()).unwrap();
        let opts = CString::new(cfg.opts.as_str()).unwrap();
        let raw = unsafe {
            sx_open(cfg.width as c_int, cfg.height as c_int, cfg.fps_num as c_int,
                    cfg.fps_den as c_int, preset.as_ptr(), tune.as_ptr(), opts.as_ptr(),
                    cfg.stats as c_int)
        };
        if raw.is_null() {
            return Err(format!("x264 rejected config {cfg:?}"));
        }
        Ok(X264 { raw })
    }

    pub fn sps(&self) -> Vec<u8> {
        let mut n = 0;
        let p = unsafe { sx_sps(self.raw, &mut n) };
        unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec()
    }

    pub fn pps(&self) -> Vec<u8> {
        let mut n = 0;
        let p = unsafe { sx_pps(self.raw, &mut n) };
        unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec()
    }

    fn take(&self, r: c_int) -> Result<Option<Packet>, String> {
        if r < 0 {
            return Err("x264 encode failed".into());
        }
        if r == 0 {
            return Ok(None);
        }
        unsafe {
            Ok(Some(Packet {
                data: std::slice::from_raw_parts(sx_data(self.raw), r as usize).to_vec(),
                pts: sx_pts(self.raw),
                dts: sx_dts(self.raw),
                keyframe: sx_keyframe(self.raw) != 0,
                psnr_y: sx_psnr_y(self.raw),
                ssim: sx_ssim(self.raw),
            }))
        }
    }

    /// Planes are I420 at the configured size.
    pub fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8], stride_y: usize, stride_c: usize,
                  pts: i64, force_idr: bool) -> Result<Option<Packet>, String> {
        let r = unsafe {
            sx_encode(self.raw, y.as_ptr(), u.as_ptr(), v.as_ptr(), stride_y as c_int,
                      stride_c as c_int, pts, force_idr as c_int)
        };
        self.take(r)
    }

    /// Returns delayed frames one at a time until None.
    pub fn flush(&mut self) -> Result<Option<Packet>, String> {
        let r = unsafe { sx_flush(self.raw) };
        self.take(r)
    }
}

impl Drop for X264 {
    fn drop(&mut self) {
        unsafe { sx_close(self.raw) }
    }
}

pub struct Opus {
    raw: *mut c_void,
    channels: usize,
    out: Vec<u8>,
}

unsafe impl Send for Opus {}

impl Opus {
    pub const SAMPLE_RATE: u32 = 48_000;
    /// 20 ms frames.
    pub const FRAME: usize = 960;

    pub fn open(channels: usize, bitrate: u32) -> Result<Self, String> {
        let mut err = 0;
        let raw = unsafe { so_open(Self::SAMPLE_RATE as c_int, channels as c_int, bitrate as c_int, &mut err) };
        if raw.is_null() {
            return Err(format!("opus_encoder_create failed ({err})"));
        }
        Ok(Opus { raw, channels, out: vec![0; 4000] })
    }

    /// Encoder delay in samples at 48 kHz (the MP4 "pre-skip").
    pub fn lookahead(&self) -> u32 {
        unsafe { so_lookahead(self.raw) as u32 }
    }

    /// `pcm` holds exactly FRAME interleaved samples per channel.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>, String> {
        assert_eq!(pcm.len(), Self::FRAME * self.channels);
        let n = unsafe {
            so_encode(self.raw, pcm.as_ptr(), Self::FRAME as c_int, self.out.as_mut_ptr(),
                      self.out.len() as c_int)
        };
        if n < 0 {
            return Err(format!("opus_encode_float failed ({n})"));
        }
        Ok(self.out[..n as usize].to_vec())
    }
}

impl Drop for Opus {
    fn drop(&mut self) {
        unsafe { so_close(self.raw) }
    }
}

#[cfg(feature = "av1")]
pub mod av1 {
    //! SVT-AV1 (needs real threads: native, or wasm32-wasip1-threads).

    use std::ffi::{CString, c_char, c_int, c_void};

    unsafe extern "C" {
        fn sv_open(w: c_int, h: c_int, fps_num: c_int, fps_den: c_int, opts: *const c_char, pass: c_int,
                   stats: *const u8, stats_len: usize) -> *mut c_void;
        fn sv_header(e: *mut c_void, len: *mut c_int) -> *const u8;
        fn sv_send(e: *mut c_void, y: *const u8, u: *const u8, v: *const u8, stride_y: c_int, stride_c: c_int,
                   pts: i64) -> c_int;
        fn sv_eos(e: *mut c_void) -> c_int;
        fn sv_receive(e: *mut c_void, block: c_int) -> c_int;
        fn sv_data(e: *mut c_void) -> *const u8;
        fn sv_pts(e: *mut c_void) -> i64;
        fn sv_keyframe(e: *mut c_void) -> c_int;
        fn sv_stats(e: *mut c_void, len: *mut usize) -> *const u8;
        fn sv_close(e: *mut c_void);
    }

    #[derive(Clone, Debug)]
    pub struct SvtConfig {
        pub width: u32,
        pub height: u32,
        pub fps_num: u32,
        pub fps_den: u32,
        /// SVT-AV1 options as "key=value:key=value" (its CLI names).
        pub opts: String,
        /// 0 single pass, 1 first pass, 2 second pass.
        pub pass: u8,
        /// First-pass statistics, for pass 2.
        pub stats: Vec<u8>,
    }

    /// One temporal unit, temporal delimiter removed (as MP4 wants).
    #[derive(Clone, Debug)]
    pub struct Packet {
        pub data: Vec<u8>,
        pub pts: i64,
        pub keyframe: bool,
    }

    pub struct Svt {
        raw: *mut c_void,
        out: Vec<Packet>,
    }

    unsafe impl Send for Svt {}

    impl Svt {
        pub fn open(cfg: &SvtConfig) -> Result<Self, String> {
            let opts = CString::new(cfg.opts.as_str()).unwrap();
            let raw = unsafe {
                sv_open(cfg.width as c_int, cfg.height as c_int, cfg.fps_num as c_int, cfg.fps_den as c_int,
                        opts.as_ptr(), cfg.pass as c_int, cfg.stats.as_ptr(), cfg.stats.len())
            };
            if raw.is_null() {
                return Err(format!("SVT-AV1 rejected config {:?}", SvtConfig { stats: vec![], ..cfg.clone() }));
            }
            Ok(Svt { raw, out: Vec::new() })
        }

        /// Sequence header OBU(s).
        pub fn header(&self) -> Vec<u8> {
            let mut n = 0;
            let p = unsafe { sv_header(self.raw, &mut n) };
            if p.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec() }
        }

        fn take(&mut self, n: c_int) {
            let data = unsafe { std::slice::from_raw_parts(sv_data(self.raw), n as usize) };
            self.out.push(Packet {
                data: strip_temporal_delimiters(data),
                pts: unsafe { sv_pts(self.raw) },
                keyframe: unsafe { sv_keyframe(self.raw) } != 0,
            });
        }

        /// Sends an I420 frame; any packets that are ready are collected.
        pub fn send(&mut self, y: &[u8], u: &[u8], v: &[u8], stride_y: usize, stride_c: usize, pts: i64)
                    -> Result<(), String> {
            let r = unsafe {
                sv_send(self.raw, y.as_ptr(), u.as_ptr(), v.as_ptr(), stride_y as c_int, stride_c as c_int, pts)
            };
            if r < 0 {
                return Err("SVT-AV1 send failed".into());
            }
            // Drain without blocking, or the encoder's output queue can fill
            // and stall its pipeline.
            loop {
                match unsafe { sv_receive(self.raw, 0) } {
                    n if n > 0 => self.take(n),
                    0 | -2 => return Ok(()),
                    _ => return Err("SVT-AV1 receive failed".into()),
                }
            }
        }

        /// Ends the stream and returns all packets in output order.
        pub fn finish(&mut self) -> Result<Vec<Packet>, String> {
            if unsafe { sv_eos(self.raw) } < 0 {
                return Err("SVT-AV1 eos failed".into());
            }
            loop {
                match unsafe { sv_receive(self.raw, 1) } {
                    n if n > 0 => self.take(n),
                    -2 => break,
                    0 => continue,
                    _ => return Err("SVT-AV1 receive failed".into()),
                }
            }
            Ok(std::mem::take(&mut self.out))
        }

        /// First-pass statistics (after `finish` on a pass-1 encoder).
        pub fn stats(&self) -> Vec<u8> {
            let mut n = 0usize;
            let p = unsafe { sv_stats(self.raw, &mut n) };
            if p.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(p, n) }.to_vec() }
        }
    }

    impl Drop for Svt {
        fn drop(&mut self) {
            unsafe { sv_close(self.raw) }
        }
    }

    fn leb128(d: &[u8]) -> Option<(usize, usize)> {
        let mut v = 0usize;
        for (i, &b) in d.iter().enumerate().take(8) {
            v |= ((b & 0x7f) as usize) << (7 * i);
            if b & 0x80 == 0 {
                return Some((v, i + 1));
            }
        }
        None
    }

    /// Drops temporal delimiter OBUs (type 2); MP4 samples must not carry them.
    pub fn strip_temporal_delimiters(d: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(d.len());
        let mut p = 0;
        while p < d.len() {
            let h = d[p];
            let typ = (h >> 3) & 0xf;
            let ext = (h >> 2) & 1;
            let has_size = (h >> 1) & 1;
            let hdr = 1 + ext as usize;
            if has_size == 0 {
                // Last OBU extends to the end of the unit.
                if typ != 2 {
                    out.extend_from_slice(&d[p..]);
                }
                break;
            }
            let Some((size, n)) = d.get(p + hdr..).and_then(leb128) else {
                out.extend_from_slice(&d[p..]);
                break;
            };
            let end = (p + hdr + n + size).min(d.len());
            if typ != 2 {
                out.extend_from_slice(&d[p..end]);
            }
            p = end;
        }
        out
    }
}
