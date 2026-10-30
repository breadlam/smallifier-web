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
