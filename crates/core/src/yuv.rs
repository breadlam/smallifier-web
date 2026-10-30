//! 8-bit I420 frames, pixel-format conversion and separable resampling.
//!
//! Scaling is done here (not by the browser) so that the encoder input, and
//! therefore the output, is identical on every OS and browser, and so the
//! analysis can model the exact filter the encoder input went through.

#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Plane { w, h, data: vec![0; w * h] }
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..(y + 1) * self.w]
    }
}

/// Tightly packed I420 frame (stride == width).
#[derive(Clone, Debug)]
pub struct Frame {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
}

impl Frame {
    pub fn new(w: usize, h: usize) -> Self {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Frame { y: Plane::new(w, h), u: Plane::new(cw, ch), v: Plane::new(cw, ch) }
    }

    pub fn width(&self) -> usize {
        self.y.w
    }

    pub fn height(&self) -> usize {
        self.y.h
    }

    /// Packed I420 bytes (Y then U then V).
    pub fn to_i420(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.y.data.len() + 2 * self.u.data.len());
        out.extend_from_slice(&self.y.data);
        out.extend_from_slice(&self.u.data);
        out.extend_from_slice(&self.v.data);
        out
    }

    pub fn from_i420(w: usize, h: usize, buf: &[u8]) -> Self {
        let mut f = Frame::new(w, h);
        let ys = w * h;
        let cs = f.u.data.len();
        f.y.data.copy_from_slice(&buf[..ys]);
        f.u.data.copy_from_slice(&buf[ys..ys + cs]);
        f.v.data.copy_from_slice(&buf[ys + cs..ys + 2 * cs]);
        f
    }
}

/// Pixel layouts VideoFrame.copyTo() can hand us.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    I420,
    I420A,
    I422,
    I444,
    NV12,
    I420P10,
    I422P10,
    I444P10,
    I420P12,
    RGBA,
    RGBX,
    BGRA,
    BGRX,
}

impl PixelFormat {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "I420" => Self::I420,
            "I420A" => Self::I420A,
            "I422" => Self::I422,
            "I444" => Self::I444,
            "NV12" => Self::NV12,
            "I420P10" => Self::I420P10,
            "I422P10" => Self::I422P10,
            "I444P10" => Self::I444P10,
            "I420P12" => Self::I420P12,
            "RGBA" => Self::RGBA,
            "RGBX" => Self::RGBX,
            "BGRA" => Self::BGRA,
            "BGRX" => Self::BGRX,
            _ => return None,
        })
    }
}

/// One plane's position inside a copyTo() buffer.
#[derive(Clone, Copy, Debug)]
pub struct PlaneLayout {
    pub offset: usize,
    pub stride: usize,
}

/// Converts a decoded frame (any supported layout, visible rect already
/// applied by copyTo) into a packed 8-bit I420 frame at the same size.
pub fn convert(fmt: PixelFormat, w: usize, h: usize, buf: &[u8], layout: &[PlaneLayout]) -> Frame {
    let mut f = Frame::new(w, h);
    let (cw, ch) = (f.u.w, f.u.h);
    let rd8 = |p: usize, x: usize, y: usize| buf[layout[p].offset + y * layout[p].stride + x];
    let rd16 = |p: usize, x: usize, y: usize, shift: u32| {
        let i = layout[p].offset + y * layout[p].stride + 2 * x;
        let v = u16::from_le_bytes([buf[i], buf[i + 1]]) as u32;
        ((v + (1 << (shift - 1))) >> shift).min(255) as u8
    };
    match fmt {
        PixelFormat::I420 | PixelFormat::I420A => {
            copy_plane(&mut f.y, buf, layout[0]);
            copy_plane(&mut f.u, buf, layout[1]);
            copy_plane(&mut f.v, buf, layout[2]);
        }
        PixelFormat::NV12 => {
            copy_plane(&mut f.y, buf, layout[0]);
            for y in 0..ch {
                let row = layout[1].offset + y * layout[1].stride;
                for x in 0..cw {
                    f.u.data[y * cw + x] = buf[row + 2 * x];
                    f.v.data[y * cw + x] = buf[row + 2 * x + 1];
                }
            }
        }
        PixelFormat::I422 | PixelFormat::I444 => {
            copy_plane(&mut f.y, buf, layout[0]);
            let sx = if fmt == PixelFormat::I444 { 2 } else { 1 };
            for p in 1..3 {
                let dst = if p == 1 { &mut f.u } else { &mut f.v };
                let src_w = if sx == 2 { w } else { cw };
                for y in 0..ch {
                    let y1 = (2 * y + 1).min(h - 1);
                    for x in 0..cw {
                        let (x0, x1) = if sx == 2 { (2 * x, (2 * x + 1).min(src_w - 1)) } else { (x, x) };
                        let s = rd8(p, x0, 2 * y) as u32 + rd8(p, x1, 2 * y) as u32
                            + rd8(p, x0, y1) as u32 + rd8(p, x1, y1) as u32;
                        dst.data[y * cw + x] = ((s + 2) / 4) as u8;
                    }
                }
            }
        }
        PixelFormat::I420P10 | PixelFormat::I420P12 => {
            let shift = if fmt == PixelFormat::I420P10 { 2 } else { 4 };
            for y in 0..h {
                for x in 0..w {
                    f.y.data[y * w + x] = rd16(0, x, y, shift);
                }
            }
            for y in 0..ch {
                for x in 0..cw {
                    f.u.data[y * cw + x] = rd16(1, x, y, shift);
                    f.v.data[y * cw + x] = rd16(2, x, y, shift);
                }
            }
        }
        PixelFormat::I422P10 | PixelFormat::I444P10 => {
            for y in 0..h {
                for x in 0..w {
                    f.y.data[y * w + x] = rd16(0, x, y, 2);
                }
            }
            let full = fmt == PixelFormat::I444P10;
            for y in 0..ch {
                let y1 = (2 * y + 1).min(h - 1);
                for x in 0..cw {
                    let (x0, x1) = if full { (2 * x, (2 * x + 1).min(w - 1)) } else { (x, x) };
                    for p in 1..3 {
                        let s = rd16(p, x0, 2 * y, 2) as u32 + rd16(p, x1, 2 * y, 2) as u32
                            + rd16(p, x0, y1, 2) as u32 + rd16(p, x1, y1, 2) as u32;
                        let dst = if p == 1 { &mut f.u } else { &mut f.v };
                        dst.data[y * cw + x] = ((s + 2) / 4) as u8;
                    }
                }
            }
        }
        PixelFormat::RGBA | PixelFormat::RGBX | PixelFormat::BGRA | PixelFormat::BGRX => {
            let bgr = matches!(fmt, PixelFormat::BGRA | PixelFormat::BGRX);
            rgb_to_i420(&mut f, buf, layout[0], bgr);
        }
    }
    f
}

fn copy_plane(dst: &mut Plane, buf: &[u8], l: PlaneLayout) {
    for y in 0..dst.h {
        let s = l.offset + y * l.stride;
        dst.data[y * dst.w..(y + 1) * dst.w].copy_from_slice(&buf[s..s + dst.w]);
    }
}

/// BT.709 limited range, chroma averaged over each 2x2 block. Fixed point
/// (16-bit coefficients), two rows at a time: this runs on every decoded
/// frame in browsers that hand out RGB (Firefox on Linux does).
fn rgb_to_i420(f: &mut Frame, buf: &[u8], l: PlaneLayout, bgr: bool) {
    // Y  = 16 + (0.2126 R + 0.7152 G + 0.0722 B) * 219/255
    // Cb = 128 + (B - Y') / 1.8556 * 224/255,  Cr = 128 + (R - Y') / 1.5748 * 224/255
    const YR: i32 = 11966; // 0.2126 * 219/255 * 65536
    const YG: i32 = 40254;
    const YB: i32 = 4064;
    const UR: i32 = -6596; // (-0.2126 / 1.8556) * 224/255 * 65536
    const UG: i32 = -22189;
    const UB: i32 = 28784;
    const VR: i32 = 28784;
    const VG: i32 = -26145;
    const VB: i32 = -2639;
    let (w, h) = (f.y.w, f.y.h);
    let (ri, bi) = if bgr { (2, 0) } else { (0, 2) };
    let cw = f.u.w;
    for cy in 0..f.u.h {
        let y0 = 2 * cy;
        let y1 = (y0 + 1).min(h - 1);
        let r0 = &buf[l.offset + y0 * l.stride..];
        let r1 = &buf[l.offset + y1 * l.stride..];
        for cx in 0..cw {
            let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
            for (dy, row) in [(0usize, r0), (1, r1)] {
                for dx in 0..2 {
                    let x = (2 * cx + dx).min(w - 1);
                    let p = &row[4 * x..4 * x + 3];
                    let (r, g, b) = (p[ri] as i32, p[1] as i32, p[bi] as i32);
                    sr += r;
                    sg += g;
                    sb += b;
                    let yy = y0 + dy;
                    if yy < h && (dy == 0 || y1 != y0) && 2 * cx + dx < w {
                        let v = (16 << 16) + YR * r + YG * g + YB * b + (1 << 15);
                        f.y.data[yy * w + x] = (v >> 16).clamp(0, 255) as u8;
                    }
                }
            }
            // Averages of 4 samples: the /4 folds into the shift.
            let u = (128 << 18) + UR * sr + UG * sg + UB * sb + (1 << 17);
            let v = (128 << 18) + VR * sr + VG * sg + VB * sb + (1 << 17);
            f.u.data[cy * cw + cx] = (u >> 18).clamp(0, 255) as u8;
            f.v.data[cy * cw + cx] = (v >> 18).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------- scaling

const FIX: i32 = 14;

/// Precomputed 1-D filter taps for one src->dst length mapping.
#[derive(Clone, Debug)]
struct Taps {
    start: Vec<i32>,
    n: usize,
    w: Vec<i32>,
}

fn lanczos3(x: f64) -> f64 {
    let x = x.abs();
    if x < 1e-9 {
        1.0
    } else if x < 3.0 {
        let px = std::f64::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    } else {
        0.0
    }
}

impl Taps {
    fn new(src: usize, dst: usize) -> Taps {
        let scale = dst as f64 / src as f64;
        // Widen the kernel when shrinking so it also low-passes.
        let support = if scale < 1.0 { 3.0 / scale } else { 3.0 };
        let n = (2.0 * support).ceil() as usize + 1;
        let mut start = Vec::with_capacity(dst);
        let mut w = Vec::with_capacity(dst * n);
        let fscale = scale.min(1.0);
        for i in 0..dst {
            let center = (i as f64 + 0.5) / scale - 0.5;
            let s = (center - support).ceil() as i32;
            let mut ws: Vec<f64> = (0..n).map(|k| lanczos3((s as f64 + k as f64 - center) * fscale)).collect();
            let sum: f64 = ws.iter().sum();
            for v in &mut ws {
                *v /= sum;
            }
            // Fixed point; put rounding error on the largest tap so rows sum exactly to 1.
            let mut q: Vec<i32> = ws.iter().map(|v| (v * (1 << FIX) as f64).round() as i32).collect();
            let err = (1 << FIX) - q.iter().sum::<i32>();
            let imax = (0..n).max_by(|&a, &b| ws[a].partial_cmp(&ws[b]).unwrap()).unwrap();
            q[imax] += err;
            start.push(s);
            w.extend(q);
        }
        Taps { start, n, w }
    }
}

/// Lanczos-3 resampler for one fixed (src size -> dst size) mapping.
#[derive(Clone, Debug)]
pub struct Scaler {
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    hx: Taps,
    vy: Taps,
}

impl Scaler {
    pub fn new(sw: usize, sh: usize, dw: usize, dh: usize) -> Scaler {
        Scaler { sw, sh, dw, dh, hx: Taps::new(sw, dw), vy: Taps::new(sh, dh) }
    }

    pub fn scale_plane(&self, src: &Plane) -> Plane {
        debug_assert_eq!((src.w, src.h), (self.sw, self.sh));
        if (self.sw, self.sh) == (self.dw, self.dh) {
            return src.clone();
        }
        // Horizontal pass into i16 (values carry 14-bit precision headroom
        // after the >> (FIX - 6) shift: range ~[-64*255*0.3, 64*255*1.3]).
        let mut tmp = vec![0i16; self.dw * self.sh];
        let (n, sw) = (self.hx.n, self.sw as i32);
        for y in 0..self.sh {
            let row = src.row(y);
            let out = &mut tmp[y * self.dw..(y + 1) * self.dw];
            for (x, o) in out.iter_mut().enumerate() {
                let s = self.hx.start[x];
                let w = &self.hx.w[x * n..(x + 1) * n];
                let mut acc = 0i32;
                if s >= 0 && s + n as i32 <= sw {
                    let px = &row[s as usize..s as usize + n];
                    for k in 0..n {
                        acc += w[k] * px[k] as i32;
                    }
                } else {
                    for k in 0..n {
                        let xi = (s + k as i32).clamp(0, sw - 1) as usize;
                        acc += w[k] * row[xi] as i32;
                    }
                }
                *o = ((acc + (1 << (FIX - 7))) >> (FIX - 6)) as i16;
            }
        }
        // Vertical pass.
        let mut dst = Plane::new(self.dw, self.dh);
        let (n, sh) = (self.vy.n, self.sh as i32);
        let mut acc = vec![0i32; self.dw];
        for y in 0..self.dh {
            let s = self.vy.start[y];
            let w = &self.vy.w[y * n..(y + 1) * n];
            acc.iter_mut().for_each(|a| *a = 0);
            for k in 0..n {
                let yi = (s + k as i32).clamp(0, sh - 1) as usize;
                let r = &tmp[yi * self.dw..(yi + 1) * self.dw];
                let wk = w[k];
                for (a, &p) in acc.iter_mut().zip(r) {
                    *a += wk * p as i32;
                }
            }
            let out = &mut dst.data[y * self.dw..(y + 1) * self.dw];
            let round = 1 << (FIX + 6 - 1);
            for (o, &a) in out.iter_mut().zip(&acc) {
                *o = ((a + round) >> (FIX + 6)).clamp(0, 255) as u8;
            }
        }
        dst
    }
}

/// Scales a whole I420 frame. Chroma uses its own (half-size) mapping.
#[derive(Clone, Debug)]
pub struct FrameScaler {
    luma: Scaler,
    chroma: Scaler,
    pub dw: usize,
    pub dh: usize,
}

impl FrameScaler {
    pub fn new(sw: usize, sh: usize, dw: usize, dh: usize) -> Self {
        FrameScaler {
            luma: Scaler::new(sw, sh, dw, dh),
            chroma: Scaler::new(sw.div_ceil(2), sh.div_ceil(2), dw.div_ceil(2), dh.div_ceil(2)),
            dw,
            dh,
        }
    }

    pub fn scale(&self, f: &Frame) -> Frame {
        Frame { y: self.luma.scale_plane(&f.y), u: self.chroma.scale_plane(&f.u), v: self.chroma.scale_plane(&f.v) }
    }
}

/// Down to (rw, rh) and back up to the frame's own size: the per-scene
/// "prefilter rung" that removes detail the budget can't pay for while
/// keeping a constant output resolution.
#[derive(Clone, Debug)]
pub struct Prefilter {
    down: FrameScaler,
    up: FrameScaler,
}

impl Prefilter {
    pub fn new(w: usize, h: usize, rw: usize, rh: usize) -> Option<Self> {
        if (rw, rh) == (w, h) {
            return None;
        }
        Some(Prefilter { down: FrameScaler::new(w, h, rw, rh), up: FrameScaler::new(rw, rh, w, h) })
    }

    pub fn apply(&self, f: &Frame) -> Frame {
        self.up.scale(&self.down.scale(f))
    }
}

/// Gain of the down/up round trip (len -> rlen -> len) for a cosine at
/// `cycles_per_px`, measured by running the real fixed-point scaler on a
/// synthetic signal. Used by the analysis to model what a rung removes.
pub fn roundtrip_gain(len: usize, rlen: usize, cycles_per_px: f64) -> f64 {
    if rlen >= len {
        return 1.0;
    }
    let down = Scaler::new(len, 1, rlen, 1);
    let up = Scaler::new(rlen, 1, len, 1);
    let amp = 60.0;
    // Average over a few phases so the result doesn't depend on alignment.
    let mut g = 0.0;
    let phases = 4;
    for p in 0..phases {
        let ph = p as f64 * std::f64::consts::FRAC_PI_2 / phases as f64;
        let sig: Vec<f64> = (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * cycles_per_px * i as f64 + ph).cos())
            .collect();
        let src = Plane { w: len, h: 1, data: sig.iter().map(|v| (128.0 + v).round() as u8).collect() };
        let out = up.scale_plane(&down.scale_plane(&src));
        // Project onto the input cosine, ignoring edges.
        let (a, b) = (len / 8, len - len / 8);
        let (mut num, mut den) = (0.0, 0.0);
        for i in a..b {
            num += (out.data[i] as f64 - 128.0) * sig[i];
            den += sig[i] * sig[i];
        }
        g += (num / den).clamp(0.0, 1.0);
    }
    g / phases as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant() {
        let mut p = Plane::new(37, 21);
        p.data.iter_mut().for_each(|v| *v = 77);
        for (dw, dh) in [(37, 21), (20, 11), (80, 50)] {
            let s = Scaler::new(37, 21, dw, dh);
            let o = s.scale_plane(&p);
            assert!(o.data.iter().all(|&v| v == 77), "{dw}x{dh}");
        }
    }

    #[test]
    fn rgb_matches_reference() {
        // Fixed-point conversion against the BT.709 formulas in float.
        let (w, h) = (6usize, 4usize);
        let mut buf = vec![0u8; w * h * 4];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i * 37 + 11) % 256) as u8;
        }
        let f = convert(PixelFormat::RGBX, w, h, &buf, &[PlaneLayout { offset: 0, stride: w * 4 }]);
        for y in 0..h {
            for x in 0..w {
                let p = &buf[(y * w + x) * 4..];
                let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
                let yy = 16.0 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * 219.0 / 255.0;
                assert!((f.y.data[y * w + x] as f64 - yy).abs() <= 0.51, "Y at {x},{y}");
            }
        }
        let p = |x: usize, y: usize| {
            let q = &buf[(y * w + x) * 4..];
            (q[0] as f64, q[1] as f64, q[2] as f64)
        };
        let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (pr, pg, pb) = p(x, y);
            r += pr / 4.0;
            g += pg / 4.0;
            b += pb / 4.0;
        }
        let yl = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let u = 128.0 + (b - yl) / 1.8556 * 224.0 / 255.0;
        let v = 128.0 + (r - yl) / 1.5748 * 224.0 / 255.0;
        assert!((f.u.data[0] as f64 - u).abs() <= 0.51 && (f.v.data[0] as f64 - v).abs() <= 0.51);
    }

    #[test]
    fn gain_is_lowpass() {
        let lo = roundtrip_gain(1280, 640, 0.05);
        let hi = roundtrip_gain(1280, 640, 0.40);
        assert!(lo > 0.9, "{lo}");
        assert!(hi < 0.1, "{hi}");
        assert_eq!(roundtrip_gain(1280, 1280, 0.4), 1.0);
    }
}
