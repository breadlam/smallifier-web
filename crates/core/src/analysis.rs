//! Per-frame content analysis feeding the size/quality model.
//!
//! For a sampled frame we estimate what an H.264 encoder would have to code:
//! a coarse-to-fine block motion search against the previous output frame,
//! a per-macroblock inter/intra decision, and an 8x8 orthonormal DCT of the
//! chosen residual. Coefficient magnitudes go into log-spaced histograms per
//! coefficient position. Because the histograms keep the full magnitude
//! distribution per frequency, the model can later evaluate any quantizer
//! and any prefilter rung (a per-frequency gain) without touching pixels.

use crate::yuv::Plane;

/// Log-magnitude bins: bin 0 holds |c| < 0.5, bin i >= 1 holds
/// [2^(-1 + (i-1)/4), 2^(-1 + i/4)).
pub const NBINS: usize = 52;
pub const NPOS: usize = 64;
pub const HIST_LEN: usize = NPOS * NBINS;

#[inline]
pub fn bin_of(mag: f32) -> usize {
    if mag < 0.5 {
        return 0;
    }
    let b = ((mag.log2() + 1.0) * 4.0) as usize + 1;
    b.min(NBINS - 1)
}

/// Geometric centre of a bin.
#[inline]
pub fn bin_mag(i: usize) -> f64 {
    if i == 0 { 0.25 } else { 2f64.powf(-1.0 + (i as f64 - 0.5) / 4.0) }
}

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// Coefficient histogram of the residual an encoder would code for this
    /// frame as a P frame (per-MB best of inter and intra).
    pub inter: Vec<u32>,
    /// Same, coding the frame as an I frame.
    pub intra: Vec<u32>,
    /// Energy per coefficient position of the source itself (for the
    /// detail a prefilter rung removes).
    pub src_energy: Vec<f64>,
    /// Luma pixels analysed.
    pub pixels: u64,
    /// Macroblocks where intra beat inter.
    pub intra_mbs: u32,
    pub mbs: u32,
    /// Sum over MBs of log2(1 + |mv - predicted mv|), a motion-vector bit proxy.
    pub mv_cost: f64,
    /// Mean absolute luma motion-compensated residual.
    pub inter_sad: f64,
}

impl FrameStats {
    fn new() -> Self {
        FrameStats {
            inter: vec![0; HIST_LEN],
            intra: vec![0; HIST_LEN],
            src_energy: vec![0.0; NPOS],
            ..Default::default()
        }
    }

    /// Accumulate another frame's stats (for per-chunk totals).
    pub fn add(&mut self, o: &FrameStats) {
        if self.inter.is_empty() {
            *self = FrameStats::new();
        }
        for (a, b) in self.inter.iter_mut().zip(&o.inter) {
            *a += b;
        }
        for (a, b) in self.intra.iter_mut().zip(&o.intra) {
            *a += b;
        }
        for (a, b) in self.src_energy.iter_mut().zip(&o.src_energy) {
            *a += b;
        }
        self.pixels += o.pixels;
        self.intra_mbs += o.intra_mbs;
        self.mbs += o.mbs;
        self.mv_cost += o.mv_cost;
        self.inter_sad += o.inter_sad;
    }

    /// Compact little-endian serialisation for passing between workers.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HIST_LEN * 8 + NPOS * 8 + 40);
        for &v in self.inter.iter().chain(&self.intra) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.src_energy {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.pixels.to_le_bytes());
        out.extend_from_slice(&self.intra_mbs.to_le_bytes());
        out.extend_from_slice(&self.mbs.to_le_bytes());
        out.extend_from_slice(&self.mv_cost.to_le_bytes());
        out.extend_from_slice(&self.inter_sad.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != Self::BYTES {
            return None;
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let f64_at = |i: usize| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let mut s = FrameStats::new();
        for i in 0..HIST_LEN {
            s.inter[i] = u32_at(4 * i);
            s.intra[i] = u32_at(4 * (HIST_LEN + i));
        }
        let mut o = 8 * HIST_LEN;
        for i in 0..NPOS {
            s.src_energy[i] = f64_at(o);
            o += 8;
        }
        s.pixels = u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        s.intra_mbs = u32_at(o + 8);
        s.mbs = u32_at(o + 12);
        s.mv_cost = f64_at(o + 16);
        s.inter_sad = f64_at(o + 24);
        Some(s)
    }

    pub const BYTES: usize = HIST_LEN * 8 + NPOS * 8 + 8 + 4 + 4 + 8 + 8;
}

/// Orthonormal 8-point DCT-II basis, row k = frequency k.
fn dct_basis() -> [[f32; 8]; 8] {
    let mut m = [[0f32; 8]; 8];
    for (k, row) in m.iter_mut().enumerate() {
        let a = if k == 0 { (1.0f64 / 8.0).sqrt() } else { (2.0f64 / 8.0).sqrt() };
        for (n, v) in row.iter_mut().enumerate() {
            *v = (a * (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos()) as f32;
        }
    }
    m
}

#[inline]
fn dct8x8(m: &[[f32; 8]; 8], blk: &[f32; 64], out: &mut [f32; 64]) {
    let mut t = [0f32; 64];
    // rows: t[y][k] = sum_n blk[y][n] m[k][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0;
            for n in 0..8 {
                s += blk[y * 8 + n] * m[k][n];
            }
            t[y * 8 + k] = s;
        }
    }
    // cols: out[v][u] = sum_y t[y][u] m[v][y]
    for v in 0..8 {
        for u in 0..8 {
            let mut s = 0.0;
            for y in 0..8 {
                s += t[y * 8 + u] * m[v][y];
            }
            out[v * 8 + u] = s;
        }
    }
}

/// 2x box downsample.
fn half(p: &Plane) -> Plane {
    let (w, h) = (p.w / 2, p.h / 2);
    let mut o = Plane::new(w.max(1), h.max(1));
    for y in 0..h {
        let r0 = p.row(2 * y);
        let r1 = p.row(2 * y + 1);
        for x in 0..w {
            let s = r0[2 * x] as u32 + r0[2 * x + 1] as u32 + r1[2 * x] as u32 + r1[2 * x + 1] as u32;
            o.data[y * w + x] = ((s + 2) / 4) as u8;
        }
    }
    o
}

/// SAD of a bs x bs block at (x, y) in `cur` against (x+dx, y+dy) in `refp`,
/// with the reference clamped to the plane (as encoders pad edges).
#[inline]
fn sad(cur: &Plane, refp: &Plane, x: i32, y: i32, dx: i32, dy: i32, bs: i32) -> u32 {
    let (rx, ry) = (x + dx, y + dy);
    let inside = rx >= 0 && ry >= 0 && rx + bs <= refp.w as i32 && ry + bs <= refp.h as i32;
    let mut s = 0u32;
    if inside {
        for j in 0..bs {
            let c = &cur.data[((y + j) as usize) * cur.w + x as usize..][..bs as usize];
            let r = &refp.data[((ry + j) as usize) * refp.w + rx as usize..][..bs as usize];
            for i in 0..bs as usize {
                s += (c[i] as i32 - r[i] as i32).unsigned_abs();
            }
        }
    } else {
        for j in 0..bs {
            for i in 0..bs {
                let c = cur.data[((y + j) as usize) * cur.w + (x + i) as usize] as i32;
                let rxx = (rx + i).clamp(0, refp.w as i32 - 1) as usize;
                let ryy = (ry + j).clamp(0, refp.h as i32 - 1) as usize;
                s += (c - refp.data[ryy * refp.w + rxx] as i32).unsigned_abs();
            }
        }
    }
    s
}

/// Exhaustive +-range search around `centre`. Cost is SAD plus `lam` per
/// unit of distance from `anchor` (an MV-cost term, as in real encoders,
/// which keeps flat or 1-D textures from picking arbitrary vectors).
fn search(cur: &Plane, refp: &Plane, x: i32, y: i32, bs: i32, centre: (i32, i32), range: i32,
          anchor: (i32, i32), lam: u32) -> ((i32, i32), u32) {
    let cost = |mv: (i32, i32)| {
        sad(cur, refp, x, y, mv.0, mv.1, bs)
            + lam * ((mv.0 - anchor.0).unsigned_abs() + (mv.1 - anchor.1).unsigned_abs())
    };
    let mut best = (centre, cost(centre));
    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue;
            }
            let mv = (centre.0 + dx, centre.1 + dy);
            let c = cost(mv);
            if c < best.1 {
                best = (mv, c);
            }
        }
    }
    best
}

pub struct Analyzer {
    w: usize,
    h: usize,
    basis: [[f32; 8]; 8],
}

impl Analyzer {
    /// Frame size must be a multiple of 16 in each dimension for full
    /// coverage; any remainder strip is ignored.
    pub fn new(w: usize, h: usize) -> Self {
        Analyzer { w, h, basis: dct_basis() }
    }

    /// Analyse `cur` (luma) as an encoder would see it, predicting from
    /// `prev` (the previous output frame's luma), or intra-only without one.
    pub fn analyze(&self, prev: Option<&Plane>, cur: &Plane) -> FrameStats {
        assert_eq!((cur.w, cur.h), (self.w, self.h));
        let mut st = FrameStats::new();
        let mbw = self.w / 16;
        let mbh = self.h / 16;
        st.mbs = (mbw * mbh) as u32;
        st.pixels = (mbw * mbh * 256) as u64;

        // Motion field (full-res pixels), coarse to fine.
        let mut mvs = vec![(0i32, 0i32); mbw * mbh];
        if let Some(prev) = prev {
            let (c2, p2) = (half(cur), half(prev));
            let (c4, p4) = (half(&c2), half(&p2));
            let (c8, p8) = (half(&c4), half(&p4));
            for my in 0..mbh {
                for mx in 0..mbw {
                    // 1/8 res: 2x2 block per MB is too small to be reliable, so
                    // search with the 4x4 block covering this MB's 2x2 neighbourhood.
                    let bx = (mx as i32 * 2 - 1).clamp(0, c8.w as i32 - 4);
                    let by = (my as i32 * 2 - 1).clamp(0, c8.h as i32 - 4);
                    let mut cands = vec![(0, 0)];
                    if mx > 0 {
                        cands.push(mvs[my * mbw + mx - 1]);
                    }
                    if my > 0 {
                        cands.push(mvs[(my - 1) * mbw + mx]);
                        if mx + 1 < mbw {
                            cands.push(mvs[(my - 1) * mbw + mx + 1]);
                        }
                    }
                    let mut best = search(&c8, &p8, bx, by, 4, (0, 0), 6, (0, 0), 4);
                    for &(px, py) in &cands[1..] {
                        let c = ((px + 4) >> 3, (py + 4) >> 3);
                        let r = search(&c8, &p8, bx, by, 4, c, 1, (0, 0), 4);
                        if r.1 < best.1 {
                            best = r;
                        }
                    }
                    // Refine at 1/4 and 1/2 res on the MB itself.
                    let mut mv = (best.0.0 * 2, best.0.1 * 2);
                    let (x4, y4) = (mx as i32 * 4, my as i32 * 4);
                    mv = search(&c4, &p4, x4, y4, 4, mv, 1, mv, 1).0;
                    mv = (mv.0 * 2, mv.1 * 2);
                    let (x2, y2) = (mx as i32 * 8, my as i32 * 8);
                    mv = search(&c2, &p2, x2, y2, 8, mv, 1, mv, 1).0;
                    mv = (mv.0 * 2, mv.1 * 2);
                    // Full res: the refined vector competes with the neighbours'
                    // vectors and zero, then gets one +-1 step.
                    let (x1, y1) = (mx as i32 * 16, my as i32 * 16);
                    let pred = cands.get(1).copied().unwrap_or((0, 0));
                    let mut full = search(cur, prev, x1, y1, 16, mv, 0, pred, 4);
                    for &c in &cands {
                        let r = search(cur, prev, x1, y1, 16, c, 0, pred, 4);
                        if r.1 < full.1 {
                            full = r;
                        }
                    }
                    mv = search(cur, prev, x1, y1, 16, full.0, 1, pred, 4).0;
                    mvs[my * mbw + mx] = mv;
                }
            }
        }

        let mut blk = [0f32; 64];
        let mut res = [0f32; 64];
        let mut coef = [0f32; 64];
        let mut sad_total = 0u64;
        for my in 0..mbh {
            for mx in 0..mbw {
                let (x0, y0) = (mx * 16, my * 16);
                let mv = mvs[my * mbw + mx];
                if mx > 0 || my > 0 {
                    let pred = if mx > 0 { mvs[my * mbw + mx - 1] } else { mvs[(my - 1) * mbw + mx] };
                    let d = (mv.0 - pred.0).abs() + (mv.1 - pred.1).abs();
                    st.mv_cost += (1.0 + d as f64).log2();
                }
                // Per-MB mode decision on SAD: intra residual vs inter residual.
                let mut intra_sad = 0u32;
                let mut inter_sad = 0u32;
                let mut intra_res = [[0f32; 64]; 4];
                let mut inter_res = [[0f32; 64]; 4];
                let mut src_coef = [[0f32; 64]; 4];
                for b in 0..4 {
                    let (bx, by) = (x0 + (b & 1) * 8, y0 + (b >> 1) * 8);
                    for j in 0..8 {
                        let row = cur.row(by + j);
                        for i in 0..8 {
                            blk[j * 8 + i] = row[bx + i] as f32;
                        }
                    }
                    dct8x8(&self.basis, &blk, &mut src_coef[b]);
                    // Intra: best of DC / horizontal / vertical prediction from
                    // the source neighbours (stand-in for reconstructed ones).
                    let top: Option<[f32; 8]> = (by > 0).then(|| {
                        let r = cur.row(by - 1);
                        std::array::from_fn(|i| r[bx + i] as f32)
                    });
                    let left: Option<[f32; 8]> =
                        (bx > 0).then(|| std::array::from_fn(|j| cur.row(by + j)[bx - 1] as f32));
                    let dc = match (top, left) {
                        (Some(t), Some(l)) => (t.iter().sum::<f32>() + l.iter().sum::<f32>()) / 16.0,
                        (Some(t), None) => t.iter().sum::<f32>() / 8.0,
                        (None, Some(l)) => l.iter().sum::<f32>() / 8.0,
                        (None, None) => 128.0,
                    };
                    let mut best_s = f32::MAX;
                    for mode in 0..3 {
                        let p = |i: usize, j: usize| match mode {
                            0 => dc,
                            1 => match top { Some(t) => t[i], None => dc },
                            _ => match left { Some(l) => l[j], None => dc },
                        };
                        if (mode == 1 && top.is_none()) || (mode == 2 && left.is_none()) {
                            continue;
                        }
                        let mut s = 0.0;
                        for j in 0..8 {
                            for i in 0..8 {
                                let r = blk[j * 8 + i] - p(i, j);
                                res[j * 8 + i] = r;
                                s += r.abs();
                            }
                        }
                        if s < best_s {
                            best_s = s;
                            intra_res[b] = res;
                        }
                    }
                    intra_sad += best_s as u32;
                    if let Some(prev) = prev {
                        for j in 0..8 {
                            for i in 0..8 {
                                let rx = (bx as i32 + i as i32 + mv.0).clamp(0, self.w as i32 - 1) as usize;
                                let ry = (by as i32 + j as i32 + mv.1).clamp(0, self.h as i32 - 1) as usize;
                                let r = blk[j * 8 + i] - prev.data[ry * self.w + rx] as f32;
                                inter_res[b][j * 8 + i] = r;
                                inter_sad += r.abs() as u32;
                            }
                        }
                    }
                }
                sad_total += inter_sad.min(intra_sad) as u64;
                // Intra costs more side info; bias like an encoder's lambda would.
                let use_intra = prev.is_none() || (intra_sad as f32) < 0.85 * inter_sad as f32;
                if use_intra && prev.is_some() {
                    st.intra_mbs += 1;
                }
                for b in 0..4 {
                    for (k, &c) in src_coef[b].iter().enumerate() {
                        st.src_energy[k] += (c * c) as f64;
                    }
                    dct8x8(&self.basis, &intra_res[b], &mut coef);
                    for (k, &c) in coef.iter().enumerate() {
                        st.intra[k * NBINS + bin_of(c.abs())] += 1;
                    }
                    if !use_intra {
                        dct8x8(&self.basis, &inter_res[b], &mut coef);
                    }
                    for (k, &c) in coef.iter().enumerate() {
                        st.inter[k * NBINS + bin_of(c.abs())] += 1;
                    }
                }
            }
        }
        st.inter_sad = sad_total as f64 / st.pixels as f64;
        st
    }
}

/// Small luma+chroma thumbnail used for scene-cut detection.
#[derive(Clone, Debug)]
pub struct Thumb(pub Vec<u8>);

pub const THUMB_W: usize = 64;
pub const THUMB_H: usize = 36;

impl Thumb {
    pub fn of(f: &crate::yuv::Frame) -> Thumb {
        let mut out = Vec::with_capacity(THUMB_W * THUMB_H * 3 / 2);
        for (p, tw, th) in [(&f.y, THUMB_W, THUMB_H), (&f.u, THUMB_W / 2, THUMB_H / 2), (&f.v, THUMB_W / 2, THUMB_H / 2)] {
            for ty in 0..th {
                let (y0, y1) = (ty * p.h / th, ((ty + 1) * p.h / th).max(ty * p.h / th + 1));
                for tx in 0..tw {
                    let (x0, x1) = (tx * p.w / tw, ((tx + 1) * p.w / tw).max(tx * p.w / tw + 1));
                    let mut s = 0u32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            s += p.data[y * p.w + x] as u32;
                        }
                    }
                    out.push((s / ((y1 - y0) * (x1 - x0)) as u32) as u8);
                }
            }
        }
        Thumb(out)
    }

    /// Mean absolute difference, luma weighted 2:1 over each chroma plane
    /// (roughly the scale of PySceneDetect's content_val).
    pub fn diff(&self, o: &Thumb) -> f32 {
        let ny = THUMB_W * THUMB_H;
        let mut dy = 0u32;
        let mut dc = 0u32;
        for (i, (a, b)) in self.0.iter().zip(&o.0).enumerate() {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            if i < ny { dy += d } else { dc += d }
        }
        let nc = self.0.len() - ny;
        (2.0 * dy as f32 / ny as f32 + dc as f32 / nc as f32) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-periodic texture: bilinear value noise on an 8 px grid plus grain.
    fn noise_plane(w: usize, h: usize, seed: u32) -> Plane {
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as f32
        };
        let (gw, gh) = (w / 8 + 2, h / 8 + 2);
        let grid: Vec<f32> = (0..gw * gh).map(|_| rnd()).collect();
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (gx, gy) = (x / 8, y / 8);
                let (fx, fy) = ((x % 8) as f32 / 8.0, (y % 8) as f32 / 8.0);
                let g = |i: usize, j: usize| grid[(gy + j) * gw + gx + i];
                let v = g(0, 0) * (1.0 - fx) * (1.0 - fy) + g(1, 0) * fx * (1.0 - fy)
                    + g(0, 1) * (1.0 - fx) * fy + g(1, 1) * fx * fy;
                p.data[y * w + x] = (v * 0.8 + rnd() * 0.1) as u8;
            }
        }
        p
    }

    #[test]
    fn roundtrip_bytes() {
        let a = Analyzer::new(64, 48);
        let p = noise_plane(64, 48, 1);
        let s = a.analyze(None, &p);
        let b = FrameStats::from_bytes(&s.to_bytes()).unwrap();
        assert_eq!(s.inter, b.inter);
        assert_eq!(s.mbs, b.mbs);
    }

    #[test]
    fn finds_shift() {
        // A shifted copy should be predicted almost perfectly.
        let w = 128;
        let h = 64;
        let base = noise_plane(w + 8, h, 3);
        let mut prev = Plane::new(w, h);
        let mut cur = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                prev.data[y * w + x] = base.data[y * (w + 8) + x + 5];
                cur.data[y * w + x] = base.data[y * (w + 8) + x];
            }
        }
        let a = Analyzer::new(w, h);
        let s = a.analyze(Some(&prev), &cur);
        assert!(s.inter_sad < 3.0, "sad {}", s.inter_sad);
    }
}
