//! Size and quality prediction from analysis histograms.
//!
//! Fitted by tools/fit_model.py against real encodes made with the same
//! x264 build and settings the app uses (see crates/lab). Rate is a
//! rho-domain style model: bits are linear in the number of coefficients
//! that survive quantization and in their log-levels, evaluated at the
//! H.264 quantizer step implied by the CRF, after the prefilter rung's
//! per-frequency gain.

use crate::analysis::{FrameStats, HIST_LEN, NBINS, NPOS, bin_mag};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RateParams {
    pub a_i: f64,
    pub b_i: f64,
    pub c_i: f64,
    pub a_p: f64,
    pub b_p: f64,
    pub c_p: f64,
    pub d_i: f64,
    pub d_p: f64,
    pub tau: f64,
    pub tau_hf: f64,
    pub mv_p: f64,
}

/// Quantization distortion: exp(ln_a) * Qstep(crf)^b * V^g, with V the
/// residual coefficient energy per pixel after the rung's gain.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DistParams {
    pub ln_a: f64,
    pub b: f64,
    pub g: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Model {
    pub rate: RateParams,
    pub dist: DistParams,
    /// CRF units per doubling of the quantizer step (6 for H.264's QP).
    pub q_double: f64,
    /// CRF range the planner searches.
    pub crf_min: f64,
    pub crf_max: f64,
    /// Two-pass size bias: actual/requested ~ base + per_frame / frames.
    pub bias_base: f64,
    pub bias_per_frame: f64,
    /// Relative size error of one two-pass chunk (sd).
    pub chunk_sigma: f64,
}

impl Model {
    /// Quantizer step (in H.264 QP terms) at a CRF.
    #[inline]
    pub fn qstep(&self, q: f64) -> f64 {
        0.625 * 2f64.powf(q / self.q_double)
    }
}

/// Default x264 preset. The model is fitted per preset; the encoder must
/// use exactly `x264_opts(preset)` for the predictions to hold.
pub const X264_PRESET: &str = "medium";

/// Options applied after the preset (rate control is added separately).
/// rc-lookahead is capped at 20: each worker holds its own encoder, and
/// medium's default of 40 costs ~400 MB per encoder at 1080p.
pub fn x264_opts(preset: &str) -> String {
    let base = "keyint=300:min-keyint=30:aq-mode=3";
    match preset {
        "ultrafast" | "superfast" | "veryfast" | "faster" => base.to_string(),
        _ => format!("{base}:rc-lookahead=20"),
    }
}

// FITTED by tools/fit_model.py on the lab data (4 game clips x 360p..1080p,
// 30/60 fps, 8 CRFs x 4 rungs, true PSNR from decoded output); see MODEL.md.
pub const MEDIUM: Model = Model {
    rate: RateParams {
        a_i: 8.527947972584839, b_i: 5.278589195496304, c_i: 0.24358782558543007,
        a_p: 3.9171040865360456, b_p: 0.047090979339267654, c_p: 0.7963571021775048,
        d_i: -4.170370166263235, d_p: 1.933984216401496, tau: 0.687901350586257, tau_hf: 1.9389995416382755, mv_p: 3.9642794429599797,
    },
    dist: DistParams { ln_a: -3.6052932762756376, b: 1.2271408234489998, g: 0.8583397234367744 },
    q_double: 6.0,
    crf_min: 12.0,
    crf_max: 51.0,
    // x264's second pass overshoots short chunks (lab and browser runs).
    bias_base: 1.03,
    bias_per_frame: 1.5,
    chunk_sigma: 0.03,
};

pub const VERYFAST: Model = Model {
    rate: RateParams {
        a_i: 7.393969865707169, b_i: 4.400717455386344, c_i: 0.4093293004668074,
        a_p: 3.6618344143271266, b_p: 0.3717157969746513, c_p: 0.5912442185287187,
        d_i: -3.5310281261410803, d_p: 1.9320865810476782, tau: 0.723612594712486, tau_hf: 1.9443700819162792, mv_p: 2.9655491934869476,
    },
    dist: DistParams { ln_a: -2.9743123990412705, b: 1.2455486229916988, g: 0.7437714572286993 },
    q_double: 6.0,
    crf_min: 12.0,
    crf_max: 51.0,
    bias_base: 1.03,
    bias_per_frame: 1.5,
    chunk_sigma: 0.03,
};

/// x264 slower and veryslow, fitted on the same lab design as medium. Their
/// second passes overshoot short chunks like medium's, but ~3% less overall
/// (browser runs), hence bias_base 1.0.
pub const SLOWER: Model = Model {
    rate: RateParams {
        a_i: 5.1977192788568125, b_i: 3.661438452072924, c_i: 0.5107445601572048,
        a_p: 3.6965730798542213, b_p: 0.0018737200472413148, c_p: 0.9301822929682398,
        d_i: -3.189346337839274, d_p: 2.0753939283596115, tau: 0.6939812830822337, tau_hf: 1.3558463296914411, mv_p: 3.4479934569437103,
    },
    dist: DistParams { ln_a: -3.7082046313757595, b: 1.2286535645789538, g: 0.8775743273161757 },
    q_double: 6.0,
    crf_min: 12.0,
    crf_max: 51.0,
    bias_base: 1.0,
    bias_per_frame: 1.5,
    chunk_sigma: 0.03,
};

pub const VERYSLOW: Model = Model {
    rate: RateParams {
        a_i: 10.442815004923478, b_i: 4.264528516363398, c_i: 4.600494488921234,
        a_p: 3.9878065049373523, b_p: 0.04623854872756038, c_p: 0.7989782947383715,
        d_i: -4.5469630402224315, d_p: 1.9423834391685826, tau: 0.6970535202805725, tau_hf: 2.401434532615181, mv_p: 2.991036483833553,
    },
    dist: DistParams { ln_a: -3.839628391748531, b: 1.2182280326075368, g: 0.9239134169374331 },
    q_double: 6.0,
    crf_min: 12.0,
    crf_max: 51.0,
    bias_base: 1.0,
    bias_per_frame: 1.5,
    chunk_sigma: 0.03,
};

/// SVT-AV1 (preset, bit depth) of an AV1 preset name: "svtN" is preset N
/// at 8 bits, "svtN-10" at 10 bits.
pub fn svt_preset(preset: &str) -> Option<(u32, u32)> {
    let rest = preset.strip_prefix("svt")?;
    let (p, depth) = match rest.split_once('-') {
        Some((p, d)) => (p, d.parse().ok()?),
        None => (rest, 8),
    };
    Some((p.parse().ok()?, depth))
}

/// SVT-AV1 options (its CLI names; rate control added by the caller as
/// `rc`). smallifier.py's tuning (tune=0, no temporal filtering, quant
/// matrices, aq-mode 2) minus overlays, which multi-pass doesn't allow;
/// closed GOPs so chunks are independent; lp=1, which makes SVT-AV1 run its
/// kernels cooperatively on one thread (the wasm build has no threads).
pub fn svt_opts(preset: u32, depth: u32, rc: &str) -> String {
    format!(
        "preset={preset}:input-depth={depth}:{rc}:keyint=300:irefresh-type=2:tune=0:enable-tf=0:enable-qm=1:\
         qm-min=0:qm-max=15:aq-mode=2:scd=1:enable-overlays=0:film-grain=0:lp=1"
    )
}

/// SVT-AV1 preset 8 (CRF 1..63, two-pass VBR with gop-constraint-rc:
/// chunks land within ~1% of target with no systematic bias).
pub const SVT8: Model = Model {
    rate: RateParams {
        a_i: 6.099890593807541, b_i: 3.7461101041289147, c_i: 0.676754387488648,
        a_p: 2.372335319717647, b_p: 0.08268569304900862, c_p: 1.135203224275056,
        d_i: -3.2376343437937116, d_p: 2.368175503266811, tau: 0.8379292927225688, tau_hf: 1.234389512193184, mv_p: 0.5952344625489631,
    },
    dist: DistParams { ln_a: -3.0247568848677346, b: 0.8369546436111518, g: 1.0378966140527712 },
    q_double: 10.597564844496198,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// SVT-AV1 preset 6: fitted on a reduced lab design (5 clip/size sets,
/// rungs 0.6 and 1.0, 5 CRFs); two-pass lands within ~1% like preset 8.
pub const SVT6: Model = Model {
    rate: RateParams {
        a_i: 6.901279650411129, b_i: 4.750835387084431, c_i: 0.2648118333932189,
        a_p: 2.562145602023137, b_p: 0.22848346272740871, c_p: 0.522426452319615,
        d_i: -6.5534244006583195, d_p: 3.2981708487453947, tau: 0.8942312601075101, tau_hf: 1.9636820580015808, mv_p: 0.5542621211146328,
    },
    dist: DistParams { ln_a: -3.5243489012835854, b: 0.9582802480158579, g: 1.0595992741347677 },
    q_double: 10.948074497981313,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// SVT-AV1 preset 5: fitted on a reduced lab design (5 clip/size sets,
/// rungs 0.6 and 1.0, 5 CRFs); two-pass lands within ~1% like preset 8.
pub const SVT5: Model = Model {
    rate: RateParams {
        a_i: 10.72132607497544, b_i: 6.223601958841172, c_i: 2.06304032522098,
        a_p: 2.8037179365937726, b_p: 0.02959635246831241, c_p: 1.0361385942975847,
        d_i: -4.419828366448159, d_p: 3.8546146664602423, tau: 0.8564374146911404, tau_hf: 2.0596423463803153, mv_p: 1.0436675525739318,
    },
    dist: DistParams { ln_a: -3.8696168155277273, b: 0.8895715111032971, g: 1.1431596884120838 },
    q_double: 10.405259193038736,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// SVT-AV1 preset 4: fitted on a reduced lab design (5 clip/size sets,
/// rungs 0.6 and 1.0, 5 CRFs); two-pass lands within ~1% like preset 8.
pub const SVT4: Model = Model {
    rate: RateParams {
        a_i: 7.447758375601836, b_i: 3.541635035590568, c_i: 0.43845037632641715,
        a_p: 2.3859465400444737, b_p: 0.09976932010405787, c_p: 0.9503342829929615,
        d_i: -5.2516092560915615, d_p: 4.159763905574993, tau: 0.8473000198279849, tau_hf: 1.7467205391454363, mv_p: 0.6497026462015939,
    },
    dist: DistParams { ln_a: -3.9360154323577587, b: 0.9171484326108035, g: 1.1575089598924413 },
    q_double: 10.80599948925056,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// SVT-AV1 preset 3: fitted on a reduced lab design (5 clip/size sets,
/// rungs 0.6 and 1.0, 5 CRFs); two-pass lands within ~1% like preset 8.
pub const SVT3: Model = Model {
    rate: RateParams {
        a_i: 7.447224066029272, b_i: 3.541272495990696, c_i: 0.43843915057661054,
        a_p: 2.372722622844683, b_p: 0.09917251122189369, c_p: 0.9479274880352149,
        d_i: -5.251250626459328, d_p: 4.160319786875874, tau: 0.8481970143909792, tau_hf: 1.747309397187073, mv_p: 0.6493753370835833,
    },
    dist: DistParams { ln_a: -4.108094729801061, b: 0.9395554943115955, g: 1.1715194066491035 },
    q_double: 10.80412692757622,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// SVT-AV1 preset 2: fitted on a reduced lab design (5 clip/size sets,
/// rungs 0.6 and 1.0, 5 CRFs); two-pass lands within ~1% like preset 8.
pub const SVT2: Model = Model {
    rate: RateParams {
        a_i: 5.815678122300166, b_i: 4.809428088730607, c_i: 0.1462127167871439,
        a_p: 2.5436428483595406, b_p: 0.29271644672998953, c_p: 0.5425628631882208,
        d_i: -10.550362573706025, d_p: 4.527123242194438, tau: 0.9101037123178956, tau_hf: 2.163526600378827, mv_p: 0.29464653519132816,
    },
    dist: DistParams { ln_a: -4.508723401538453, b: 1.0478303182135127, g: 1.1979096671661849 },
    q_double: 11.08642675264924,
    crf_min: 10.0,
    crf_max: 63.0,
    bias_base: 1.0,
    bias_per_frame: 0.0,
    chunk_sigma: 0.02,
};

/// The fitted model of a preset (each preset needs its own fit).
pub fn model_for(preset: &str) -> Option<&'static Model> {
    Some(match preset {
        "veryfast" => &VERYFAST,
        "medium" => &MEDIUM,
        "slower" => &SLOWER,
        "veryslow" => &VERYSLOW,
        "svt8" => &SVT8,
        "svt6" => &SVT6,
        "svt5" => &SVT5,
        "svt4" => &SVT4,
        "svt3" => &SVT3,
        "svt2" => &SVT2,
        _ => return None,
    })
}

/// Analysis aggregated over one chunk: its first frame coded as I, the
/// sampled remaining frames as P.
#[derive(Clone, Debug, Default)]
pub struct ChunkStats {
    pub i_hist: Vec<u32>,
    pub p_hist: Vec<u32>,
    /// Frames that went into p_hist.
    pub p_sampled: u32,
    /// Frames in the chunk.
    pub frames: u32,
    pub src_energy: Vec<f64>,
    /// Frames that went into src_energy.
    pub src_frames: u32,
    pub mbs_per_frame: f64,
    pub pixels_per_frame: f64,
    /// MV cost summed over the sampled P frames.
    pub mv_cost: f64,
}

impl ChunkStats {
    /// `first` is the analysis of the chunk's first frame (its intra
    /// histogram is used); `rest` are sampled frames after it.
    pub fn build(frames: u32, first: &FrameStats, rest: &[&FrameStats]) -> ChunkStats {
        let mut s = ChunkStats {
            i_hist: first.intra.clone(),
            p_hist: vec![0; HIST_LEN],
            p_sampled: 0,
            frames,
            src_energy: first.src_energy.clone(),
            src_frames: 1,
            mbs_per_frame: first.mbs as f64,
            pixels_per_frame: first.pixels as f64,
            mv_cost: 0.0,
        };
        for r in rest {
            for (a, b) in s.p_hist.iter_mut().zip(&r.inter) {
                *a += b;
            }
            for (a, b) in s.src_energy.iter_mut().zip(&r.src_energy) {
                *a += b;
            }
            s.src_frames += 1;
            s.p_sampled += 1;
            s.mv_cost += r.mv_cost;
        }
        if rest.is_empty() {
            // Single-frame chunk or nothing sampled: treat the I frame's
            // intra residual as a pessimistic P estimate.
            s.p_hist = first.intra.clone();
            s.p_sampled = 1;
        }
        s
    }

    fn p_frames(&self) -> f64 {
        (self.frames.saturating_sub(1)) as f64
    }
}

/// Per-coefficient-position gain of a prefilter rung (1.0 = untouched),
/// plus a correction for the spectral estimate of the detail it removes.
#[derive(Clone, Debug)]
pub struct Gain(pub [f64; NPOS], pub f64);

/// The spectral estimate of prefilter loss is exact for a 0.5x rung (its
/// cut-off sits on a DCT basis frequency) but low by ~16-19% for 0.6x and
/// 0.75x (measured against the real round trip in the lab).
fn scale_fix(factor: f64) -> f64 {
    const PTS: [(f64, f64); 4] = [(0.5, 1.0), (0.6, 1.16), (0.75, 1.19), (1.0, 1.0)];
    if factor <= PTS[0].0 {
        return PTS[0].1;
    }
    for w in PTS.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if factor <= x1 {
            return y0 + (y1 - y0) * (factor - x0) / (x1 - x0);
        }
    }
    1.0
}

impl Gain {
    pub fn identity() -> Gain {
        Gain([1.0; NPOS], 1.0)
    }

    /// Separable gain from the measured 1-D round-trip responses of the
    /// rung's scaler for each DCT basis frequency (u / 16 cycles per pixel).
    pub fn for_rung(w: usize, h: usize, rw: usize, rh: usize) -> Gain {
        let gx: Vec<f64> = (0..8).map(|u| if u == 0 { 1.0 } else { crate::yuv::roundtrip_gain(w, rw, u as f64 / 16.0) }).collect();
        let gy: Vec<f64> = (0..8).map(|v| if v == 0 { 1.0 } else { crate::yuv::roundtrip_gain(h, rh, v as f64 / 16.0) }).collect();
        let mut g = [0.0; NPOS];
        for v in 0..8 {
            for u in 0..8 {
                g[v * 8 + u] = gx[u] * gy[v];
            }
        }
        Gain(g, scale_fix(rw as f64 / w as f64))
    }
}

fn freq(k: usize) -> f64 {
    ((k % 8) + (k / 8)) as f64 / 14.0
}

/// (nonzero count, sum of log2 levels) for a histogram at quantizer q.
fn features(m: &Model, hist: &[u32], gain: &Gain, q: f64) -> (f64, f64) {
    let p = &m.rate;
    let inv_step = 1.0 / m.qstep(q);
    let (mut nz, mut lv) = (0.0, 0.0);
    for k in 0..NPOS {
        let t = p.tau * (1.0 + p.tau_hf * freq(k));
        let g = gain.0[k] * inv_step;
        let row = &hist[k * NBINS..(k + 1) * NBINS];
        for (i, &n) in row.iter().enumerate().rev() {
            if n == 0 {
                continue;
            }
            let x = bin_mag(i) * g;
            if x < t {
                break; // bins are sorted by magnitude
            }
            nz += n as f64;
            lv += n as f64 * (x / t).log2();
        }
    }
    (nz, lv)
}

/// The terms of `predict_bytes` in bits (I, I overhead, P, P overhead, MV),
/// for diagnostics.
pub fn predict_terms(m: &Model, c: &ChunkStats, gain: &Gain, crf: f64) -> [f64; 5] {
    let p = &m.rate;
    let (nz_i, lv_i) = features(m, &c.i_hist, gain, crf + p.d_i);
    let (nz_p, lv_p) = features(m, &c.p_hist, gain, crf + p.d_p);
    let scale = c.p_frames() / c.p_sampled.max(1) as f64;
    [
        p.a_i * nz_i + p.b_i * lv_i,
        p.c_i * c.mbs_per_frame,
        scale * (p.a_p * nz_p + p.b_p * lv_p),
        p.c_p * c.mbs_per_frame * c.p_frames(),
        p.mv_p * c.mv_cost * scale * m.qstep(30.0) / m.qstep(crf),
    ]
}

/// Predicted encoded size of a chunk, in bytes.
pub fn predict_bytes(m: &Model, c: &ChunkStats, gain: &Gain, crf: f64) -> f64 {
    let p = &m.rate;
    let (nz_i, lv_i) = features(m, &c.i_hist, gain, crf + p.d_i);
    let (nz_p, lv_p) = features(m, &c.p_hist, gain, crf + p.d_p);
    let scale = c.p_frames() / c.p_sampled.max(1) as f64;
    let mv = p.mv_p * c.mv_cost * scale * m.qstep(30.0) / m.qstep(crf);
    let bits = p.a_i * nz_i + p.b_i * lv_i + p.c_i * c.mbs_per_frame
        + scale * (p.a_p * nz_p + p.b_p * lv_p)
        + p.c_p * c.mbs_per_frame * c.p_frames()
        + mv;
    (bits / 8.0).max(64.0)
}

/// Detail removed by the rung alone (luma MSE per pixel).
pub fn scale_mse(c: &ChunkStats, gain: &Gain) -> f64 {
    let mut e = 0.0;
    for k in 0..NPOS {
        e += c.src_energy[k] * (1.0 - gain.0[k]).powi(2);
    }
    gain.1 * e / (c.src_frames.max(1) as f64 * c.pixels_per_frame)
}

/// Residual coefficient energy per pixel after the rung's gain.
pub fn residual_energy(c: &ChunkStats, gain: &Gain) -> f64 {
    let mut e = 0.0;
    for k in 0..NPOS {
        let g2 = gain.0[k] * gain.0[k];
        for i in 0..NBINS {
            let n = c.p_hist[k * NBINS + i];
            if n != 0 {
                let m = bin_mag(i);
                e += n as f64 * m * m * g2;
            }
        }
    }
    e / (c.p_sampled.max(1) as f64 * c.pixels_per_frame)
}

/// Predicted luma MSE of the encode against the unfiltered source.
pub fn predict_mse(m: &Model, c: &ChunkStats, gain: &Gain, crf: f64) -> f64 {
    predict_mse_with(m, scale_mse(c, gain), residual_energy(c, gain), crf)
}

/// `predict_mse` with the CRF-independent parts precomputed.
pub fn predict_mse_with(m: &Model, scale: f64, residual: f64, crf: f64) -> f64 {
    let d = &m.dist;
    scale + d.ln_a.exp() * m.qstep(crf).powf(d.b) * residual.max(1e-6).powf(d.g)
}

/// VMAF (vmaf_v0.6.1neg, the model smallifier.py used) predicted from luma
/// PSNR: a logistic fitted by tools/vmaf_map.py to 2,400 two-second windows
/// of game footage (4 clips x 4 rungs x 5 CRFs; rmse ~11 VMAF, so only a
/// rough guide per window). Its shape is what matters for allocation: it
/// saturates for clean scenes and collapses for starved ones, so the planner
/// protects hard scenes instead of trading them for easy ones. (A variant
/// with a rung term fitted at equal PSNR made real-VMAF results worse; the
/// planner compares rungs at equal bytes, which that fit doesn't see.)
pub fn vmaf_proxy(psnr: f64) -> f64 {
    96.0 / (1.0 + (-(psnr - 26.883) / 3.798).exp())
}

pub fn psnr(mse: f64) -> f64 {
    10.0 * (255.0f64 * 255.0 / mse.max(1e-6)).log10()
}
