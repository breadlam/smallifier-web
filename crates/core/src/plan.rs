//! Budget allocation and the encode-time controller.
//!
//! Planning works on *predicted* operating points: for every scene, every
//! (prefilter rung, CRF) pair gets a predicted size and quality from the
//! model, the scene's upper convex hull of (bytes, frames x quality) is
//! taken, and a greedy equal-slope pass spends the budget where it buys the
//! most quality (the same allocation smallifier.py ran on measured points).
//! Quality is predicted VMAF (from predicted PSNR), as smallifier.py
//! maximised duration-weighted mean VMAF.
//!
//! Encoding is two-pass per chunk, so a chunk's size is whatever target it
//! is given (to ~2%). The controller sets each chunk's final target only
//! when its first pass has finished, from the budget actually left.

use crate::model::{self, ChunkStats, Gain, Model};
use crate::scene::Chunk;
use serde::{Deserialize, Serialize};

const CRF_STEP: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct Rung {
    pub w: usize,
    pub h: usize,
    pub gain: Gain,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    rung: usize,
    crf: f64,
    bytes: f64,
    value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenePlan {
    pub rung: usize,
    pub crf: f64,
    pub bytes: f64,
    pub psnr: f64,
    pub frames: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub scenes: Vec<ScenePlan>,
    /// Planned bytes per chunk (sums to the budget when feasible).
    pub chunk_bytes: Vec<f64>,
    /// PSNR of the whole (from mean MSE), and mean predicted VMAF.
    pub predicted_psnr: f64,
    pub predicted_vmaf: f64,
    /// False when even the cheapest option everywhere exceeds the budget.
    pub feasible: bool,
}

fn hull(mut pts: Vec<Point>) -> Vec<Point> {
    pts.sort_by(|a, b| a.bytes.partial_cmp(&b.bytes).unwrap().then(b.value.partial_cmp(&a.value).unwrap()));
    let mut pareto: Vec<Point> = Vec::new();
    for p in pts {
        if pareto.last().is_none_or(|l| p.value > l.value) {
            pareto.push(p);
        }
    }
    // Upper concave hull.
    let mut h: Vec<Point> = Vec::new();
    for p in pareto {
        while h.len() >= 2 {
            let (a, b) = (h[h.len() - 2], h[h.len() - 1]);
            let cross = (b.bytes - a.bytes) * (p.value - a.value) - (b.value - a.value) * (p.bytes - a.bytes);
            if cross >= 0.0 {
                h.pop();
            } else {
                break;
            }
        }
        h.push(p);
    }
    h
}

/// Allocates `budget` bytes over the scenes. `stats[i]` belongs to `chunks[i]`.
pub fn allocate(m: &Model, chunks: &[Chunk], stats: &[ChunkStats], rungs: &[Rung], budget: f64) -> Plan {
    let nscenes = chunks.iter().map(|c| c.scene + 1).max().unwrap_or(0);
    let mut hulls = Vec::with_capacity(nscenes);
    let mut scene_frames = vec![0usize; nscenes];
    for c in chunks {
        scene_frames[c.scene] += c.len();
    }
    for s in 0..nscenes {
        let idx: Vec<usize> = (0..chunks.len()).filter(|&i| chunks[i].scene == s).collect();
        let frames = scene_frames[s] as f64;
        let mut pts = Vec::new();
        for (ri, r) in rungs.iter().enumerate() {
            let fixed: Vec<(f64, f64)> =
                idx.iter().map(|&i| (model::scale_mse(&stats[i], &r.gain), model::residual_energy(&stats[i], &r.gain))).collect();
            let mut crf = m.crf_min;
            while crf <= m.crf_max + 1e-9 {
                let (mut bytes, mut se) = (0.0, 0.0);
                for (j, &i) in idx.iter().enumerate() {
                    bytes += model::predict_bytes(m, &stats[i], &r.gain, crf);
                    se += model::predict_mse_with(m, fixed[j].0, fixed[j].1, crf) * chunks[i].len() as f64;
                }
                let q = model::vmaf_proxy(model::psnr(se / frames));
                pts.push(Point { rung: ri, crf, bytes, value: q * frames });
                crf += CRF_STEP;
            }
        }
        hulls.push(hull(pts));
    }

    let mut choice = vec![0usize; nscenes];
    let mut total: f64 = hulls.iter().map(|h| h[0].bytes).sum();
    let feasible = total <= budget;
    loop {
        let mut best: Option<(usize, f64)> = None;
        for s in 0..nscenes {
            let h = &hulls[s];
            let k = choice[s];
            if k + 1 >= h.len() {
                continue;
            }
            let db = h[k + 1].bytes - h[k].bytes;
            if db <= 0.0 || total + db > budget {
                continue;
            }
            let slope = (h[k + 1].value - h[k].value) / db;
            if best.is_none_or(|(_, b)| slope > b) {
                best = Some((s, slope));
            }
        }
        let Some((s, _)) = best else { break };
        total += hulls[s][choice[s] + 1].bytes - hulls[s][choice[s]].bytes;
        choice[s] += 1;
    }

    let mut scenes = Vec::with_capacity(nscenes);
    let mut chunk_bytes = vec![0.0; chunks.len()];
    let (mut total_se, mut weighted_v) = (0.0, 0.0);
    for s in 0..nscenes {
        let p = hulls[s][choice[s]];
        let gain = &rungs[p.rung].gain;
        let mut se = 0.0;
        for (i, c) in chunks.iter().enumerate().filter(|(_, c)| c.scene == s) {
            chunk_bytes[i] = model::predict_bytes(m, &stats[i], gain, p.crf);
            se += model::predict_mse(m, &stats[i], gain, p.crf) * c.len() as f64;
        }
        let psnr = model::psnr(se / scene_frames[s] as f64);
        total_se += se;
        weighted_v += model::vmaf_proxy(psnr) * scene_frames[s] as f64;
        scenes.push(ScenePlan { rung: p.rung, crf: p.crf, bytes: p.bytes, psnr, frames: scene_frames[s] });
    }
    // The greedy stops short of the budget by up to one hull step; two-pass
    // encoding hits whatever we ask for, so spread the remainder.
    let sum: f64 = chunk_bytes.iter().sum();
    if sum > 0.0 {
        let k = budget / sum;
        let k = if feasible { k.max(1.0) } else { k };
        chunk_bytes.iter_mut().for_each(|b| *b *= k);
    }
    let frames: usize = scene_frames.iter().sum();
    let n = frames.max(1) as f64;
    Plan { scenes, chunk_bytes, predicted_psnr: model::psnr(total_se / n), predicted_vmaf: weighted_v / n, feasible }
}

// --------------------------------------------------------------- controller

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum State {
    Pending,
    FirstPass,
    SecondPass { requested: f64 },
    Done { requested: f64, actual: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Controller {
    pub budget: f64,
    pub planned: Vec<f64>,
    pub frames: Vec<usize>,
    /// The encoder's two-pass bias prior: base + per_frame / frames.
    pub bias_base: f64,
    pub bias_per_frame: f64,
    /// Relative size error of one chunk's second pass (sd).
    pub chunk_sigma: f64,
    pub state: Vec<State>,
    /// Learned correction on top of `prior_bias`, from finished chunks.
    pub bias: f64,
    bias_n: f64,
}

/// A second-pass size request, or a re-run of one.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Request {
    pub chunk: usize,
    pub bytes: f64,
}

const PRIOR_WEIGHT: f64 = 2.0;

impl Controller {
    pub fn new(budget: f64, planned: Vec<f64>, frames: Vec<usize>, model: &Model) -> Self {
        let n = planned.len();
        Controller {
            budget,
            planned,
            frames,
            bias_base: model.bias_base,
            bias_per_frame: model.bias_per_frame,
            chunk_sigma: model.chunk_sigma,
            state: vec![State::Pending; n],
            bias: 1.0,
            bias_n: 0.0,
        }
    }

    /// Prior actual/requested ratio of a two-pass chunk of `frames` frames
    /// (x264 overshoots short chunks, whose I frame dominates).
    fn prior_bias(&self, frames: usize) -> f64 {
        self.bias_base + self.bias_per_frame / frames.max(1) as f64
    }

    /// Expected actual/requested ratio for chunk i's second pass.
    pub fn expected_ratio(&self, i: usize) -> f64 {
        self.prior_bias(self.frames[i]) * self.bias
    }

    /// Next chunk to start (largest planned first: long jobs first keeps
    /// workers busy to the end, and small late chunks absorb error).
    pub fn next_chunk(&mut self) -> Option<usize> {
        let i = (0..self.planned.len())
            .filter(|&i| self.state[i] == State::Pending)
            .max_by(|&a, &b| self.planned[a].partial_cmp(&self.planned[b]).unwrap())?;
        self.state[i] = State::FirstPass;
        Some(i)
    }

    /// Bytes to ask the second pass of chunk `i` for, given everything
    /// known now. The remaining budget is shared among the chunks without a
    /// target yet in proportion to their planned sizes, after holding back
    /// one sigma of the size uncertainty of everything not finished (each
    /// two-pass chunk lands within `chunk_sigma`, independently). Early chunks thus
    /// keep a cushion and the last ones get exactly what is left.
    pub fn second_pass_target(&mut self, i: usize) -> f64 {
        let mut committed = 0.0; // done (actual) + running second passes (expected)
        let mut open = 0.0; // planned bytes of chunks without a target yet, incl. i
        let mut var = 0.0;
        for (j, s) in self.state.iter().enumerate() {
            match *s {
                State::Done { actual, .. } => committed += actual,
                State::SecondPass { requested } => {
                    let e = requested * self.expected_ratio(j);
                    committed += e;
                    var += (self.chunk_sigma * e).powi(2);
                }
                _ => {
                    open += self.planned[j];
                    var += (self.chunk_sigma * self.planned[j]).powi(2);
                }
            }
        }
        let margin = var.sqrt();
        let scale = ((self.budget - committed - margin) / open.max(1.0)).max(0.05);
        // Keep corrections gentle: a few early misses shouldn't swing a chunk wildly.
        let scale = scale.clamp(0.6, 1.4);
        let bytes = self.planned[i] * scale / self.expected_ratio(i);
        self.state[i] = State::SecondPass { requested: bytes };
        bytes
    }

    pub fn finished(&mut self, i: usize, actual: f64) {
        let requested = match self.state[i] {
            State::SecondPass { requested } | State::Done { requested, .. } => requested,
            _ => actual,
        };
        self.state[i] = State::Done { requested, actual };
        // Learn how far off the prior is, shrunk towards "not at all".
        let r = (actual / requested.max(1.0) / self.prior_bias(self.frames[i])).clamp(0.7, 1.5);
        self.bias = (self.bias * (PRIOR_WEIGHT + self.bias_n) + r) / (PRIOR_WEIGHT + self.bias_n + 1.0);
        self.bias_n += 1.0;
    }

    /// Chunk `i` was already encoded (an earlier, interrupted run): record
    /// it as finished, as if its second pass had been asked for `requested`.
    pub fn restore(&mut self, i: usize, requested: f64, actual: f64) {
        self.state[i] = State::SecondPass { requested };
        self.finished(i, actual);
    }

    pub fn total(&self) -> f64 {
        self.state.iter().map(|s| if let State::Done { actual, .. } = s { *actual } else { 0.0 }).sum()
    }

    pub fn all_done(&self) -> bool {
        self.state.iter().all(|s| matches!(s, State::Done { .. }))
    }

    /// Second-pass re-runs that remove `excess` bytes (or, if negative, use
    /// that many more). The caller measures the excess on the real muxed
    /// file, so container overhead is exact. `workers` re-runs can go in
    /// parallel.
    ///
    /// The change is spread over the shortest chunks of at least `MIN_FIX`
    /// frames that can absorb it (each by at most 15%), so the re-runs
    /// finish quickly in parallel; very short chunks are avoided because
    /// their I frame dominates and two-pass can barely move them. Requests
    /// are corrected by each chunk's own observed actual/requested ratio.
    pub fn fixups(&mut self, excess: f64, workers: usize) -> Vec<Request> {
        const MIN_FIX: usize = 60;
        if excess == 0.0 {
            return Vec::new();
        }
        let need = excess.abs();
        let mut done: Vec<(usize, f64, f64)> = self
            .state
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                State::Done { actual, requested } => Some((i, *actual, (actual / requested.max(1.0)).clamp(0.7, 1.5))),
                _ => None,
            })
            .collect();
        let frames = &self.frames;
        // Long-enough chunks first, shortest of those first; then the rest.
        done.sort_by_key(|&(i, ..)| (frames[i] < MIN_FIX, frames[i]));
        let mut pick = Vec::new();
        let mut capacity = 0.0;
        for &d in &done {
            if capacity >= need && pick.len() >= workers.clamp(1, 2) {
                break;
            }
            pick.push(d);
            capacity += 0.15 * d.1;
        }
        let sum: f64 = pick.iter().map(|p| p.1).sum();
        let mut out = Vec::new();
        for (i, actual, ratio) in pick {
            let delta = (need * actual / sum).min(0.25 * actual);
            let target = if excess > 0.0 { actual - delta } else { actual + delta };
            out.push(Request { chunk: i, bytes: target / ratio });
            self.state[i] = State::SecondPass { requested: target / ratio };
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_fills_budget() {
        let mut c = Controller::new(1000.0, vec![400.0, 300.0, 300.0], vec![200, 150, 150], &model::MEDIUM);
        let a = c.next_chunk().unwrap();
        assert_eq!(a, 0);
        let t = c.second_pass_target(a);
        c.finished(a, t * 1.1); // came in 10% over the request
        let b = c.next_chunk().unwrap();
        let tb = c.second_pass_target(b);
        assert!(tb < 300.0 / 1.02, "should compensate: {tb}");
        c.finished(b, tb * c.expected_ratio(b));
        let d = c.next_chunk().unwrap();
        let td = c.second_pass_target(d);
        c.finished(d, td * c.expected_ratio(d));
        assert!((c.total() - 1000.0).abs() < 20.0, "{}", c.total());
    }

    #[test]
    fn fixups_cut_overshoot() {
        let mut c = Controller::new(1000.0, vec![500.0, 500.0], vec![100, 100], &model::MEDIUM);
        for _ in 0..2 {
            let i = c.next_chunk().unwrap();
            c.second_pass_target(i);
        }
        c.finished(0, 560.0);
        c.finished(1, 520.0);
        let requested: Vec<f64> = (0..2)
            .map(|i| match c.state[i] {
                State::Done { requested, .. } => requested,
                _ => unreachable!(),
            })
            .collect();
        let actual = [560.0, 520.0];
        let f = c.fixups(80.0, 4);
        assert!(!f.is_empty());
        // Each re-run is asked for its share, corrected by its own
        // actual/requested ratio, so the expected outcomes remove the excess.
        let before: f64 = f.iter().map(|r| actual[r.chunk]).sum();
        let after: f64 = f.iter().map(|r| r.bytes * actual[r.chunk] / requested[r.chunk]).sum();
        assert!(before - after >= 79.0, "cut {} of the 80 byte excess", before - after);
    }
}
