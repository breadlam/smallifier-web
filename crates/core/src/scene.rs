//! Scene cuts and chunking.
//!
//! A *scene* is the allocation unit: every chunk of a scene gets the same
//! prefilter rung and quality, so sharpness never changes mid-shot.
//! A *chunk* is the encode unit: an independent closed-GOP x264 run, which
//! lets chunks encode in parallel workers.

/// Frame indices where a new scene starts (always includes 0).
///
/// `diffs[t]` is the thumbnail difference between frame t-1 and t
/// (`diffs[0]` is ignored). A cut needs an absolute jump and must also stand
/// out from its neighbourhood, so fast camera motion in games, which makes
/// every frame differ a lot, doesn't shred a shot into pieces.
pub fn detect_cuts(diffs: &[f32], fps: f64, min_scene_sec: f64) -> Vec<usize> {
    let n = diffs.len();
    let mut cuts = vec![0];
    if n < 2 {
        return cuts;
    }
    let min_len = (min_scene_sec * fps).round().max(1.0) as usize;
    let half = (fps * 0.5).round().max(4.0) as usize;
    let mut last = 0;
    for t in 1..n {
        let d = diffs[t];
        if d < ABS_THRESHOLD || t - last < min_len || n - t < min_len / 2 {
            continue;
        }
        let lo = t.saturating_sub(half).max(1);
        let hi = (t + half + 1).min(n);
        let mut local: Vec<f32> = (lo..hi).filter(|&i| i != t).map(|i| diffs[i]).collect();
        if local.is_empty() {
            continue;
        }
        local.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = local[local.len() / 2];
        if d > REL_THRESHOLD * med + 2.0 {
            cuts.push(t);
            last = t;
        }
    }
    cuts
}

const ABS_THRESHOLD: f32 = 18.0;
const REL_THRESHOLD: f32 = 3.0;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// First output frame (inclusive).
    pub start: usize,
    /// One past the last output frame.
    pub end: usize,
    pub scene: usize,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// Splits scenes into chunks no longer than `max_len` frames (evenly, so a
/// long shot doesn't end in a tiny tail chunk).
pub fn chunk_scenes(cuts: &[usize], total: usize, max_len: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (si, &s) in cuts.iter().enumerate() {
        let e = cuts.get(si + 1).copied().unwrap_or(total);
        if e <= s {
            continue;
        }
        let len = e - s;
        let parts = len.div_ceil(max_len.max(1));
        for p in 0..parts {
            let a = s + len * p / parts;
            let b = s + len * (p + 1) / parts;
            out.push(Chunk { start: a, end: b, scene: si });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_and_chunks() {
        let mut d = vec![2.0f32; 300];
        d[100] = 60.0;
        d[105] = 60.0; // too close to the previous cut
        d[200] = 60.0;
        let c = detect_cuts(&d, 30.0, 1.0);
        assert_eq!(c, vec![0, 100, 200]);
        let ch = chunk_scenes(&c, 300, 60);
        assert_eq!(ch.len(), 6);
        assert!(ch.iter().all(|c| c.len() == 50));
        assert_eq!(ch[2].scene, 1);
    }

    #[test]
    fn busy_motion_is_not_a_cut() {
        // Uniformly high differences (fast pans) with one real cut.
        let mut d: Vec<f32> = (0..300).map(|i| 25.0 + (i % 7) as f32).collect();
        d[150] = 140.0;
        assert_eq!(detect_cuts(&d, 30.0, 1.0), vec![0, 150]);
    }
}
