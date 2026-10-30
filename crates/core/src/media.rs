//! Container-independent description of an input file's tracks.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Video,
    Audio,
}

/// One coded sample (frame or audio packet) and where it lives in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    /// Presentation time in seconds (edit lists already applied).
    pub pts: f64,
    /// Decode time in seconds.
    pub dts: f64,
    pub keyframe: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: u32,
    pub kind: Kind,
    /// WebCodecs codec string ("avc1.640028", "opus", ...), or empty if the
    /// codec can't be handed to WebCodecs.
    pub codec: String,
    /// Human-readable codec name for messages.
    pub codec_name: String,
    /// WebCodecs `description` (avcC, hvcC, AudioSpecificConfig, ...).
    #[serde(with = "hex_bytes")]
    pub description: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Clockwise rotation from the container's display matrix.
    pub rotation: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: f64,
    #[serde(skip)]
    pub samples: Vec<Sample>,
}

impl TrackInfo {
    pub fn new(id: u32, kind: Kind) -> Self {
        TrackInfo {
            id,
            kind,
            codec: String::new(),
            codec_name: String::new(),
            description: Vec::new(),
            width: 0,
            height: 0,
            rotation: 0,
            sample_rate: 0,
            channels: 0,
            duration: 0.0,
            samples: Vec::new(),
        }
    }
}

/// Description bytes travel to JS as hex inside JSON (they are small).
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let mut out = String::with_capacity(v.len() * 2);
        for b in v {
            out.push_str(&format!("{b:02x}"));
        }
        s.serialize_str(&out)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(serde::de::Error::custom))
            .collect()
    }
}

// ------------------------------------------------------------ codec strings

pub fn avc_codec_string(avcc: &[u8]) -> Option<String> {
    (avcc.len() >= 4).then(|| format!("avc1.{:02x}{:02x}{:02x}", avcc[1], avcc[2], avcc[3]))
}

/// ISO/IEC 14496-15 Annex E codec string from an hvcC record.
pub fn hevc_codec_string(prefix: &str, h: &[u8]) -> Option<String> {
    if h.len() < 13 {
        return None;
    }
    let space = h[1] >> 6;
    let tier = (h[1] >> 5) & 1;
    let idc = h[1] & 0x1f;
    let compat = u32::from_be_bytes([h[2], h[3], h[4], h[5]]).reverse_bits();
    let level = h[12];
    let mut s = format!(
        "{prefix}.{}{idc}.{compat:x}.{}{level}",
        ["", "A", "B", "C"][space as usize],
        if tier == 1 { "H" } else { "L" }
    );
    // Constraint flags: 6 bytes, trailing zero bytes omitted.
    let cons = &h[6..12];
    let n = cons.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    for b in &cons[..n] {
        s.push_str(&format!(".{b:x}"));
    }
    Some(s)
}

/// From a vpcC box payload (version/flags first).
pub fn vp9_codec_string(v: &[u8]) -> Option<String> {
    if v.len() < 8 {
        return None;
    }
    let profile = v[4];
    let level = v[5];
    let depth = v[6] >> 4;
    Some(format!("vp09.{profile:02}.{level:02}.{depth:02}"))
}

pub fn av1_codec_string(c: &[u8]) -> Option<String> {
    if c.len() < 4 {
        return None;
    }
    let profile = c[1] >> 5;
    let level = c[1] & 0x1f;
    let tier = if c[2] & 0x80 != 0 { 'H' } else { 'M' };
    let high = c[2] & 0x40 != 0;
    let twelve = c[2] & 0x20 != 0;
    let depth = if high && twelve { 12 } else if high { 10 } else { 8 };
    Some(format!("av01.{profile}.{level:02}{tier}.{depth:02}"))
}

/// AAC object type from an AudioSpecificConfig.
pub fn aac_codec_string(asc: &[u8]) -> String {
    let aot = if asc.is_empty() { 2 } else { asc[0] >> 3 };
    format!("mp4a.40.{}", if aot == 0 { 2 } else { aot })
}

/// Builds an OpusHead (as WebCodecs wants for `description`) from an MP4
/// dOps box payload.
pub fn opus_head_from_dops(d: &[u8]) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    if d.len() < 11 {
        return h;
    }
    h.push(1); // version
    h.push(d[1]); // channels
    h.extend_from_slice(&u16::from_be_bytes([d[2], d[3]]).to_le_bytes());
    h.extend_from_slice(&u32::from_be_bytes([d[4], d[5], d[6], d[7]]).to_le_bytes());
    h.extend_from_slice(&i16::from_be_bytes([d[8], d[9]]).to_le_bytes());
    h.push(d[10]); // mapping family
    if d[10] != 0 && d.len() >= 13 {
        h.extend_from_slice(&d[11..]);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hevc_string() {
        // Main profile, level 4.0 (120), progressive+frame-only constraint (0xb0).
        let mut h = vec![1u8, 0x01, 0x60, 0, 0, 0, 0xb0, 0, 0, 0, 0, 0, 120];
        h.extend_from_slice(&[0; 10]);
        assert_eq!(hevc_codec_string("hvc1", &h).unwrap(), "hvc1.1.6.L120.b0");
    }

    #[test]
    fn av1_string() {
        assert_eq!(av1_codec_string(&[0x81, 0x08, 0x0c, 0]).unwrap(), "av01.0.08M.08");
    }
}
