//! Compiles vendored x264 and libopus without running their configure
//! scripts: the handful of platform defines they need are written here.
//!
//! Native builds use the host C compiler (for the lab tools); wasm32 builds
//! use clang from the wasi-sdk at $WASI_SDK_PATH.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const X264_SRCS: &[&str] = &[
    "common/osdep.c", "common/base.c", "common/cpu.c", "common/tables.c",
    "encoder/api.c",
];

// Compiled once per bit depth by x264's Makefile; we only build 8-bit.
const X264_SRCS_X: &[&str] = &[
    "common/mc.c", "common/predict.c", "common/pixel.c", "common/macroblock.c",
    "common/frame.c", "common/dct.c", "common/cabac.c", "common/common.c",
    "common/rectangle.c", "common/set.c", "common/quant.c", "common/deblock.c",
    "common/vlc.c", "common/mvpred.c", "common/bitstream.c",
    "encoder/analyse.c", "encoder/me.c", "encoder/ratecontrol.c", "encoder/set.c",
    "encoder/macroblock.c", "encoder/cabac.c", "encoder/cavlc.c",
    "encoder/encoder.c", "encoder/lookahead.c",
];

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let x264_dir = root.join("third_party/x264");
    let opus_dir = root.join("third_party/opus");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wasm = env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "wasm32";

    for d in [&x264_dir, &opus_dir] {
        if !d.join("COPYING").exists() {
            panic!("{} missing: run tools/fetch-deps.sh", d.display());
        }
    }
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");

    let cfg_dir = out.join("x264cfg");
    fs::create_dir_all(&cfg_dir).unwrap();
    write_x264_config(&cfg_dir, &x264_dir);

    let base = || {
        let mut b = cc::Build::new();
        if wasm {
            let sdk = PathBuf::from(env::var("WASI_SDK_PATH").expect("WASI_SDK_PATH not set"));
            let sdk = sdk.canonicalize().unwrap_or(sdk);
            b.compiler(sdk.join("bin/clang"))
                .archiver(sdk.join("bin/llvm-ar"))
                .flag(format!("--sysroot={}", sdk.join("share/wasi-sysroot").display()))
                .flag("-msimd128");
        }
        b.opt_level(3).warnings(false).define("NDEBUG", None);
        b
    };

    // ---- x264 ----
    let mut b = base();
    b.include(&cfg_dir).include(&x264_dir).flag("-std=gnu99").define("_GNU_SOURCE", None)
        .flag("-ffast-math");
    for s in X264_SRCS {
        b.file(x264_dir.join(s));
    }
    b.compile("x264_base");

    // The bit-depth-templated sources go into a separate library so that the
    // object names do not collide with the base set.
    let mut b = base();
    b.include(&cfg_dir).include(&x264_dir).flag("-std=gnu99").define("_GNU_SOURCE", None)
        .flag("-ffast-math")
        .define("HIGH_BIT_DEPTH", "0").define("BIT_DEPTH", "8");
    for s in X264_SRCS_X {
        b.file(x264_dir.join(s));
    }
    b.file("csrc/x264_shim.c");
    b.compile("x264_8");

    // ---- libopus (float build) ----
    let mut b = base();
    for inc in ["include", "silk", "celt", "silk/float", "src"] {
        b.include(opus_dir.join(inc));
    }
    b.define("OPUS_BUILD", None).define("VAR_ARRAYS", None).define("HAVE_LRINTF", None)
        .define("HAVE_LRINT", None);
    let mut files = Vec::new();
    files.extend(mk_list(&opus_dir.join("silk_sources.mk"), "SILK_SOURCES"));
    files.extend(mk_list(&opus_dir.join("silk_sources.mk"), "SILK_SOURCES_FLOAT"));
    files.extend(mk_list(&opus_dir.join("celt_sources.mk"), "CELT_SOURCES"));
    files.extend(mk_list(&opus_dir.join("opus_sources.mk"), "OPUS_SOURCES"));
    files.extend(mk_list(&opus_dir.join("opus_sources.mk"), "OPUS_SOURCES_FLOAT"));
    for f in files {
        b.file(opus_dir.join(f));
    }
    b.file("csrc/opus_shim.c");
    b.compile("opus");

    if !wasm {
        println!("cargo:rustc-link-lib=m");
    }
}

/// Reads a `NAME = \` file list out of one of libopus's .mk fragments.
fn mk_list(path: &Path, name: &str) -> Vec<String> {
    let text = fs::read_to_string(path).unwrap();
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let t = line.trim();
        if !inside {
            if t.starts_with(name) && t[name.len()..].trim_start().starts_with('=') {
                inside = true;
            }
            continue;
        }
        let f = t.trim_end_matches('\\').trim();
        if !f.is_empty() {
            out.push(f.to_string());
        }
        if !t.ends_with('\\') {
            break;
        }
    }
    assert!(!out.is_empty(), "no {name} in {}", path.display());
    out
}

fn write_x264_config(dir: &Path, x264_dir: &Path) {
    // Everything x264's configure would set, with no asm, no threads and
    // 8-bit 4:2:0 only. Undefined HAVE_* evaluate to 0 in #if, but x264's
    // configure defines them all explicitly, so do the same.
    let zero = [
        "ALTIVEC", "ALTIVEC_H", "MMX", "ARMV6", "ARMV6T2", "NEON", "AARCH64",
        "BEOSTHREAD", "POSIXTHREAD", "WIN32THREAD", "THREAD", "SWSCALE", "LAVF",
        "FFMS", "GPAC", "AVS", "VECTOREXT", "INTERLACED", "CPU_COUNT", "OPENCL",
        "THP", "LSMASH", "X86_INLINE_ASM", "AS_FUNC", "INTEL_DISPATCHER", "MSA",
        "LSX", "MMAP", "WINRT", "VSX", "ARM_INLINE_ASM", "ELF_AUX_INFO",
        "GETAUXVAL", "SYSCONF", "SYNC_FETCH_AND_ADD", "BITDEPTH10", "DOTPROD",
        "I8MM", "SVE", "SVE2", "AS_ARCHEXT_DOTPROD_DIRECTIVE",
        "AS_ARCHEXT_I8MM_DIRECTIVE", "AS_ARCHEXT_SVE_DIRECTIVE",
        "AS_ARCHEXT_SVE2_DIRECTIVE", "MALLOC_H",
    ];
    let one = ["LOG2F", "GPL", "STRTOK_R", "CLOCK_GETTIME", "BITDEPTH8"];
    let mut c = String::from("#define ARCH_GENERIC 1\n#define SYS_GENERIC 1\n#define STACK_ALIGNMENT 16\n");
    c.push_str("#define CHROMA_FORMAT CHROMA_420\n");
    for z in zero {
        c.push_str(&format!("#define HAVE_{z} 0\n"));
    }
    for o in one {
        c.push_str(&format!("#define HAVE_{o} 1\n"));
    }
    fs::write(dir.join("config.h"), c).unwrap();

    let header = fs::read_to_string(x264_dir.join("x264.h")).unwrap();
    let build = header
        .lines()
        .find_map(|l| l.strip_prefix("#define X264_BUILD").map(|v| v.trim().to_string()))
        .unwrap();
    let cfg = format!(
        "#define X264_GPL           1\n#define X264_INTERLACED    0\n\
         #define X264_BIT_DEPTH     8\n#define X264_CHROMA_FORMAT X264_CSP_I420\n\
         #define X264_VERSION \" smallifier\"\n#define X264_POINTVER \"0.{build}.x smallifier\"\n"
    );
    fs::write(dir.join("x264_config.h"), cfg).unwrap();
}
