/* Narrow C interface over libx264 so the Rust side never has to mirror
 * x264_param_t / x264_picture_t layouts. One encoder = one chunk. */
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "x264.h"

typedef struct {
    x264_t *h;
    x264_param_t param;
    uint8_t *buf;
    size_t cap, len;
    int64_t pts, dts;
    int keyframe, frame_type;
    double psnr_y, ssim;
    uint8_t sps[256], pps[256];
    int sps_len, pps_len;
} sx_enc;

/* x264 only measures PSNR/SSIM at log level >= INFO, so stats runs keep
 * INFO; only errors are passed on (to stderr, i.e. the JS console). */
static void quiet_log(void *priv, int level, const char *fmt, va_list ap) {
    (void)priv;
    if (level > X264_LOG_ERROR) return;
    fputs("x264: ", stderr);
    vfprintf(stderr, fmt, ap);
}

static int ensure(sx_enc *e, size_t need) {
    if (need <= e->cap) return 0;
    size_t cap = e->cap ? e->cap : 65536;
    while (cap < need) cap *= 2;
    uint8_t *b = realloc(e->buf, cap);
    if (!b) return -1;
    e->buf = b;
    e->cap = cap;
    return 0;
}

/* Copies NAL units (4-byte length prefixed, since b_annexb = 0) into e->buf,
 * dropping SEI: x264 writes a ~600 byte version/options SEI into every
 * stream, and with chunked encoding that would repeat per chunk. */
static int collect(sx_enc *e, x264_nal_t *nal, int n) {
    e->len = 0;
    for (int i = 0; i < n; i++) {
        if (nal[i].i_type == NAL_SEI || nal[i].i_type == NAL_AUD) continue;
        if (ensure(e, e->len + nal[i].i_payload)) return -1;
        memcpy(e->buf + e->len, nal[i].p_payload, nal[i].i_payload);
        e->len += nal[i].i_payload;
    }
    return (int)e->len;
}

/* opts: "key=value:key=value" applied with x264_param_parse after the
 * preset, so any x264 option can be set from Rust. */
sx_enc *sx_open(int w, int h, int fps_num, int fps_den, const char *preset,
                const char *tune, const char *opts, int want_stats) {
    sx_enc *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    if (x264_param_default_preset(&e->param, preset, (tune && *tune) ? tune : NULL) < 0)
        goto fail;
    x264_param_t *p = &e->param;
    p->i_width = w;
    p->i_height = h;
    p->i_csp = X264_CSP_I420;
    p->i_fps_num = fps_num;
    p->i_fps_den = fps_den;
    p->i_timebase_num = fps_den;
    p->i_timebase_den = fps_num;
    p->b_vfr_input = 0;
    p->i_threads = 1;
    p->b_annexb = 0;
    p->b_repeat_headers = 0;
    p->b_stitchable = 1;
    p->i_log_level = want_stats ? X264_LOG_INFO : X264_LOG_ERROR;
    p->pf_log = quiet_log;
    p->analyse.b_psnr = want_stats;
    p->analyse.b_ssim = want_stats;
    p->vui.i_colorprim = 1;  /* BT.709 */
    p->vui.i_transfer = 1;
    p->vui.i_colmatrix = 1;

    /* "fastfirstpass" is not an x264 option (the CLI applies it); handle
     * it here, after the other options, as x264cli does. */
    int fast_first = 0;
    if (opts && *opts) {
        char *s = strdup(opts), *save = NULL;
        for (char *kv = strtok_r(s, ":", &save); kv; kv = strtok_r(NULL, ":", &save)) {
            char *eq = strchr(kv, '=');
            int r;
            if (!strcmp(kv, "fastfirstpass")) { fast_first = 1; continue; }
            if (eq) { *eq = 0; r = x264_param_parse(p, kv, eq + 1); }
            else r = x264_param_parse(p, kv, "1");
            if (r) { free(s); goto fail; }
        }
        free(s);
    }
    if (fast_first) x264_param_apply_fastfirstpass(p);
    if (x264_param_apply_profile(p, "high") < 0) goto fail;
    e->h = x264_encoder_open(p);
    if (!e->h) goto fail;

    x264_nal_t *nal;
    int n;
    if (x264_encoder_headers(e->h, &nal, &n) < 0) goto fail;
    for (int i = 0; i < n; i++) {
        int plen = nal[i].i_payload - 4;  /* strip length prefix */
        if (plen <= 0 || plen > 256) continue;
        if (nal[i].i_type == NAL_SPS) { memcpy(e->sps, nal[i].p_payload + 4, plen); e->sps_len = plen; }
        if (nal[i].i_type == NAL_PPS) { memcpy(e->pps, nal[i].p_payload + 4, plen); e->pps_len = plen; }
    }
    return e;
fail:
    if (e->h) x264_encoder_close(e->h);
    free(e);
    return NULL;
}

const uint8_t *sx_sps(sx_enc *e, int *len) { *len = e->sps_len; return e->sps; }
const uint8_t *sx_pps(sx_enc *e, int *len) { *len = e->pps_len; return e->pps; }

static int finish(sx_enc *e, int r, x264_nal_t *nal, int n, x264_picture_t *out) {
    if (r < 0) return -1;
    if (r == 0) { e->len = 0; return 0; }
    e->pts = out->i_pts;
    e->dts = out->i_dts;
    e->keyframe = out->b_keyframe;
    e->frame_type = out->i_type;
    e->psnr_y = out->prop.f_psnr[0];
    e->ssim = out->prop.f_ssim;
    return collect(e, nal, n);
}

/* Returns the number of bytes now in the output buffer (0 = frame delayed
 * inside the encoder), or -1 on error. */
int sx_encode(sx_enc *e, const uint8_t *y, const uint8_t *u, const uint8_t *v,
              int stride_y, int stride_c, int64_t pts, int force_idr) {
    x264_picture_t pic, out;
    x264_picture_init(&pic);
    pic.img.i_csp = X264_CSP_I420;
    pic.img.i_plane = 3;
    pic.img.plane[0] = (uint8_t *)y;
    pic.img.plane[1] = (uint8_t *)u;
    pic.img.plane[2] = (uint8_t *)v;
    pic.img.i_stride[0] = stride_y;
    pic.img.i_stride[1] = stride_c;
    pic.img.i_stride[2] = stride_c;
    pic.i_pts = pts;
    pic.i_type = force_idr ? X264_TYPE_IDR : X264_TYPE_AUTO;
    x264_nal_t *nal;
    int n;
    int r = x264_encoder_encode(e->h, &nal, &n, &pic, &out);
    return finish(e, r, nal, n, &out);
}

/* Drains one delayed frame. Returns bytes, 0 when fully drained, -1 on error. */
int sx_flush(sx_enc *e) {
    if (x264_encoder_delayed_frames(e->h) <= 0) { e->len = 0; return 0; }
    x264_picture_t out;
    x264_nal_t *nal;
    int n;
    int r = x264_encoder_encode(e->h, &nal, &n, NULL, &out);
    return finish(e, r, nal, n, &out);
}

const uint8_t *sx_data(sx_enc *e) { return e->buf; }
int64_t sx_pts(sx_enc *e) { return e->pts; }
int64_t sx_dts(sx_enc *e) { return e->dts; }
int sx_keyframe(sx_enc *e) { return e->keyframe; }
int sx_frame_type(sx_enc *e) { return e->frame_type; }
double sx_psnr_y(sx_enc *e) { return e->psnr_y; }
double sx_ssim(sx_enc *e) { return e->ssim; }

void sx_close(sx_enc *e) {
    if (!e) return;
    if (e->h) x264_encoder_close(e->h);
    free(e->buf);
    free(e);
}
