/* Narrow C interface over SVT-AV1 (one encoder = one chunk), mirroring the
 * x264 shim: options as "key=value:..." strings, packets copied out. */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "EbSvtAv1Enc.h"
#include "EbSvtAv1ErrorCodes.h"

typedef struct {
    EbComponentType *h;
    EbSvtAv1EncConfiguration cfg;
    int width, height;
    uint8_t *hdr;
    size_t hdr_len;
    uint8_t *buf;
    size_t cap, len;
    int64_t pts;
    int keyframe;
    uint8_t *stats_in;
    uint8_t *stats_out;
    size_t stats_out_len;
} sv_enc;

/* Only warnings and errors reach stderr (the JS console in the browser). */
static void quiet_log(void *ctx, SvtAv1LogLevel level, const char *tag, const char *fmt, va_list ap) {
    (void)ctx;
    if (level > SVT_AV1_LOG_WARN) return;
    fprintf(stderr, "svt-av1 %s: ", tag ? tag : "");
    vfprintf(stderr, fmt, ap);
}

static int log_set = 0;

/* pass: 0 single pass, 1 first pass, 2 second pass (stats from pass 1). */
sv_enc *sv_open(int w, int h, int fps_num, int fps_den, const char *opts, int pass, const uint8_t *stats,
                size_t stats_len) {
    if (!log_set) {
        svt_av1_set_log_callback(quiet_log, NULL);
        log_set = 1;
    }
    sv_enc *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->width = w;
    e->height = h;
    if (svt_av1_enc_init_handle(&e->h, &e->cfg) != EB_ErrorNone) goto fail;
    EbSvtAv1EncConfiguration *c = &e->cfg;
    c->source_width = w;
    c->source_height = h;
    c->frame_rate_numerator = fps_num;
    c->frame_rate_denominator = fps_den;
    c->encoder_bit_depth = 8;
    if (opts && *opts) {
        char *s = strdup(opts), *save = NULL;
        for (char *kv = strtok_r(s, ":", &save); kv; kv = strtok_r(NULL, ":", &save)) {
            char *eq = strchr(kv, '=');
            if (!eq) continue;
            *eq = 0;
            if (svt_av1_enc_parse_parameter(c, kv, eq + 1) != EB_ErrorNone) {
                fprintf(stderr, "svt-av1: bad option %s=%s\n", kv, eq + 1);
                free(s);
                goto fail;
            }
        }
        free(s);
    }
    c->pass = pass;
    if (pass == 2) {
        e->stats_in = malloc(stats_len ? stats_len : 1);
        memcpy(e->stats_in, stats, stats_len);
        c->rc_stats_buffer.buf = e->stats_in;
        c->rc_stats_buffer.sz = stats_len;
    }
    if (svt_av1_enc_set_parameter(e->h, c) != EB_ErrorNone) goto fail;
    if (svt_av1_enc_init(e->h) != EB_ErrorNone) goto fail;
    EbBufferHeaderType *hdr = NULL;
    if (svt_av1_enc_stream_header(e->h, &hdr) == EB_ErrorNone && hdr) {
        e->hdr_len = hdr->n_filled_len;
        e->hdr = malloc(e->hdr_len ? e->hdr_len : 1);
        memcpy(e->hdr, hdr->p_buffer, e->hdr_len);
        svt_av1_enc_stream_header_release(hdr);
    }
    return e;
fail:
    if (e->h) svt_av1_enc_deinit_handle(e->h);
    free(e->stats_in);
    free(e);
    return NULL;
}

/* Sequence header OBU(s), for the MP4 av1C box. */
const uint8_t *sv_header(sv_enc *e, int *len) {
    *len = (int)e->hdr_len;
    return e->hdr;
}

int sv_send(sv_enc *e, const uint8_t *y, const uint8_t *u, const uint8_t *v, int stride_y, int stride_c,
            int64_t pts) {
    EbSvtIOFormat io = {(uint8_t *)y, (uint8_t *)u, (uint8_t *)v, (uint32_t)stride_y, (uint32_t)stride_c,
                        (uint32_t)stride_c};
    EbBufferHeaderType in;
    memset(&in, 0, sizeof(in));
    in.size = sizeof(in);
    in.p_buffer = (uint8_t *)&io;
    in.n_filled_len = (uint32_t)(stride_y * e->height + 2 * stride_c * ((e->height + 1) / 2));
    in.pts = pts;
    in.pic_type = EB_AV1_INVALID_PICTURE;
    return svt_av1_enc_send_picture(e->h, &in) == EB_ErrorNone ? 0 : -1;
}

int sv_eos(sv_enc *e) {
    EbBufferHeaderType in;
    memset(&in, 0, sizeof(in));
    in.size = sizeof(in);
    in.flags = EB_BUFFERFLAG_EOS;
    in.pic_type = EB_AV1_INVALID_PICTURE;
    return svt_av1_enc_send_picture(e->h, &in) == EB_ErrorNone ? 0 : -1;
}

/* Returns packet bytes (> 0), 0 when nothing is ready (only when not
 * blocking), -2 after the end of stream, -1 on error. With `block`, waits
 * for the next packet (use after sv_eos). */
int sv_receive(sv_enc *e, int block) {
    EbBufferHeaderType *out = NULL;
    EbErrorType r = svt_av1_enc_get_packet(e->h, &out, (uint8_t)(block ? 1 : 0));
    if (r == EB_NoErrorEmptyQueue) return 0;
    if (r != EB_ErrorNone || !out) return -1;
    int eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
    e->len = out->n_filled_len;
    if (e->len > e->cap) {
        size_t cap = e->cap ? e->cap : 65536;
        while (cap < e->len) cap *= 2;
        uint8_t *b = realloc(e->buf, cap);
        if (!b) {
            svt_av1_enc_release_out_buffer(&out);
            return -1;
        }
        e->buf = b;
        e->cap = cap;
    }
    if (e->len) memcpy(e->buf, out->p_buffer, e->len);
    e->pts = out->pts;
    e->keyframe = out->pic_type == EB_AV1_KEY_PICTURE;
    svt_av1_enc_release_out_buffer(&out);
    if (e->len) return (int)e->len;
    return eos ? -2 : 0;
}

const uint8_t *sv_data(sv_enc *e) { return e->buf; }
int64_t sv_pts(sv_enc *e) { return e->pts; }
int sv_keyframe(sv_enc *e) { return e->keyframe; }

/* First-pass statistics, valid after the stream has ended. */
const uint8_t *sv_stats(sv_enc *e, size_t *len) {
    if (!e->stats_out) {
        SvtAv1FixedBuf fb = {0};
        if (svt_av1_enc_get_stream_info(e->h, SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT, &fb) == EB_ErrorNone &&
            fb.sz) {
            e->stats_out = malloc(fb.sz);
            memcpy(e->stats_out, fb.buf, fb.sz);
            e->stats_out_len = fb.sz;
        }
    }
    *len = e->stats_out_len;
    return e->stats_out;
}

void sv_close(sv_enc *e) {
    if (!e) return;
    if (e->h) {
        svt_av1_enc_deinit(e->h);
        svt_av1_enc_deinit_handle(e->h);
    }
    free(e->hdr);
    free(e->buf);
    free(e->stats_in);
    free(e->stats_out);
    free(e);
}
