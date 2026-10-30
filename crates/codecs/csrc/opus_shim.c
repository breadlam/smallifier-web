/* Non-variadic wrappers over the opus_encoder_ctl() macros. */
#include <stddef.h>
#include <stdint.h>
#include "opus.h"

OpusEncoder *so_open(int sample_rate, int channels, int bitrate, int *err) {
    OpusEncoder *e = opus_encoder_create(sample_rate, channels, OPUS_APPLICATION_AUDIO, err);
    if (!e) return NULL;
    opus_encoder_ctl(e, OPUS_SET_BITRATE(bitrate));
    opus_encoder_ctl(e, OPUS_SET_VBR(1));
    opus_encoder_ctl(e, OPUS_SET_VBR_CONSTRAINT(0));
    opus_encoder_ctl(e, OPUS_SET_COMPLEXITY(10));
    return e;
}

int so_lookahead(OpusEncoder *e) {
    opus_int32 v = 0;
    opus_encoder_ctl(e, OPUS_GET_LOOKAHEAD(&v));
    return v;
}

int so_encode(OpusEncoder *e, const float *pcm, int frame_size, uint8_t *out, int cap) {
    return opus_encode_float(e, pcm, frame_size, out, cap);
}

void so_close(OpusEncoder *e) { opus_encoder_destroy(e); }
