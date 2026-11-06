/*
 * patch.h - cutting a small grayscale patch out of a frame and preparing it
 * for the correlation filter.
 */
#ifndef TT_PATCH_H
#define TT_PATCH_H

#include "tether/tether.h"

/* 1 if the frame is usable (non-null data, positive size, sane stride/format). */
int tt_frame_valid(const tether_frame *f);

/* log(1 + v) for every pixel value v: filled by tt_make_log_lut. */
typedef struct {
    float v[256];
} tt_log_lut;

void tt_make_log_lut(tt_log_lut *lut);

/*
 * Sample a w x h patch of log(1 + gray) values centred on (cx, cy).
 * Taking the log of each source pixel through a 256-entry table, then
 * interpolating, is much cheaper than calling logf on every patch pixel.
 *
 * Patch pixel (u, v) is taken from the image point
 *     (cx, cy) + R(angle) * (u - w/2, v - h/2) * step
 * where `step` is image pixels per patch pixel (step > 1 shrinks a big region
 * into a small patch, step < 1 magnifies a tiny target). Bilinear
 * interpolation; outside the image the border pixels are repeated. BGR frames
 * are converted to gray on the fly, so no full-frame buffer is needed.
 */
void tt_sample_patch(const tether_frame *f, const tt_log_lut *lut, float cx, float cy,
                     float step, float angle, int w, int h, float *out);

/* Separable Hann (cosine) window, w*h values in [0, 1]. */
void tt_make_hann(int w, int h, float *win);

/*
 * MOSSE preprocessing, in place, on a patch that is already log(1 + gray)
 * (the log compresses bright highlights so lighting changes matter less):
 *   1. zero mean, unit variance  - removes brightness and contrast
 *   2. multiply by the cosine window  - fades the patch edges to 0 so the FFT,
 *      which treats the patch as periodic, does not see a hard seam at the borders
 */
void tt_preprocess(float *patch, const float *win, int n);

#endif
