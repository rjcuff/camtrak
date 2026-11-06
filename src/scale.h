/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
/*
 * scale.h - DSST-style scale filter (Danelljan et al., BMVC 2014).
 *
 * The translation filter answers "where is the target?". This one answers
 * "how big is it now?", once the position is known:
 *
 *   1. Cut the target box (no padding, so nearby clutter such as a shadow is
 *      left out) at S sizes:  size * a^n  for n = -(S-1)/2 .. (S-1)/2,
 *      and resize every cut to the same small model size (mw x mh pixels).
 *   2. Treat those S cuts as a 1D signal along the scale axis, one per pixel:
 *      a D x S table (D = mw*mh). Learn a correlation filter along the scale
 *      axis, exactly like MOSSE but 1D, with a Gaussian peak at the middle
 *      (the current size) as the desired output.
 *   3. Next frame: correlate, and the peak's offset from the middle says how
 *      many steps of a the target grew or shrank.
 *
 * S is small (17), so a plain DFT (S x S multiply) is cheaper than an FFT.
 */
#ifndef TT_SCALE_H
#define TT_SCALE_H

#include "patch.h"

typedef struct {
    int    S;               /* number of scale samples (odd) */
    int    mw, mh, D;       /* model patch size, D = mw*mh */
    float  step;            /* ratio between neighboring scale samples (a) */
    float  rate;            /* learning rate */
    float  lambda;          /* regulariser, fixed after the first training */
    int    trained;
    float *factors;         /* S: a^(n - center) */
    float *win;             /* S: Hann window along the scale axis */
    float *Gr, *Gi;         /* S: DFT of the desired 1D Gaussian */
    float *dft_c, *dft_s;   /* S*S: cos/sin(2*pi*k*n/S) */
    float *X;               /* D*S: current samples, row d = one pixel across all scales */
    float *Fr, *Fi;         /* D*S: their DFT along the scale axis */
    float *Ar, *Ai;         /* D*S: numerator  avg(G . conj(F)) */
    float *B;               /* S:   denominator avg(sum_d |F_d|^2) */
    float *patch;           /* D: one resampled cut */
    float *resp;            /* S: response */
} tt_scale;

/* target_w/h: box size at lock (sets the model's aspect ratio). Returns 0 on success. */
int  tt_scale_init(tt_scale *s, float target_w, float target_h, int S, float step, float rate);
void tt_scale_free(tt_scale *s);

/* Learn the box (cx, cy, w, h) at all S sizes. First call initializes, later calls blend. */
void tt_scale_train(tt_scale *s, const camtrak_frame *f, const tt_log_lut *lut,
                    float cx, float cy, float w, float h);

/* Return the size change (multiply w and h by it) that best matches the model. */
float tt_scale_detect(tt_scale *s, const camtrak_frame *f, const tt_log_lut *lut,
                      float cx, float cy, float w, float h);

#endif
