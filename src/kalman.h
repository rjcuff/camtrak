/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
/*
 * kalman.h - constant-velocity Kalman filter for the target center.
 *
 * The state per axis is position p and velocity v (pixels, pixels/frame).
 * x and y are independent, so we run two tiny 2-state filters instead of
 * one 4x4: same result, much easier to read.
 *
 * Each frame:
 *   predict:  p += v                      (assume velocity stays the same)
 *             uncertainty P grows by q    (it might not have)
 *   correct:  blend in a measurement z with weight K = P / (P + r):
 *             trust the measurement more when our prediction is uncertain
 *             (P large) or the measurement is precise (r small).
 * Without a measurement (occlusion) we only predict: that is "coasting".
 */
#ifndef TT_KALMAN_H
#define TT_KALMAN_H

typedef struct {
    float p, v;                 /* state: position, velocity */
    float P00, P01, P11;        /* covariance [[P00, P01], [P01, P11]] */
} tt_kf_axis;

typedef struct {
    tt_kf_axis x, y;
    float q;                    /* process noise: how much acceleration we expect */
    float r;                    /* measurement noise variance, px^2 */
} tt_kalman;

void tt_kalman_init(tt_kalman *k, float x, float y, float q, float r);
void tt_kalman_predict(tt_kalman *k);
void tt_kalman_correct(tt_kalman *k, float zx, float zy);
/* Multiply velocities by `factor` (used to slow a coasting prediction down). */
void tt_kalman_damp(tt_kalman *k, float factor);

#endif
