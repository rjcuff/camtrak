/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
#include "kalman.h"

/* Velocity is unknown at lock: start with a large variance (std 5 px/frame). */
#define TT_KF_INIT_POS_VAR 1.0f
#define TT_KF_INIT_VEL_VAR 25.0f

static void axis_init(tt_kf_axis *a, float pos)
{
    a->p = pos;
    a->v = 0.0f;
    a->P00 = TT_KF_INIT_POS_VAR;
    a->P01 = 0.0f;
    a->P11 = TT_KF_INIT_VEL_VAR;
}

/*
 * Transition F = [[1, 1], [0, 1]] (dt = 1 frame).  P' = F P F^T + Q, where
 * Q = q * [[1/4, 1/2], [1/2, 1]] is the covariance of a random acceleration
 * kick of variance q applied during the frame.
 */
static void axis_predict(tt_kf_axis *a, float q)
{
    const float P00 = a->P00, P01 = a->P01, P11 = a->P11;
    a->p += a->v;
    a->P00 = P00 + 2.0f * P01 + P11 + 0.25f * q;
    a->P01 = P01 + P11 + 0.5f * q;
    a->P11 = P11 + q;
}

/* Measurement H = [1, 0]: we observe position only. */
static void axis_correct(tt_kf_axis *a, float z, float r)
{
    const float S  = a->P00 + r;          /* innovation variance */
    const float K0 = a->P00 / S;          /* gain for position */
    const float K1 = a->P01 / S;          /* gain for velocity */
    const float innov = z - a->p;         /* how far the measurement is from the prediction */
    const float P00 = a->P00, P01 = a->P01;

    a->p += K0 * innov;
    a->v += K1 * innov;
    a->P00 = (1.0f - K0) * P00;
    a->P01 = (1.0f - K0) * P01;
    a->P11 = a->P11 - K1 * P01;
}

void tt_kalman_init(tt_kalman *k, float x, float y, float q, float r)
{
    axis_init(&k->x, x);
    axis_init(&k->y, y);
    k->q = q;
    k->r = r;
}

void tt_kalman_predict(tt_kalman *k)
{
    axis_predict(&k->x, k->q);
    axis_predict(&k->y, k->q);
}

void tt_kalman_correct(tt_kalman *k, float zx, float zy)
{
    axis_correct(&k->x, zx, k->r);
    axis_correct(&k->y, zy, k->r);
}

void tt_kalman_damp(tt_kalman *k, float factor)
{
    k->x.v *= factor;
    k->y.v *= factor;
}
