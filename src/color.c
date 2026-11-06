/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
#include "color.h"

#include <math.h>
#include <string.h>

/*
 * Only the middle of the box is described: it is mostly the target itself,
 * while the edges are mostly background (snow), which would make every box
 * look alike. Fractions of box width / height.
 */
#define TT_COLOR_INNER_W 0.5f
#define TT_COLOR_INNER_H 0.6f
/* Surroundings for the "stands out" test: a region this many box sizes
 * across, minus the box. */
#define TT_COLOR_RING 2.0f
/* A candidate may blend into its surroundings this much more than the
 * selection did, and never more than BLEND_MAX (1 = indistinguishable). */
#define TT_COLOR_BLEND_SLACK 0.25f
#define TT_COLOR_BLEND_MAX 0.85f
/* Pixels sampled per box at most (subsampled on a grid): keeps it ~free. */
#define TT_COLOR_SAMPLES 1500.0f

/* 4 levels per channel: coarse on purpose, so lighting changes between
 * shots move colors less often into another bin. */
static int bin_of(const uint8_t *p)
{
    return ((p[0] >> 6) << 4) | ((p[1] >> 6) << 2) | (p[2] >> 6);
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/*
 * Normalized color histogram of the rectangle centered on the box center with
 * size (fw * box.w, fh * box.h), leaving out the box itself if `ring` is set.
 * Returns 0 if no pixel was inside the frame.
 */
static int region_hist(const camtrak_frame *f, camtrak_box box, float fw, float fh, int ring,
                       float *hist)
{
    const float iw = box.w * fw, ih = box.h * fh;
    const float cx = box.x + 0.5f * box.w, cy = box.y + 0.5f * box.h;
    const int ix0 = clampi((int)(cx - 0.5f * iw), 0, f->width);
    const int ix1 = clampi((int)(cx + 0.5f * iw), 0, f->width);
    const int iy0 = clampi((int)(cy - 0.5f * ih), 0, f->height);
    const int iy1 = clampi((int)(cy + 0.5f * ih), 0, f->height);
    const float s = sqrtf(fmaxf(1.0f, iw * ih) / TT_COLOR_SAMPLES);
    const int step = s < 1.0f ? 1 : (int)s;
    float n = 0.0f;
    int x, y, i;

    memset(hist, 0, sizeof(float) * TT_COLOR_BINS);
    for (y = iy0; y < iy1; y += step) {
        const uint8_t *row = f->data + (size_t)y * (size_t)f->stride;
        const int in_rows = y >= box.y && y < box.y + box.h;
        for (x = ix0; x < ix1; x += step) {
            if (ring && in_rows && x >= box.x && x < box.x + box.w) {
                continue;
            }
            hist[bin_of(row + 3 * x)] += 1.0f;
            n += 1.0f;
        }
    }
    if (n < 1.0f) {
        return 0;
    }
    for (i = 0; i < TT_COLOR_BINS; i++) {
        hist[i] /= n;
    }
    return 1;
}

static int inner_hist(const camtrak_frame *f, camtrak_box box, float *hist)
{
    return region_hist(f, box, TT_COLOR_INNER_W, TT_COLOR_INNER_H, 0, hist);
}

/* Bhattacharyya coefficient: 1 = identical color distributions, 0 = disjoint. */
static float bhattacharyya(const float *a, const float *b)
{
    float bc = 0.0f;
    int i;
    for (i = 0; i < TT_COLOR_BINS; i++) {
        bc += sqrtf(a[i] * b[i]);
    }
    return bc;
}

float tt_color_contrast(const camtrak_frame *f, camtrak_box box)
{
    float inner[TT_COLOR_BINS], ring[TT_COLOR_BINS];
    if (f->format != CAMTRAK_FORMAT_BGR8 || !inner_hist(f, box, inner)
        || !region_hist(f, box, TT_COLOR_RING, TT_COLOR_RING, 1, ring)) {
        return -1.0f;
    }
    return bhattacharyya(inner, ring);
}

void tt_color_learn(tt_color *c, const camtrak_frame *f, camtrak_box box)
{
    memset(c, 0, sizeof(*c));
    if (f->format == CAMTRAK_FORMAT_BGR8 && inner_hist(f, box, c->ref)) {
        c->enabled = 1;
        c->ref_contrast = tt_color_contrast(f, box);
    }
}

float tt_color_match(const tt_color *c, const camtrak_frame *f, camtrak_box box)
{
    float hist[TT_COLOR_BINS];

    if (!c->enabled || f->format != CAMTRAK_FORMAT_BGR8) {
        return -1.0f;
    }
    if (!inner_hist(f, box, hist)) {
        return 0.0f;
    }
    return bhattacharyya(hist, c->ref);
}

int tt_color_stands_out(const tt_color *c, const camtrak_frame *f, camtrak_box box)
{
    float s;
    if (!c->enabled || f->format != CAMTRAK_FORMAT_BGR8) {
        return 1;   /* no color: cannot judge, do not veto */
    }
    s = tt_color_contrast(f, box);
    return s >= 0.0f && s <= fminf(TT_COLOR_BLEND_MAX, c->ref_contrast + TT_COLOR_BLEND_SLACK);
}
