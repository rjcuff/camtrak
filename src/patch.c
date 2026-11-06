/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
#include "patch.h"

#include <math.h>
#include <stddef.h>

static const double TT_PI = 3.14159265358979323846;

int tt_frame_valid(const camtrak_frame *f)
{
    if (!f || !f->data || f->width <= 0 || f->height <= 0) {
        return 0;
    }
    if (f->format != CAMTRAK_FORMAT_GRAY8 && f->format != CAMTRAK_FORMAT_BGR8) {
        return 0;
    }
    return f->stride >= f->width * (int)f->format;
}

void tt_make_log_lut(tt_log_lut *lut)
{
    int i;
    for (i = 0; i < 256; i++) {
        lut->v[i] = (float)log(1.0 + i);
    }
    /* luma, integer weights summing to 256 (0.114 B + 0.587 G + 0.299 R) */
    lut->wb = 29;
    lut->wg = 150;
    lut->wr = 77;
    lut->off = 0;
}

/* log(1 + gray) of integer pixel (x, y), clamped to the image. */
static inline float logval_at(const camtrak_frame *f, const tt_log_lut *lut, int x, int y)
{
    const uint8_t *p;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= f->width)  x = f->width - 1;
    if (y >= f->height) y = f->height - 1;
    p = f->data + (size_t)y * (size_t)f->stride + (size_t)x * (size_t)f->format;
    if (f->format == CAMTRAK_FORMAT_GRAY8) {
        return lut->v[p[0]];
    }
    {
        /* BGR -> the tracker's channel (luma by default, see tt_choose_channel) */
        int v = lut->wb * p[0] + lut->wg * p[1] + lut->wr * p[2] + lut->off;
        v = v < 0 ? 0 : v >> 8;
        return lut->v[v > 255 ? 255 : v];
    }
}

/* Bilinear sample at continuous point (px, py). Pixel centers sit at i + 0.5. */
static inline float sample_bilinear(const camtrak_frame *f, const tt_log_lut *lut,
                                    float px, float py)
{
    float fx = px - 0.5f;
    float fy = py - 0.5f;
    int x0, y0;
    float ax, ay, top, bot;

    /* Keep far-away / non-finite coordinates from overflowing the int cast. */
    if (!(fx > -1.0f)) fx = -1.0f;
    if (!(fy > -1.0f)) fy = -1.0f;
    if (fx > (float)f->width)  fx = (float)f->width;
    if (fy > (float)f->height) fy = (float)f->height;

    x0 = (int)floorf(fx);
    y0 = (int)floorf(fy);
    ax = fx - (float)x0;
    ay = fy - (float)y0;

    top = (1.0f - ax) * logval_at(f, lut, x0, y0)     + ax * logval_at(f, lut, x0 + 1, y0);
    bot = (1.0f - ax) * logval_at(f, lut, x0, y0 + 1) + ax * logval_at(f, lut, x0 + 1, y0 + 1);
    return (1.0f - ay) * top + ay * bot;
}

void tt_sample_patch(const camtrak_frame *f, const tt_log_lut *lut, float cx, float cy,
                     float step, float angle, int w, int h, float *out)
{
    const float ca = cosf(angle) * step;
    const float sa = sinf(angle) * step;
    int u, v;

    for (v = 0; v < h; v++) {
        const float oy = (float)(v - h / 2);
        for (u = 0; u < w; u++) {
            const float ox = (float)(u - w / 2);
            const float px = cx + ox * ca - oy * sa;
            const float py = cy + ox * sa + oy * ca;
            out[v * w + u] = sample_bilinear(f, lut, px, py);
        }
    }
}

void tt_make_hann(int w, int h, float *win)
{
    int x, y;
    for (y = 0; y < h; y++) {
        const double wy = 0.5 * (1.0 - cos(2.0 * TT_PI * (y + 0.5) / h));
        for (x = 0; x < w; x++) {
            const double wx = 0.5 * (1.0 - cos(2.0 * TT_PI * (x + 0.5) / w));
            win[y * w + x] = (float)(wx * wy);
        }
    }
}

void tt_preprocess(float *p, const float *win, int n)
{
    double sum = 0.0, sq = 0.0, wsum = 0.0, mean, std;
    float inv;
    int i;

    /* Mean and spread are weighted by the cosine window, i.e. measured mostly
     * on the target in the middle. Unweighted, a slightly bigger search region
     * (more plain background) would lower the spread, boost the target after
     * normalizing, and bias the scale search towards ever-bigger boxes. */
    for (i = 0; i < n; i++) {
        sum += (double)win[i] * p[i];
        wsum += win[i];
    }
    mean = sum / wsum;
    for (i = 0; i < n; i++) {
        const double d = p[i] - mean;
        sq += win[i] * d * d;
    }
    std = sqrt(sq / wsum);
    /* A perfectly flat patch has no information; make it all zeros. */
    inv = std > 1e-5 ? (float)(1.0 / std) : 0.0f;
    for (i = 0; i < n; i++) {
        p[i] = (p[i] - (float)mean) * inv * win[i];
    }
}

/* ------------------------------------------------------------ channel choice */

typedef struct {
    double n, m[3], c[6];    /* count, sum, sum of products (xx xy xz yy yz zz) */
} tt_stats3;

static void stats_add(tt_stats3 *s, const uint8_t *p)
{
    const double b = p[0], g = p[1], r = p[2];
    s->n += 1.0;
    s->m[0] += b; s->m[1] += g; s->m[2] += r;
    s->c[0] += b * b; s->c[1] += b * g; s->c[2] += b * r;
    s->c[3] += g * g; s->c[4] += g * r; s->c[5] += r * r;
}

/* mean (3) and covariance (3x3) */
static void stats_finish(const tt_stats3 *s, double mu[3], double cov[3][3])
{
    int i, j, k = 0;
    for (i = 0; i < 3; i++) {
        mu[i] = s->m[i] / s->n;
    }
    for (i = 0; i < 3; i++) {
        for (j = i; j < 3; j++, k++) {
            cov[i][j] = cov[j][i] = s->c[k] / s->n - mu[i] * mu[j];
        }
    }
}

static const uint8_t *pixel(const camtrak_frame *f, int x, int y)
{
    return f->data + (size_t)y * (size_t)f->stride + (size_t)x * 3u;
}

/* separability of projection w: (w.d)^2 / (w' S w) */
static double fisher(const double w[3], const double d[3], double S[3][3])
{
    double num = w[0] * d[0] + w[1] * d[1] + w[2] * d[2], den = 0.0;
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            den += w[i] * S[i][j] * w[j];
        }
    }
    return den > 1e-9 ? num * num / den : 0.0;
}

float tt_choose_channel(const camtrak_frame *f, float cx, float cy, float w, float h,
                        tt_log_lut *lut)
{
    /* blends tried: share of the discriminant direction (0 = pure luma) */
    static const double k_mix[] = { 0.0, 0.35, 0.7 };
    /* a color projection must separate this much better than luma to be used */
    const double k_gain_needed = 1.5;
    const double luma[3] = { 0.114, 0.587, 0.299 };
    tt_stats3 st = { 0 }, sb = { 0 };
    double mt[3], mb[3], ct[3][3], cb[3][3], S[3][3], d[3], inv[3][3], lda[3], det;
    double best_w[3], best_j, j_luma, nl, nd, lo = 1e9, hi = -1e9, scale;
    int i, j, k, n = 24;

    tt_make_log_lut(lut);
    if (f->format != CAMTRAK_FORMAT_BGR8 || w < 2.0f || h < 2.0f) {
        return 1.0f;
    }
    /* target: central 70% of the box; background: a ring from 1.3x to 2.2x the box */
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            const int x = (int)(cx + ((i + 0.5f) / n - 0.5f) * 0.7f * w);
            const int y = (int)(cy + ((j + 0.5f) / n - 0.5f) * 0.7f * h);
            if (x >= 0 && y >= 0 && x < f->width && y < f->height) {
                stats_add(&st, pixel(f, x, y));
            }
        }
    }
    n = 40;
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            const float u = ((i + 0.5f) / n - 0.5f) * 2.2f, v = ((j + 0.5f) / n - 0.5f) * 2.2f;
            const int x = (int)(cx + u * w), y = (int)(cy + v * h);
            if (fabsf(u) < 0.65f && fabsf(v) < 0.65f) {
                continue;               /* inside 1.3x the box: target or its edge */
            }
            if (x >= 0 && y >= 0 && x < f->width && y < f->height) {
                stats_add(&sb, pixel(f, x, y));
            }
        }
    }
    if (st.n < 16 || sb.n < 64) {
        return 1.0f;
    }
    stats_finish(&st, mt, ct);
    stats_finish(&sb, mb, cb);
    for (i = 0; i < 3; i++) {
        d[i] = mt[i] - mb[i];
        for (j = 0; j < 3; j++) {
            S[i][j] = ct[i][j] + cb[i][j] + (i == j ? 16.0 : 0.0);   /* + sensor noise */
        }
    }
    /* lda = S^-1 d (3x3 inverse by cofactors) */
    inv[0][0] = S[1][1] * S[2][2] - S[1][2] * S[2][1];
    inv[0][1] = S[0][2] * S[2][1] - S[0][1] * S[2][2];
    inv[0][2] = S[0][1] * S[1][2] - S[0][2] * S[1][1];
    inv[1][0] = S[1][2] * S[2][0] - S[1][0] * S[2][2];
    inv[1][1] = S[0][0] * S[2][2] - S[0][2] * S[2][0];
    inv[1][2] = S[0][2] * S[1][0] - S[0][0] * S[1][2];
    inv[2][0] = S[1][0] * S[2][1] - S[1][1] * S[2][0];
    inv[2][1] = S[0][1] * S[2][0] - S[0][0] * S[2][1];
    inv[2][2] = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    det = S[0][0] * inv[0][0] + S[0][1] * inv[1][0] + S[0][2] * inv[2][0];
    if (fabs(det) < 1e-9) {
        return 1.0f;
    }
    for (i = 0; i < 3; i++) {
        lda[i] = (inv[i][0] * d[0] + inv[i][1] * d[1] + inv[i][2] * d[2]) / det;
    }
    nl = sqrt(luma[0] * luma[0] + luma[1] * luma[1] + luma[2] * luma[2]);
    nd = sqrt(lda[0] * lda[0] + lda[1] * lda[1] + lda[2] * lda[2]);
    if (nd < 1e-12) {
        return 1.0f;
    }
    /* point the discriminant the same way as luma, so blends do not cancel */
    if (lda[0] * luma[0] + lda[1] * luma[1] + lda[2] * luma[2] < 0.0) {
        nd = -nd;
    }
    j_luma = fisher(luma, d, S);
    best_j = j_luma;
    for (i = 0; i < 3; i++) {
        best_w[i] = luma[i];
    }
    for (k = 1; k < (int)(sizeof(k_mix) / sizeof(k_mix[0])); k++) {
        double wk[3], jk;
        for (i = 0; i < 3; i++) {
            wk[i] = (1.0 - k_mix[k]) * luma[i] / nl + k_mix[k] * lda[i] / nd;
        }
        jk = fisher(wk, d, S);
        if (jk > best_j * 1.05 && jk > j_luma * k_gain_needed) {
            best_j = jk;
            for (i = 0; i < 3; i++) {
                best_w[i] = wk[i];
            }
        }
    }
    if (best_j <= j_luma) {
        return 1.0f;                    /* luma is as good: keep it */
    }
    /* stretch the projection over the colors actually seen (+50% margin) */
    {
        const double cands[2][3] = { { mt[0], mt[1], mt[2] }, { mb[0], mb[1], mb[2] } };
        const double *covs[2] = { &ct[0][0], &cb[0][0] };
        for (k = 0; k < 2; k++) {
            double m = 0.0, var = 0.0;
            for (i = 0; i < 3; i++) {
                m += best_w[i] * cands[k][i];
                for (j = 0; j < 3; j++) {
                    var += best_w[i] * covs[k][i * 3 + j] * best_w[j];
                }
            }
            lo = fmin(lo, m - 3.0 * sqrt(fmax(var, 0.0)));
            hi = fmax(hi, m + 3.0 * sqrt(fmax(var, 0.0)));
        }
    }
    {
        const double span = fmax(hi - lo, 40.0), mid = 0.5 * (hi + lo);
        lo = mid - 0.75 * span;
        hi = mid + 0.75 * span;
    }
    scale = 255.0 / (hi - lo);
    lut->wb = (int)lround(256.0 * best_w[0] * scale);
    lut->wg = (int)lround(256.0 * best_w[1] * scale);
    lut->wr = (int)lround(256.0 * best_w[2] * scale);
    lut->off = (int)lround(-256.0 * lo * scale);
    return (float)(best_j / fmax(j_luma, 1e-9));
}
