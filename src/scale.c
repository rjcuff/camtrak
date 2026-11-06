#include "scale.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const double TT_PI = 3.14159265358979323846;

/* Model patch area (pixels). Small is fine: only the overall shape matters for size. */
#define TT_SCALE_MODEL_AREA 256.0f
/* Width of the desired Gaussian along the scale axis, in samples. */
#define TT_SCALE_SIGMA 1.0f
/* Regulariser relative to the average spectral energy, as in mosse.c. */
#define TT_SCALE_LAMBDA_REL 0.01f

int tt_scale_init(tt_scale *s, float target_w, float target_h, int S, float step, float rate)
{
    const float k = sqrtf(TT_SCALE_MODEL_AREA / (target_w * target_h));
    const int c = S / 2;
    size_t DS;
    int n, kk;

    memset(s, 0, sizeof(*s));
    if (S < 3 || S > 63 || S % 2 == 0 || step <= 1.0f) {
        return -1;
    }
    s->S = S;
    s->step = step;
    s->rate = rate;
    /* same aspect ratio as the target, about TT_SCALE_MODEL_AREA pixels */
    s->mw = (int)fmaxf(4.0f, floorf(target_w * k + 0.5f));
    s->mh = (int)fmaxf(4.0f, floorf(target_h * k + 0.5f));
    s->D = s->mw * s->mh;
    DS = (size_t)s->D * (size_t)S;

    s->factors = malloc(sizeof(float) * S);
    s->win     = malloc(sizeof(float) * S);
    s->Gr      = malloc(sizeof(float) * S);
    s->Gi      = malloc(sizeof(float) * S);
    s->dft_c   = malloc(sizeof(float) * S * S);
    s->dft_s   = malloc(sizeof(float) * S * S);
    s->B       = calloc((size_t)S, sizeof(float));   /* calloc: training blends into these */
    s->resp    = malloc(sizeof(float) * S);
    s->X       = malloc(sizeof(float) * DS);
    s->Fr      = malloc(sizeof(float) * DS);
    s->Fi      = malloc(sizeof(float) * DS);
    s->Ar      = calloc(DS, sizeof(float));
    s->Ai      = calloc(DS, sizeof(float));
    s->patch   = malloc(sizeof(float) * (size_t)s->D);
    if (!s->factors || !s->win || !s->Gr || !s->Gi || !s->dft_c || !s->dft_s || !s->B
        || !s->resp || !s->X || !s->Fr || !s->Fi || !s->Ar || !s->Ai || !s->patch) {
        tt_scale_free(s);
        return -1;
    }

    for (n = 0; n < S; n++) {
        s->factors[n] = powf(step, (float)(n - c));
        s->win[n] = (float)(0.5 * (1.0 - cos(2.0 * TT_PI * (n + 0.5) / S)));
    }
    for (kk = 0; kk < S; kk++) {
        for (n = 0; n < S; n++) {
            const double a = 2.0 * TT_PI * (double)kk * n / S;
            s->dft_c[kk * S + n] = (float)cos(a);
            s->dft_s[kk * S + n] = (float)sin(a);
        }
    }
    /* desired output: Gaussian peaked at the middle sample (= current size) */
    for (kk = 0; kk < S; kk++) {
        double re = 0.0, im = 0.0;
        for (n = 0; n < S; n++) {
            const double g = exp(-0.5 * (n - c) * (n - c) / (TT_SCALE_SIGMA * TT_SCALE_SIGMA));
            re += g * s->dft_c[kk * S + n];
            im -= g * s->dft_s[kk * S + n];
        }
        s->Gr[kk] = (float)re;
        s->Gi[kk] = (float)im;
    }
    return 0;
}

void tt_scale_free(tt_scale *s)
{
    free(s->factors); free(s->win); free(s->Gr); free(s->Gi);
    free(s->dft_c); free(s->dft_s); free(s->B); free(s->resp);
    free(s->X); free(s->Fr); free(s->Fi); free(s->Ar); free(s->Ai); free(s->patch);
    memset(s, 0, sizeof(*s));
}

/* Cut the box at all S sizes into X (one row per pixel), each cut normalised. */
static void sample_scales(tt_scale *s, const tether_frame *f, const tt_log_lut *lut,
                          float cx, float cy, float w, float h)
{
    const int S = s->S, D = s->D;
    int n, d;
    for (n = 0; n < S; n++) {
        /* image pixels per model pixel, averaged over both axes */
        const float step = 0.5f * (w / (float)s->mw + h / (float)s->mh) * s->factors[n];
        double sum = 0.0, sq = 0.0, mean, std;
        float inv;
        tt_sample_patch(f, lut, cx, cy, step, 0.0f, s->mw, s->mh, s->patch);
        for (d = 0; d < D; d++) {
            sum += s->patch[d];
        }
        mean = sum / D;
        for (d = 0; d < D; d++) {
            const double v = s->patch[d] - mean;
            sq += v * v;
        }
        std = sqrt(sq / D);
        inv = std > 1e-5 ? (float)(1.0 / std) : 0.0f;
        for (d = 0; d < D; d++) {
            s->X[(size_t)d * S + n] = (s->patch[d] - (float)mean) * inv * s->win[n];
        }
    }
}

/* F = DFT of every row of X along the scale axis. */
static void dft_rows(tt_scale *s)
{
    const int S = s->S;
    int d, k, n;
    for (d = 0; d < s->D; d++) {
        const float *x = s->X + (size_t)d * S;
        float *fr = s->Fr + (size_t)d * S, *fi = s->Fi + (size_t)d * S;
        for (k = 0; k < S; k++) {
            float re = 0.0f, im = 0.0f;
            for (n = 0; n < S; n++) {
                re += x[n] * s->dft_c[k * S + n];
                im -= x[n] * s->dft_s[k * S + n];
            }
            fr[k] = re;
            fi[k] = im;
        }
    }
}

void tt_scale_train(tt_scale *s, const tether_frame *f, const tt_log_lut *lut,
                    float cx, float cy, float w, float h)
{
    const int S = s->S;
    const float keep = s->trained ? 1.0f - s->rate : 0.0f;
    const float add  = s->trained ? s->rate : 1.0f;
    int d, k;

    sample_scales(s, f, lut, cx, cy, w, h);
    dft_rows(s);

    for (k = 0; k < S; k++) {
        s->B[k] *= keep;
    }
    for (d = 0; d < s->D; d++) {
        const size_t o = (size_t)d * S;
        for (k = 0; k < S; k++) {
            const float fr = s->Fr[o + k], fi = s->Fi[o + k];
            /* A = G * conj(F), same convention as mosse.c */
            s->Ar[o + k] = keep * s->Ar[o + k] + add * (s->Gr[k] * fr + s->Gi[k] * fi);
            s->Ai[o + k] = keep * s->Ai[o + k] + add * (s->Gi[k] * fr - s->Gr[k] * fi);
            s->B[k] += add * (fr * fr + fi * fi);
        }
    }
    if (!s->trained) {
        double mean_b = 0.0;
        for (k = 0; k < S; k++) {
            mean_b += s->B[k];
        }
        s->lambda = (float)(TT_SCALE_LAMBDA_REL * mean_b / S) + 1e-6f;
        s->trained = 1;
    }
}

float tt_scale_detect(tt_scale *s, const tether_frame *f, const tt_log_lut *lut,
                      float cx, float cy, float w, float h)
{
    const int S = s->S, c = S / 2;
    float Yr[64], Yi[64];   /* S <= 63 enforced by the caller */
    int d, k, n, best = 0;
    float off = 0.0f;

    if (!s->trained) {
        return 1.0f;
    }
    sample_scales(s, f, lut, cx, cy, w, h);
    dft_rows(s);

    /* Y = sum_d A_d . Z_d / (B + lambda) */
    for (k = 0; k < S; k++) {
        Yr[k] = Yi[k] = 0.0f;
    }
    for (d = 0; d < s->D; d++) {
        const size_t o = (size_t)d * S;
        for (k = 0; k < S; k++) {
            const float ar = s->Ar[o + k], ai = s->Ai[o + k];
            const float zr = s->Fr[o + k], zi = s->Fi[o + k];
            Yr[k] += ar * zr - ai * zi;
            Yi[k] += ar * zi + ai * zr;
        }
    }
    /* inverse DFT, real part only */
    for (n = 0; n < S; n++) {
        float re = 0.0f;
        for (k = 0; k < S; k++) {
            const float inv = 1.0f / (s->B[k] + s->lambda);
            re += (Yr[k] * s->dft_c[k * S + n] - Yi[k] * s->dft_s[k * S + n]) * inv;
        }
        s->resp[n] = re;
        if (re > s->resp[best]) {
            best = n;
        }
    }
    /* sub-sample refinement with a parabola, like the translation peak */
    if (best > 0 && best < S - 1) {
        const float l = s->resp[best - 1], m = s->resp[best], r = s->resp[best + 1];
        const float den = l - 2.0f * m + r;
        if (den < 0.0f) {
            off = fmaxf(-0.5f, fminf(0.5f, 0.5f * (l - r) / den));
        }
    }
    return powf(s->step, (float)(best - c) + off);
}
