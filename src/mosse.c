#include "mosse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int tt_mosse_init(tt_mosse *m, int w, int h, float sigma, float rate)
{
    const size_t n = (size_t)w * (size_t)h;
    int x, y;

    memset(m, 0, sizeof(*m));
    m->w = w;
    m->h = h;
    m->n = (int)n;
    m->rate = rate;

    if (tt_fft2d_init(&m->fft, w, h) != 0) {
        return -1;
    }
    m->G    = malloc(sizeof(float) * 2 * n);
    m->A    = malloc(sizeof(float) * 2 * n);
    m->B    = malloc(sizeof(float) * n);
    m->spec = malloc(sizeof(float) * 2 * n);
    m->spec2 = malloc(sizeof(float) * 2 * n);
    if (!m->G || !m->A || !m->B || !m->spec || !m->spec2) {
        tt_mosse_free(m);
        return -1;
    }

    /* Desired output: a Gaussian centred at patch pixel (w/2, h/2), which is
     * where tt_sample_patch puts the target centre. */
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            const float dx = (float)(x - w / 2);
            const float dy = (float)(y - h / 2);
            const size_t k = (size_t)y * (size_t)w + (size_t)x;
            m->G[2 * k]     = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            m->G[2 * k + 1] = 0.0f;
        }
    }
    tt_fft2d_forward(&m->fft, m->G);
    tt_mosse_reset(m);
    return 0;
}

void tt_mosse_free(tt_mosse *m)
{
    tt_fft2d_free(&m->fft);
    free(m->G);
    free(m->A);
    free(m->B);
    free(m->spec);
    free(m->spec2);
    memset(m, 0, sizeof(*m));
}

/* spec = FFT(patch), patch being real. */
static void patch_spectrum(tt_mosse *m, const float *patch)
{
    int k;
    for (k = 0; k < m->n; k++) {
        m->spec[2 * k]     = patch[k];
        m->spec[2 * k + 1] = 0.0f;
    }
    tt_fft2d_forward(&m->fft, m->spec);
}

/*
 * Blend one sample into A and B:  A = keep*A + add*G.conj(F),  B = keep*B + add*|F|^2.
 * Init uses keep=1, add=1 (plain sum); online update uses keep=1-rate, add=rate.
 */
static void blend_sample(tt_mosse *m, const float *patch, float keep, float add)
{
    int k;
    patch_spectrum(m, patch);
    for (k = 0; k < m->n; k++) {
        const float fr = m->spec[2 * k], fi = m->spec[2 * k + 1];
        const float gr = m->G[2 * k],    gi = m->G[2 * k + 1];
        /* G * conj(F) = (gr + i gi)(fr - i fi) */
        const float ar = gr * fr + gi * fi;
        const float ai = gi * fr - gr * fi;
        m->A[2 * k]     = keep * m->A[2 * k]     + add * ar;
        m->A[2 * k + 1] = keep * m->A[2 * k + 1] + add * ai;
        m->B[k]         = keep * m->B[k]         + add * (fr * fr + fi * fi);
    }
}

void tt_mosse_reset(tt_mosse *m)
{
    memset(m->A, 0, sizeof(float) * 2 * (size_t)m->n);
    memset(m->B, 0, sizeof(float) * (size_t)m->n);
    m->count = 0;
    m->lambda = 0.0f;
}

void tt_mosse_add_sample(tt_mosse *m, const float *patch)
{
    blend_sample(m, patch, 1.0f, 1.0f);
    m->count++;
}

void tt_mosse_finish_init(tt_mosse *m, float lambda_rel)
{
    const float inv = m->count > 0 ? 1.0f / (float)m->count : 1.0f;
    double mean_b = 0.0;
    int k;

    /* Turn the sums into averages so later online updates have the right weight. */
    for (k = 0; k < m->n; k++) {
        m->A[2 * k]     *= inv;
        m->A[2 * k + 1] *= inv;
        m->B[k]         *= inv;
        mean_b += m->B[k];
    }
    mean_b /= m->n;

    /* lambda relative to the average energy keeps it independent of patch size.
     * The tiny floor stops a division by zero on a completely flat patch. */
    m->lambda = (float)(lambda_rel * mean_b) + 1e-6f;
}

void tt_mosse_copy_model(tt_mosse *dst, const tt_mosse *src)
{
    memcpy(dst->A, src->A, sizeof(float) * 2 * (size_t)src->n);
    memcpy(dst->B, src->B, sizeof(float) * (size_t)src->n);
    dst->lambda = src->lambda;
    dst->count = src->count;
}

void tt_mosse_update(tt_mosse *m, const float *patch)
{
    blend_sample(m, patch, 1.0f - m->rate, m->rate);
}

void tt_mosse_respond(tt_mosse *m, const float *patch, float *response)
{
    int k;
    patch_spectrum(m, patch);
    for (k = 0; k < m->n; k++) {
        /* conj(H) = A / (B + lambda); B is real so this is a scale. */
        const float inv = 1.0f / (m->B[k] + m->lambda);
        const float hr = m->A[2 * k] * inv, hi = m->A[2 * k + 1] * inv;
        const float zr = m->spec[2 * k],    zi = m->spec[2 * k + 1];
        m->spec[2 * k]     = hr * zr - hi * zi;
        m->spec[2 * k + 1] = hr * zi + hi * zr;
    }
    tt_fft2d_inverse(&m->fft, m->spec);
    for (k = 0; k < m->n; k++) {
        response[k] = m->spec[2 * k];   /* imaginary part is ~0 */
    }
}

void tt_mosse_respond2(tt_mosse *m, const float *pa, const float *pb, float *ra, float *rb)
{
    const int w = m->w, h = m->h;
    int u, v, k;

    /* Z = FFT(a + i b) */
    for (k = 0; k < m->n; k++) {
        m->spec[2 * k]     = pa[k];
        m->spec[2 * k + 1] = pb[k];
    }
    tt_fft2d_forward(&m->fft, m->spec);

    for (v = 0; v < h; v++) {
        const int vn = (h - v) % h;                 /* -v mod h */
        for (u = 0; u < w; u++) {
            const int un = (w - u) % w;             /* -u mod w */
            const int i = v * w + u, j = vn * w + un;
            const float zr = m->spec[2 * i], zi = m->spec[2 * i + 1];
            const float yr = m->spec[2 * j], yi = m->spec[2 * j + 1];
            /* separate the two spectra */
            const float ar = 0.5f * (zr + yr), ai = 0.5f * (zi - yi);
            const float br = 0.5f * (zi + yi), bi = 0.5f * (yr - zr);
            /* filter conj(H) = A / (B + lambda) */
            const float inv = 1.0f / (m->B[i] + m->lambda);
            const float hr = m->A[2 * i] * inv, hi = m->A[2 * i + 1] * inv;
            const float rar = hr * ar - hi * ai, rai = hr * ai + hi * ar;
            const float rbr = hr * br - hi * bi, rbi = hr * bi + hi * br;
            /* pack RA + i*RB */
            m->spec2[2 * i]     = rar - rbi;
            m->spec2[2 * i + 1] = rai + rbr;
        }
    }
    tt_fft2d_inverse(&m->fft, m->spec2);
    for (k = 0; k < m->n; k++) {
        ra[k] = m->spec2[2 * k];
        rb[k] = m->spec2[2 * k + 1];
    }
}

/*
 * Sub-pixel refinement: fit a parabola through the peak and its two neighbours
 * (left, centre, right) and return where that parabola is highest, as an
 * offset in [-0.5, 0.5]. Done separately in x and y.
 */
static float parabola_offset(float l, float c, float r)
{
    const float den = l - 2.0f * c + r;
    float off;
    if (den >= 0.0f) {          /* not a proper maximum (flat or a dip) */
        return 0.0f;
    }
    off = 0.5f * (l - r) / den;
    if (off > 0.5f)  off = 0.5f;
    if (off < -0.5f) off = -0.5f;
    return off;
}

/* Half-size of the square around the peak left out of the sidelobe (11x11, as in the paper). */
#define TT_PSR_EXCLUDE 5

float tt_mosse_psr(const float *r, int w, int h, int ix, int iy, float peak)
{
    double sum = 0.0, sq = 0.0, mean, var;
    int count = 0;
    int x, y;

    for (y = 0; y < h; y++) {
        /* distance to the peak, measured the short way round (response wraps) */
        int dy = abs(y - iy);
        if (dy > h / 2) dy = h - dy;
        for (x = 0; x < w; x++) {
            int dx = abs(x - ix);
            if (dx > w / 2) dx = w - dx;
            if (dx <= TT_PSR_EXCLUDE && dy <= TT_PSR_EXCLUDE) {
                continue;
            }
            sum += r[y * w + x];
            sq  += (double)r[y * w + x] * r[y * w + x];
            count++;
        }
    }
    if (count < 2) {
        return 0.0f;
    }
    mean = sum / count;
    var = sq / count - mean * mean;
    if (var < 1e-12) {
        return 0.0f;   /* flat response: no evidence of a match */
    }
    return (float)((peak - mean) / sqrt(var));
}

tt_peak tt_mosse_find_peak(const float *r, int w, int h)
{
    tt_peak p;
    int best = 0;
    int k;
    const int n = w * h;

    for (k = 1; k < n; k++) {
        if (r[k] > r[best]) {
            best = k;
        }
    }
    p.ix = best % w;
    p.iy = best / w;
    p.value = r[best];

    /* The response is periodic (FFT), so neighbours wrap around the edges. */
    p.x = (float)p.ix + parabola_offset(r[p.iy * w + (p.ix - 1 + w) % w], r[best],
                                        r[p.iy * w + (p.ix + 1) % w]);
    p.y = (float)p.iy + parabola_offset(r[((p.iy - 1 + h) % h) * w + p.ix], r[best],
                                        r[((p.iy + 1) % h) * w + p.ix]);
    p.psr = tt_mosse_psr(r, w, h, p.ix, p.iy, p.value);
    return p;
}
