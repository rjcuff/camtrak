/*
 * fft.c - iterative radix-2 Cooley-Tukey FFT.
 *
 * How it works (1D, length n = 2^k):
 *   The DFT  X[f] = sum_t x[t] * e^(-2*pi*i*f*t/n)  costs O(n^2).
 *   Cooley-Tukey splits it into the DFT of the even samples and the DFT of the
 *   odd samples, each of length n/2, and combines them with one "butterfly":
 *       X[f]       = E[f] + w^f * O[f]
 *       X[f + n/2] = E[f] - w^f * O[f]        where w = e^(-2*pi*i/n)
 *   Repeating the split log2(n) times gives O(n log n).
 *
 *   Doing it without recursion: first reorder the input into bit-reversed index
 *   order (that is exactly where the repeated even/odd splitting puts each
 *   sample), then run log2(n) passes of butterflies on blocks of size
 *   2, 4, 8, ... n.
 *
 * 2D: the 2D DFT is separable, so we FFT every row, then every column.
 */
#include "fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const double TT_PI = 3.14159265358979323846;

int tt_is_pow2(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

int tt_fft1d_init(tt_fft1d *p, int n)
{
    int bits = 0;
    int i;

    memset(p, 0, sizeof(*p));
    if (n < 2 || !tt_is_pow2(n)) {
        return -1;
    }
    p->n = n;
    while ((1 << bits) < n) {
        bits++;
    }

    p->cos_t = malloc(sizeof(float) * (size_t)(n / 2));
    p->sin_t = malloc(sizeof(float) * (size_t)(n / 2));
    p->rev   = malloc(sizeof(int) * (size_t)n);
    if (!p->cos_t || !p->sin_t || !p->rev) {
        tt_fft1d_free(p);
        return -1;
    }

    /* Twiddles computed in double, then rounded once, for accuracy. */
    for (i = 0; i < n / 2; i++) {
        double a = 2.0 * TT_PI * (double)i / (double)n;
        p->cos_t[i] = (float)cos(a);
        p->sin_t[i] = (float)sin(a);
    }

    for (i = 0; i < n; i++) {
        int r = 0;
        int b;
        for (b = 0; b < bits; b++) {
            if (i & (1 << b)) {
                r |= 1 << (bits - 1 - b);
            }
        }
        p->rev[i] = r;
    }
    return 0;
}

void tt_fft1d_free(tt_fft1d *p)
{
    free(p->cos_t);
    free(p->sin_t);
    free(p->rev);
    memset(p, 0, sizeof(*p));
}

void tt_fft1d_run(const tt_fft1d *p, float *x, int inverse)
{
    const int n = p->n;
    /* Forward uses w = cos - i*sin, inverse uses w = cos + i*sin. */
    const float sign = inverse ? 1.0f : -1.0f;
    int i;
    int len;

    /* Step 1: bit-reversal permutation (swap each pair once). */
    for (i = 0; i < n; i++) {
        int r = p->rev[i];
        if (r > i) {
            float tr = x[2 * i], ti = x[2 * i + 1];
            x[2 * i]     = x[2 * r];
            x[2 * i + 1] = x[2 * r + 1];
            x[2 * r]     = tr;
            x[2 * r + 1] = ti;
        }
    }

    /* Step 2: butterflies. Blocks of size len are combined from two halves. */
    for (len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;   /* stride into the twiddle table */
        int start;
        for (start = 0; start < n; start += len) {
            int j;
            for (j = 0; j < half; j++) {
                const float wr = p->cos_t[j * step];
                const float wi = sign * p->sin_t[j * step];
                float *a = x + 2 * (start + j);
                float *b = x + 2 * (start + j + half);
                /* t = w * b */
                const float tr = wr * b[0] - wi * b[1];
                const float ti = wr * b[1] + wi * b[0];
                b[0] = a[0] - tr;
                b[1] = a[1] - ti;
                a[0] += tr;
                a[1] += ti;
            }
        }
    }
}

int tt_fft2d_init(tt_fft2d *p, int w, int h)
{
    memset(p, 0, sizeof(*p));
    p->w = w;
    p->h = h;
    if (tt_fft1d_init(&p->row, w) != 0 || tt_fft1d_init(&p->col, h) != 0) {
        tt_fft2d_free(p);
        return -1;
    }
    p->column = malloc(sizeof(float) * 2 * (size_t)h);
    if (!p->column) {
        tt_fft2d_free(p);
        return -1;
    }
    return 0;
}

void tt_fft2d_free(tt_fft2d *p)
{
    tt_fft1d_free(&p->row);
    tt_fft1d_free(&p->col);
    free(p->column);
    memset(p, 0, sizeof(*p));
}

static void fft2d_run(const tt_fft2d *p, float *d, int inverse)
{
    const int w = p->w;
    const int h = p->h;
    int x, y;

    /* Rows are contiguous in memory: transform in place. */
    for (y = 0; y < h; y++) {
        tt_fft1d_run(&p->row, d + 2 * (size_t)y * (size_t)w, inverse);
    }

    /* Columns are strided: copy each into a contiguous scratch buffer first
     * (much friendlier to the cache), transform, copy back. */
    for (x = 0; x < w; x++) {
        for (y = 0; y < h; y++) {
            const size_t k = 2 * ((size_t)y * (size_t)w + (size_t)x);
            p->column[2 * y]     = d[k];
            p->column[2 * y + 1] = d[k + 1];
        }
        tt_fft1d_run(&p->col, p->column, inverse);
        for (y = 0; y < h; y++) {
            const size_t k = 2 * ((size_t)y * (size_t)w + (size_t)x);
            d[k]     = p->column[2 * y];
            d[k + 1] = p->column[2 * y + 1];
        }
    }
}

void tt_fft2d_forward(const tt_fft2d *p, float *data)
{
    fft2d_run(p, data, 0);
}

void tt_fft2d_inverse(const tt_fft2d *p, float *data)
{
    const size_t count = 2 * (size_t)p->w * (size_t)p->h;
    const float scale = 1.0f / ((float)p->w * (float)p->h);
    size_t i;

    fft2d_run(p, data, 1);
    for (i = 0; i < count; i++) {
        data[i] *= scale;
    }
}
