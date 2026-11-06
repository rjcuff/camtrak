/*
 * fft.h - small radix-2 FFT, 1D and 2D, single precision.
 *
 * Complex arrays are stored interleaved: data[2k] = real part, data[2k+1] = imaginary part.
 *
 * A "plan" precomputes the twiddle factors (the roots of unity) and the
 * bit-reversal table once, so running a transform never allocates.
 *
 * Swapping in a faster backend (pocketfft, NEON) later only needs this header's
 * functions re-implemented.
 */
#ifndef TT_FFT_H
#define TT_FFT_H

typedef struct {
    int    n;       /* transform length, a power of two >= 2 */
    float *cos_t;   /* cos(2*pi*k/n) for k < n/2 */
    float *sin_t;   /* sin(2*pi*k/n) for k < n/2 */
    int   *rev;     /* rev[i] = i with its log2(n) bits reversed */
} tt_fft1d;

typedef struct {
    int      w, h;
    tt_fft1d row;     /* length w, used on each row */
    tt_fft1d col;     /* length h, used on each column */
    float   *column;  /* scratch for one column (2*h floats) */
} tt_fft2d;

int  tt_is_pow2(int n);

/* Return 0 on success, -1 on bad size or out of memory. */
int  tt_fft1d_init(tt_fft1d *p, int n);
void tt_fft1d_free(tt_fft1d *p);
/* In place, unnormalised. inverse=0: forward (e^-i), inverse=1: backward (e^+i). */
void tt_fft1d_run(const tt_fft1d *p, float *data, int inverse);

int  tt_fft2d_init(tt_fft2d *p, int w, int h);
void tt_fft2d_free(tt_fft2d *p);
/* data is w*h complex values, row-major. */
void tt_fft2d_forward(const tt_fft2d *p, float *data);
/* Inverse includes the 1/(w*h) scaling, so inverse(forward(x)) == x. */
void tt_fft2d_inverse(const tt_fft2d *p, float *data);

#endif
