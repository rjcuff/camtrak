/*
 * mosse.h - MOSSE correlation filter (Bolme et al., CVPR 2010).
 *
 * The idea in one paragraph:
 *   We want a small filter h such that correlating it with the target patch f
 *   gives a sharp Gaussian peak g centred on the target:   f (*) h = g.
 *   Correlation is an element-wise product in the Fourier domain, so with
 *   capital letters for FFTs:   F . conj(H) = G   =>   conj(H) = G / F.
 *   Dividing by one noisy sample is unstable, so MOSSE takes the least-squares
 *   solution over many samples:
 *         conj(H) = sum(G . conj(F)) / (sum(F . conj(F)) + lambda)
 *               =        A          /          B
 *   A and B are kept as running averages so the filter slowly adapts to
 *   appearance change. To find the target in a new frame we cut a patch z and
 *   compute  response = IFFT(conj(H) . Z): the peak of the response is where the target moved.
 */
#ifndef TT_MOSSE_H
#define TT_MOSSE_H

#include "fft.h"

typedef struct {
    int      w, h, n;   /* patch size, n = w*h */
    float    rate;      /* learning rate for the running average */
    float    lambda;    /* effective regulariser, set by tt_mosse_finish_init */
    int      count;     /* samples accumulated during init */
    tt_fft2d fft;
    float   *G;         /* FFT of the desired Gaussian response (complex, n) */
    float   *A;         /* numerator   sum G.conj(F)   (complex, n) */
    float   *B;         /* denominator sum |F|^2       (real, n) */
    float   *spec;      /* scratch spectrum (complex, n) */
    float   *spec2;     /* second scratch spectrum for tt_mosse_respond2 (complex, n) */
} tt_mosse;

/* sigma: width of the Gaussian peak, in patch pixels. Return 0 on success. */
int  tt_mosse_init(tt_mosse *m, int w, int h, float sigma, float rate);
void tt_mosse_free(tt_mosse *m);

/* Initial training: reset, add a few samples, then finish. */
void tt_mosse_reset(tt_mosse *m);
void tt_mosse_add_sample(tt_mosse *m, const float *patch);
void tt_mosse_finish_init(tt_mosse *m, float lambda_rel);

/* Copy the learned filter (A, B, lambda) of `src` into `dst` (same size, both initialised). */
void tt_mosse_copy_model(tt_mosse *dst, const tt_mosse *src);

/* Online adaptation: A,B = (1-rate)*A,B + rate*(new sample). */
void tt_mosse_update(tt_mosse *m, const float *patch);

/* Correlation response for a preprocessed patch (n real values). */
void tt_mosse_respond(tt_mosse *m, const float *patch, float *response);

/*
 * Responses for two patches for the price of one forward and one inverse FFT.
 * Trick: the FFT of a real image is conjugate-symmetric, X[-k] = conj(X[k]).
 * Put patch a in the real part and b in the imaginary part, transform once,
 * and the two spectra can be separated again:
 *     A[k] = (Z[k] + conj(Z[-k])) / 2        B[k] = (Z[k] - conj(Z[-k])) / 2i
 * The responses are real too, so packing them as RA + i*RB and inverting once
 * gives response a in the real part and response b in the imaginary part.
 */
void tt_mosse_respond2(tt_mosse *m, const float *patch_a, const float *patch_b,
                       float *response_a, float *response_b);

typedef struct {
    int   ix, iy;   /* integer location of the maximum */
    float x, y;     /* sub-pixel refined location */
    float value;    /* response at the maximum */
    float psr;      /* peak-to-sidelobe ratio, see tt_mosse_psr */
} tt_peak;

/* Find the response maximum, refine it to sub-pixel, and compute its PSR. */
tt_peak tt_mosse_find_peak(const float *response, int w, int h);

/*
 * Peak-to-sidelobe ratio: how much the peak stands out from everything else.
 *     PSR = (peak - mean(sidelobe)) / std(sidelobe)
 * where the sidelobe is the whole response except a small square around the
 * peak. A clean, unique match gives PSR ~ 20+; occlusion, blur or a wrong
 * object flattens the response and PSR drops below ~7 (MOSSE paper).
 */
float tt_mosse_psr(const float *response, int w, int h, int ix, int iy, float peak);

#endif
