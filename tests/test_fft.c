/* test_fft: compare the radix-2 FFT against a slow, obviously-correct DFT. */
#include "fft.h"
#include "mosse.h"
#include "test_util.h"

#include <math.h>
#include <stdlib.h>

static const double PI = 3.14159265358979323846;

static unsigned rng_state = 12345u;
static float rnd(void) /* uniform in [-1, 1] */
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state % 20001u) / 10000.0f - 1.0f;
}

/* Textbook 2D DFT in double precision: O((wh)^2). */
static void naive_dft2d(const float *in, double *out, int w, int h)
{
    int u, v, x, y;
    for (v = 0; v < h; v++) {
        for (u = 0; u < w; u++) {
            double re = 0.0, im = 0.0;
            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++) {
                    const double a = -2.0 * PI * ((double)u * x / w + (double)v * y / h);
                    const double xr = in[2 * (y * w + x)], xi = in[2 * (y * w + x) + 1];
                    re += xr * cos(a) - xi * sin(a);
                    im += xr * sin(a) + xi * cos(a);
                }
            }
            out[2 * (v * w + u)] = re;
            out[2 * (v * w + u) + 1] = im;
        }
    }
}

static void check_size(int w, int h)
{
    const int n = w * h;
    float *orig = malloc(sizeof(float) * 2 * n);
    float *data = malloc(sizeof(float) * 2 * n);
    double *ref = malloc(sizeof(double) * 2 * n);
    tt_fft2d plan;
    double max_err = 0.0, max_mag = 0.0, rt_err = 0.0;
    int i;

    CHECK(orig && data && ref);
    if (!orig || !data || !ref) {
        free(orig); free(data); free(ref);
        return;
    }
    for (i = 0; i < 2 * n; i++) {
        orig[i] = data[i] = rnd();
    }

    CHECK(tt_fft2d_init(&plan, w, h) == 0);
    tt_fft2d_forward(&plan, data);
    naive_dft2d(orig, ref, w, h);
    for (i = 0; i < 2 * n; i++) {
        max_err = fmax(max_err, fabs(data[i] - ref[i]));
        max_mag = fmax(max_mag, fabs(ref[i]));
    }

    tt_fft2d_inverse(&plan, data);
    for (i = 0; i < 2 * n; i++) {
        rt_err = fmax(rt_err, fabs(data[i] - orig[i]));
    }
    tt_fft2d_free(&plan);

    printf("fft %4dx%-4d  rel err vs DFT %.2e   round-trip err %.2e\n",
           w, h, max_err / max_mag, rt_err);
    CHECK(max_err / max_mag < 1e-5);
    CHECK(rt_err < 1e-5);

    free(orig); free(data); free(ref);
}

/* tt_mosse_respond2 (two patches, one FFT pair) must equal two tt_mosse_respond calls. */
static void check_respond2(void)
{
    enum { W = 32, H = 16, N = W * H };
    static float train[N], a[N], b[N], ra1[N], rb1[N], ra2[N], rb2[N];
    tt_mosse m;
    double max_err = 0.0, max_mag = 0.0;
    int i;

    for (i = 0; i < N; i++) {
        train[i] = rnd();
        a[i] = rnd();
        b[i] = rnd();
    }
    CHECK(tt_mosse_init(&m, W, H, 2.0f, 0.125f) == 0);
    tt_mosse_add_sample(&m, train);
    tt_mosse_finish_init(&m, 0.01f);

    tt_mosse_respond(&m, a, ra1);
    tt_mosse_respond(&m, b, rb1);
    tt_mosse_respond2(&m, a, b, ra2, rb2);
    for (i = 0; i < N; i++) {
        max_err = fmax(max_err, fmax(fabs(ra1[i] - ra2[i]), fabs(rb1[i] - rb2[i])));
        max_mag = fmax(max_mag, fmax(fabs(ra1[i]), fabs(rb1[i])));
    }
    tt_mosse_free(&m);
    printf("respond2 vs 2x respond: rel err %.2e\n", max_err / max_mag);
    CHECK(max_err / max_mag < 1e-4);
}

int main(void)
{
    tt_fft1d bad;
    static const int sizes[][2] = { {2, 2}, {8, 4}, {4, 16}, {16, 16}, {32, 64}, {64, 64} };
    size_t i;

    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        check_size(sizes[i][0], sizes[i][1]);
    }
    check_respond2();

    /* Non-power-of-two sizes must be rejected, not silently mis-transformed. */
    CHECK(tt_fft1d_init(&bad, 12) != 0);
    CHECK(tt_fft1d_init(&bad, 0) != 0);
    CHECK(tt_fft1d_init(&bad, 1) != 0);

    if (tt_failures) {
        fprintf(stderr, "%d check(s) failed\n", tt_failures);
        return 1;
    }
    printf("test_fft: OK\n");
    return 0;
}
