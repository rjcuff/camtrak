#include "patch.h"

#include <math.h>
#include <stddef.h>

static const double TT_PI = 3.14159265358979323846;

int tt_frame_valid(const tether_frame *f)
{
    if (!f || !f->data || f->width <= 0 || f->height <= 0) {
        return 0;
    }
    if (f->format != TETHER_FORMAT_GRAY8 && f->format != TETHER_FORMAT_BGR8) {
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
}

/* log(1 + gray) of integer pixel (x, y), clamped to the image. */
static inline float logval_at(const tether_frame *f, const tt_log_lut *lut, int x, int y)
{
    const uint8_t *p;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= f->width)  x = f->width - 1;
    if (y >= f->height) y = f->height - 1;
    p = f->data + (size_t)y * (size_t)f->stride + (size_t)x * (size_t)f->format;
    if (f->format == TETHER_FORMAT_GRAY8) {
        return lut->v[p[0]];
    }
    /* BGR -> luma, integer weights summing to 256 (0.114 B + 0.587 G + 0.299 R). */
    return lut->v[(29 * p[0] + 150 * p[1] + 77 * p[2]) >> 8];
}

/* Bilinear sample at continuous point (px, py). Pixel centres sit at i + 0.5. */
static inline float sample_bilinear(const tether_frame *f, const tt_log_lut *lut,
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

void tt_sample_patch(const tether_frame *f, const tt_log_lut *lut, float cx, float cy,
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
     * normalising, and bias the scale search towards ever-bigger boxes. */
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
