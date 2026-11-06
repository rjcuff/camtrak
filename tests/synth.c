#include "synth.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PAN_RANGE 200

static unsigned xorshift(unsigned *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Horizontal + vertical box blur of radius r, in place (uses a temp row/col). */
static void box_blur(uint8_t *img, int w, int h, int r)
{
    int *tmp = malloc(sizeof(int) * (size_t)(w > h ? w : h));
    int x, y, k;
    if (!tmp) {
        return;
    }
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            int sum = 0, cnt = 0;
            for (k = -r; k <= r; k++) {
                if (x + k >= 0 && x + k < w) { sum += img[y * w + x + k]; cnt++; }
            }
            tmp[x] = sum / cnt;
        }
        for (x = 0; x < w; x++) img[y * w + x] = (uint8_t)tmp[x];
    }
    for (x = 0; x < w; x++) {
        for (y = 0; y < h; y++) {
            int sum = 0, cnt = 0;
            for (k = -r; k <= r; k++) {
                if (y + k >= 0 && y + k < h) { sum += img[(y + k) * w + x]; cnt++; }
            }
            tmp[y] = sum / cnt;
        }
        for (y = 0; y < h; y++) img[y * w + x] = (uint8_t)tmp[y];
    }
    free(tmp);
}

int tt_synth_init(tt_synth *s, int w, int h, int target_size, int noise, unsigned seed)
{
    int i;
    memset(s, 0, sizeof(*s));
    s->w = w;
    s->h = h;
    s->tsz = target_size;
    s->noise = noise;
    s->rng = seed ? seed : 1u;
    s->bg_w = w + PAN_RANGE;
    s->occ_start = s->occ_end = -1;
    s->growth = 1.0;

    s->frame  = malloc((size_t)w * (size_t)h);
    s->bg     = malloc((size_t)s->bg_w * (size_t)h);
    s->target = malloc((size_t)target_size * (size_t)target_size);
    if (!s->frame || !s->bg || !s->target) {
        tt_synth_free(s);
        return -1;
    }

    /* Background: smoothed noise, mid contrast (clutter, but not target-like). */
    for (i = 0; i < s->bg_w * h; i++) {
        s->bg[i] = (uint8_t)(60 + xorshift(&s->rng) % 130);
    }
    box_blur(s->bg, s->bg_w, h, 3);

    /* Target: a high-contrast pattern of 4x4-pixel cells. */
    {
        int x, y;
        for (y = 0; y < target_size; y += 4) {
            for (x = 0; x < target_size; x += 4) {
                const uint8_t v = (xorshift(&s->rng) & 1u) ? 230 : 25;
                int a, b;
                for (b = y; b < y + 4 && b < target_size; b++)
                    for (a = x; a < x + 4 && a < target_size; a++)
                        s->target[b * target_size + a] = v;
            }
        }
    }
    return 0;
}

void tt_synth_free(tt_synth *s)
{
    free(s->frame);
    free(s->bg);
    free(s->target);
    memset(s, 0, sizeof(*s));
}

void tt_synth_set_occlusion(tt_synth *s, int start, int end)
{
    s->occ_start = start;
    s->occ_end = end;
}

void tt_synth_render(tt_synth *s, int index)
{
    const double t = (double)index;
    /* background pans left/right */
    const int pan = (int)(PAN_RANGE / 2 + (PAN_RANGE / 2 - 10) * sin(0.02 * t));
    /* target center follows a Lissajous curve */
    const double cx = s->w * (0.5 + 0.3 * sin(0.03 * t));
    const double cy = s->h * (0.5 + 0.25 * sin(0.047 * t + 1.0));
    /* current size: the texture is stretched (nearest neighbor) as it grows */
    const int sz = (int)floor(s->tsz * pow(s->growth, t) + 0.5);
    const int x0 = (int)floor(cx - sz / 2.0 + 0.5);
    const int y0 = (int)floor(cy - sz / 2.0 + 0.5);
    int x, y;

    s->index = index;
    for (y = 0; y < s->h; y++) {
        memcpy(s->frame + (size_t)y * s->w, s->bg + (size_t)y * s->bg_w + pan, (size_t)s->w);
    }
    for (y = 0; y < sz; y++) {
        for (x = 0; x < sz; x++) {
            const int fx = x0 + x, fy = y0 + y;
            if (fx >= 0 && fy >= 0 && fx < s->w && fy < s->h) {
                s->frame[fy * s->w + fx] = s->target[(y * s->tsz / sz) * s->tsz + x * s->tsz / sz];
            }
        }
    }
    /* Occluder: a flat gray block twice the target size that follows the
     * target, so it is completely invisible (think: skier behind a tree). */
    s->occluded = index >= s->occ_start && index <= s->occ_end;
    if (s->occluded) {
        const int m = sz / 2;
        for (y = y0 - m; y < y0 + sz + m; y++) {
            for (x = x0 - m; x < x0 + sz + m; x++) {
                if (x >= 0 && y >= 0 && x < s->w && y < s->h) {
                    s->frame[y * s->w + x] = 110;
                }
            }
        }
    }
    if (s->noise > 0) {
        const int span = 2 * s->noise + 1;
        for (y = 0; y < s->w * s->h; y++) {
            int v = s->frame[y] + (int)(xorshift(&s->rng) % (unsigned)span) - s->noise;
            s->frame[y] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }

    s->truth.x = (float)x0;
    s->truth.y = (float)y0;
    s->truth.w = (float)sz;
    s->truth.h = (float)sz;
}

camtrak_frame tt_synth_frame(const tt_synth *s)
{
    camtrak_frame f;
    f.data = s->frame;
    f.width = s->w;
    f.height = s->h;
    f.stride = s->w;
    f.format = CAMTRAK_FORMAT_GRAY8;
    return f;
}
