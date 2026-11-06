/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
/*
 * camtrak.c - public API: owns the tracker state and runs the per-frame pipeline.
 *
 * Per frame, depending on the current mode:
 *
 *   TRACKING / COASTING
 *     1. Kalman predict: where should the target be now?
 *     2. cut a patch there and correlate with the MOSSE filter
 *     3. PSR >= psr_track  -> the peak is a measurement: Kalman correct, TRACKING
 *        otherwise         -> ignore it, keep the (damped) prediction, COASTING
 *     4. PSR >= psr_update -> blend the new patch into the appearance model
 *     5. lost_after misses in a row -> LOST
 *
 *   LOST
 *     Search a spiral of windows outward from the last confident position,
 *     reacquire_budget windows per frame. A match with PSR >= psr_reacquire
 *     restarts tracking there; otherwise report LOST at the last known box.
 */
#include "camtrak/camtrak.h"

#include "kalman.h"
#include "mosse.h"
#include "color.h"
#include "patch.h"
#include "scale.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TT_MIN_PATCH 16
#define TT_MIN_BOX   4.0f

/* Confidence is PSR mapped linearly: PSR <= FLOOR -> 0, PSR >= FULL -> 1. */
#define TT_PSR_FLOOR 5.0f
#define TT_PSR_FULL  20.0f

/* Safety cap on off-frame grid cells skipped per LOST frame. */
#define TT_MAX_SKIPS_PER_WINDOW 16

/* Scale filter (scale.c): number of sizes sampled, and how fast it learns.
 * Slow learning keeps the size anchored to what the user selected. */
#define TT_SCALE_SAMPLES 17
#define TT_SCALE_LR      0.01f
/* Most the box size may change in one frame. Real objects change size
 * smoothly; sudden jumps come from clutter (a shadow, a tree) at the box edge. */
#define TT_SCALE_MAX_CHANGE 0.025f
/* Growing is capped harder than shrinking: real objects grow slowly (they
 * approach, or the camera zooms), while a box that has started to include
 * background tends to keep growing into it. */
#define TT_SCALE_MAX_GROWTH 0.01f
/* camtrak_verify: shape match needed to accept an external candidate. Lower
 * than psr_reacquire because the candidate already passed another model. */
#define TT_VERIFY_PSR 7.0f
/* Color match (Bhattacharyya, 1 = as at selection) a candidate needs; vetoes
 * look-alikes. Measured on drone ski footage: same skier 0.94-0.98 across
 * shots, other skier 0.70-0.75, snow 0.28. */
#define TT_COLOR_MIN 0.85f
/* With near-certain color, a weaker shape match is enough (new pose). */
#define TT_COLOR_STRONG 0.93f
#define TT_VERIFY_PSR_STRONG_COLOR 5.0f
/* Own LOST search: with near-certain color, this shape match re-acquires
 * (instead of psr_reacquire). Higher than for camtrak_verify because the
 * search tries many windows, so a chance match is likelier. */
#define TT_REACQ_PSR_STRONG_COLOR 8.0f
/* A candidate box must be at least this much inside the frame. */
#define TT_MIN_INSIDE 0.7f
/* Learning weight for a view confirmed via camtrak_reacquire (normal is learning_rate). */
#define TT_REACQUIRE_RATE 0.5f
/* Smallest box side we will shrink to, in pixels. */
#define TT_MIN_SIDE 6.0f
/* Re-detection also tries the target at these relative sizes, one per sweep. */
static const float k_redetect_scales[] = { 1.0f, 0.8f, 1.25f };
#define TT_N_REDETECT_SCALES 3

struct camtrak_tracker {
    camtrak_params p;
    int   locked;
    int   pw, ph;     /* patch size in patch pixels (powers of two) */
    float base_step;  /* image pixels per patch pixel at lock time */
    float base_w, base_h; /* target size at lock time */
    float scale;      /* current size relative to lock time */
    float step;       /* image pixels per patch pixel now = base_step * scale */
    float cx, cy;     /* current target center estimate, image pixels */
    float tw, th;     /* current target size, image pixels = base * scale */
    tt_mosse  mosse;  /* appearance model, adapts every confident frame */
    tt_mosse  anchor; /* frozen copy from lock time: what the user selected */
    int       lost_frames; /* frames spent LOST (alternates the re-detection model) */
    camtrak_debug dbg;      /* what the last update did, for camtrak_get_debug */
    tt_log_lut lut;   /* log(1 + v) table + BGR->channel weights used by patch sampling */
    float channel_gain; /* separability of the chosen channel vs brightness, from lock */
    tt_scale  sc;     /* size estimator; unused when scale_step <= 1 */
    tt_color  color;  /* color model of the selection (BGR input only) */
    int       use_scale;
    tt_kalman kf;     /* motion model for (cx, cy) */
    int   misses;     /* consecutive frames without an accepted measurement */
    float good_x, good_y;  /* last position with an accepted measurement */

    /* LOST-mode re-detection: spiral of grid cells around (good_x, good_y) */
    int   lost;
    int   ring, ring_idx;  /* spiral cursor: ring number, index within ring */
    int   sweep_scale;     /* which k_redetect_scales entry the current sweep uses */
    float base_spacing;    /* distance between grid cells at lock-time scale */

    float *window;    /* cosine window, pw*ph */
    float *patch;     /* working patch, pw*ph */
    float *response;  /* correlation output, pw*ph */
};

const char *camtrak_version(void)
{
    return "0.4.0";
}

/* ------------------------------------------------------------------ params */

camtrak_params camtrak_default_params(void)
{
    camtrak_params p;
    p.padding          = 2.5f;
    p.template_size    = 64;
    p.learning_rate    = 0.125f;  /* value from the MOSSE paper */
    p.sigma_factor     = 0.1f;
    p.lambda           = 0.01f;
    /* PSR thresholds measured on the synthetic clip: clean tracking >= 14.6,
     * occluded 4..11. Learning needs more certainty than measuring. */
    p.psr_update       = 12.0f;
    p.psr_track        = 11.0f;
    p.kalman_q         = 0.5f;
    p.kalman_r         = 1.0f;
    p.coast_damping    = 0.95f;   /* coasting can drift at most ~20x the last velocity */
    p.lost_after       = 20;      /* ~0.7 s at 30 fps */
    p.psr_reacquire    = 14.0f;
    p.reacquire_budget = 24;
    p.scale_step       = 1.04f;   /* scale filter samples sizes 1.04^-8 .. 1.04^8 (-27%..+37%) */
    p.adaptive_color   = 1;
    return p;
}

static int params_valid(const camtrak_params *p)
{
    return p->padding >= 1.0f && p->padding <= 10.0f
        && tt_is_pow2(p->template_size) && p->template_size >= TT_MIN_PATCH
        && p->template_size <= 256
        && p->learning_rate > 0.0f && p->learning_rate <= 1.0f
        && p->sigma_factor > 0.0f && p->sigma_factor <= 1.0f
        && p->lambda >= 0.0f
        && p->psr_update >= 0.0f && p->psr_track >= 0.0f && p->psr_reacquire >= 0.0f
        && p->kalman_q > 0.0f && p->kalman_r > 0.0f
        && p->coast_damping >= 0.0f && p->coast_damping <= 1.0f
        && p->lost_after >= 1
        && p->reacquire_budget >= 1 && p->reacquire_budget <= 1024
        && (p->scale_step == 0.0f || (p->scale_step >= 1.0f && p->scale_step <= 1.5f))
        && (p->adaptive_color == 0 || p->adaptive_color == 1);
}

/* ------------------------------------------------------------- lifecycle */

camtrak_tracker *camtrak_create(const camtrak_params *params)
{
    camtrak_tracker *t;
    const camtrak_params p = params ? *params : camtrak_default_params();

    if (!params_valid(&p)) {
        return NULL;
    }
    t = calloc(1, sizeof(*t));
    if (t) {
        t->p = p;
        tt_make_log_lut(&t->lut);
    }
    return t;
}

static void release_buffers(camtrak_tracker *t)
{
    tt_mosse_free(&t->mosse);
    tt_mosse_free(&t->anchor);
    free(t->window);
    free(t->patch);
    free(t->response);
    t->window = t->patch = t->response = NULL;
    tt_scale_free(&t->sc);
    t->use_scale = 0;
    t->locked = 0;
}

void camtrak_destroy(camtrak_tracker *t)
{
    if (t) {
        release_buffers(t);
        free(t);
    }
}

/* --------------------------------------------------------------- helpers */

static int next_pow2(int v)
{
    int p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

static int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Color veto: unavailable (-1) passes, otherwise it must be close to the selection. */
static int color_ok(float color)
{
    return color < 0.0f || color >= TT_COLOR_MIN;
}

static float psr_to_confidence(float psr)
{
    return clampf((psr - TT_PSR_FLOOR) / (TT_PSR_FULL - TT_PSR_FLOOR), 0.0f, 1.0f);
}

/* Cut, then preprocess, a patch centered on (cx, cy) into `dst`. */
static void extract_into(camtrak_tracker *t, const camtrak_frame *f, float cx, float cy,
                         float step_mul, float angle, float *dst)
{
    tt_sample_patch(f, &t->lut, cx, cy, t->step * step_mul, angle, t->pw, t->ph, dst);
    tt_preprocess(dst, t->window, t->pw * t->ph);
}

static void extract(camtrak_tracker *t, const camtrak_frame *f, float cx, float cy,
                    float step_mul, float angle)
{
    extract_into(t, f, cx, cy, step_mul, angle, t->patch);
}

/* One correlation attempt: where the target matched, how well, at what size. */
typedef struct {
    tt_peak pk;
    float   x, y;    /* matched target center, image pixels */
    float   ratio;   /* size tried, relative to the current size */
} tt_match;

/*
 * Correlate a window centered on (cx, cy) with the target assumed `ratio` times
 * its current size. Sampling a ratio-times-bigger region into the same patch
 * makes a target that grew by `ratio` look exactly like the learned one, so a
 * single filter serves every scale.
 */
static tt_match match_from_response(const camtrak_tracker *t, const float *response,
                                    float cx, float cy, float ratio)
{
    tt_match m;
    const float step = t->step * ratio;
    m.pk = tt_mosse_find_peak(response, t->pw, t->ph);
    /* the peak's offset from the patch center is the motion, in patch pixels */
    m.x = cx + (m.pk.x - (float)(t->pw / 2)) * step;
    m.y = cy + (m.pk.y - (float)(t->ph / 2)) * step;
    m.ratio = ratio;
    return m;
}

static tt_match detect_with(camtrak_tracker *t, tt_mosse *model, const camtrak_frame *f,
                            float cx, float cy, float ratio)
{
    extract(t, f, cx, cy, ratio, 0.0f);
    tt_mosse_respond(model, t->patch, t->response);
    return match_from_response(t, t->response, cx, cy, ratio);
}

static tt_match detect_at(camtrak_tracker *t, const camtrak_frame *f, float cx, float cy,
                          float ratio)
{
    return detect_with(t, &t->mosse, f, cx, cy, ratio);
}

/* Change the target size to `scale` x the lock-time size, within sane limits. */
static void set_scale(camtrak_tracker *t, const camtrak_frame *f, float scale)
{
    const float min_scale = TT_MIN_SIDE / fminf(t->base_w, t->base_h);
    const float max_scale = fminf((float)f->width / t->base_w, (float)f->height / t->base_h);
    t->scale = clampf(scale, min_scale, fmaxf(min_scale, max_scale));
    t->step = t->base_step * t->scale;
    t->tw = t->base_w * t->scale;
    t->th = t->base_h * t->scale;
}

static void learn_at(camtrak_tracker *t, const camtrak_frame *f, float cx, float cy)
{
    extract(t, f, cx, cy, 1.0f, 0.0f);
    tt_mosse_update(&t->mosse, t->patch);
}

static void write_result(const camtrak_tracker *t, float x, float y, float psr,
                         camtrak_status status, camtrak_result *out)
{
    out->box.x = x - 0.5f * t->tw;
    out->box.y = y - 0.5f * t->th;
    out->box.w = t->tw;
    out->box.h = t->th;
    out->psr = psr;
    out->confidence = psr_to_confidence(psr);
    out->status = status;
}

/* ------------------------------------------------------------------ lock */

static int box_valid(const camtrak_frame *f, camtrak_box b)
{
    float cx, cy;
    if (!isfinite(b.x) || !isfinite(b.y) || !isfinite(b.w) || !isfinite(b.h)) {
        return 0;
    }
    cx = b.x + 0.5f * b.w;
    cy = b.y + 0.5f * b.h;
    return b.w >= TT_MIN_BOX && b.h >= TT_MIN_BOX
        && cx >= 0.0f && cy >= 0.0f && cx <= (float)f->width && cy <= (float)f->height;
}

/*
 * Training samples for the first frame: small rotations of the target.
 * Learning from several slightly different views makes the filter tolerant to
 * small appearance changes instead of memorising one exact image (the MOSSE
 * paper uses random affine warps; a fixed set keeps runs reproducible).
 * No scaled copies: a filter taught that +-5% size looks the same could not
 * tell the scale search which size is right.
 */
static const float k_init_warps[][2] = {
    /* angle (rad), scale */
    { 0.00f, 1.00f }, {  0.05f, 1.00f }, { -0.05f, 1.00f }, {  0.10f, 1.00f },
    { -0.10f, 1.00f }, {  0.15f, 1.00f }, { -0.15f, 1.00f }, { 0.00f, 1.00f },
};

static void start_tracking_at(camtrak_tracker *t, float x, float y)
{
    tt_kalman_init(&t->kf, x, y, t->p.kalman_q, t->p.kalman_r);
    t->cx = t->good_x = x;
    t->cy = t->good_y = y;
    t->misses = 0;
    t->lost = 0;
}

camtrak_error camtrak_lock(camtrak_tracker *t, const camtrak_frame *frame, camtrak_box box)
{
    float win_w, win_h, sigma;
    size_t n, i;

    if (!t || !tt_frame_valid(frame) || !box_valid(frame, box)) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    release_buffers(t);

    /* Search window = padding x target. Scale it so its longest side becomes
     * template_size patch pixels, then round each side up to a power of two
     * for the radix-2 FFT. Every target, tiny or huge, costs the same. */
    win_w = box.w * t->p.padding;
    win_h = box.h * t->p.padding;
    t->base_step = t->step = fmaxf(win_w, win_h) / (float)t->p.template_size;
    t->base_w = t->tw = box.w;
    t->base_h = t->th = box.h;
    t->scale = 1.0f;
    t->pw = clamp_int(next_pow2((int)ceilf(win_w / t->step)), TT_MIN_PATCH, t->p.template_size);
    t->ph = clamp_int(next_pow2((int)ceilf(win_h / t->step)), TT_MIN_PATCH, t->p.template_size);
    n = (size_t)t->pw * (size_t)t->ph;

    /* Re-detection grid: neighboring windows overlap by half, so a target
     * anywhere lands in the well-lit middle of at least one window. */
    t->base_spacing = 0.5f * (float)(t->pw < t->ph ? t->pw : t->ph) * t->step;

    /* Gaussian width tied to target size, converted into patch pixels. */
    sigma = t->p.sigma_factor * sqrtf(box.w * box.h) / t->step;

    t->window   = malloc(sizeof(float) * n);
    t->patch    = malloc(sizeof(float) * n);
    t->response = malloc(sizeof(float) * n);
    t->use_scale = t->p.scale_step > 1.0f;
    if (!t->window || !t->patch || !t->response
        || tt_mosse_init(&t->mosse, t->pw, t->ph, sigma, t->p.learning_rate) != 0
        || tt_mosse_init(&t->anchor, t->pw, t->ph, sigma, t->p.learning_rate) != 0
        || (t->use_scale && tt_scale_init(&t->sc, box.w, box.h, TT_SCALE_SAMPLES,
                                          t->p.scale_step, TT_SCALE_LR) != 0)) {
        release_buffers(t);
        return CAMTRAK_ERR_NO_MEMORY;
    }
    tt_make_hann(t->pw, t->ph, t->window);

    /* which channel to track in: brightness, or a color projection that
     * separates this target from its surroundings better */
    t->channel_gain = t->p.adaptive_color
        ? tt_choose_channel(frame, box.x + 0.5f * box.w, box.y + 0.5f * box.h, box.w, box.h, &t->lut)
        : (tt_make_log_lut(&t->lut), 1.0f);

    start_tracking_at(t, box.x + 0.5f * box.w, box.y + 0.5f * box.h);

    for (i = 0; i < sizeof(k_init_warps) / sizeof(k_init_warps[0]); i++) {
        extract(t, frame, t->cx, t->cy, k_init_warps[i][1], k_init_warps[i][0]);
        tt_mosse_add_sample(&t->mosse, t->patch);
    }
    tt_mosse_finish_init(&t->mosse, t->p.lambda);
    tt_mosse_copy_model(&t->anchor, &t->mosse);
    tt_color_learn(&t->color, frame, box);
    if (t->use_scale) {
        tt_scale_train(&t->sc, frame, &t->lut, t->cx, t->cy, t->tw, t->th);
    }

    t->locked = 1;
    return CAMTRAK_OK;
}

/* ------------------------------------------------------- LOST: re-detect */

/*
 * Cell k of spiral ring r: the cells (i, j) with max(|i|, |j|) == r, walked
 * clockwise in four sides of 2r cells each. Ring 0 is the single center cell.
 */
static void ring_cell(int r, int k, int *i, int *j)
{
    int side, off;
    if (r == 0) {
        *i = *j = 0;
        return;
    }
    side = k / (2 * r);
    off  = k % (2 * r);
    switch (side) {
    case 0:  *i = -r + off; *j = -r;       break;   /* top, left to right   */
    case 1:  *i = r;        *j = -r + off; break;   /* right, top to bottom */
    case 2:  *i = r - off;  *j = r;        break;   /* bottom, right to left */
    default: *i = -r;       *j = r - off;  break;   /* left, bottom to top  */
    }
}

static void enter_lost(camtrak_tracker *t)
{
    t->lost = 1;
    t->ring = 1;      /* ring 0 (the last good spot) is checked every frame anyway */
    t->ring_idx = 0;
    t->sweep_scale = 0;
    t->lost_frames = 0;
}

static camtrak_box match_box(const camtrak_tracker *t, tt_match m)
{
    camtrak_box b;
    b.w = t->tw * m.ratio;
    b.h = t->th * m.ratio;
    b.x = m.x - 0.5f * b.w;
    b.y = m.y - 0.5f * b.h;
    return b;
}

/* Fraction of `b`'s area that lies inside the frame. */
static float inside_fraction(const camtrak_frame *f, camtrak_box b)
{
    const float iw = fminf(b.x + b.w, (float)f->width) - fmaxf(b.x, 0.0f);
    const float ih = fminf(b.y + b.h, (float)f->height) - fmaxf(b.y, 0.0f);
    return (iw > 0.0f && ih > 0.0f) ? (iw * ih) / (b.w * b.h) : 0.0f;
}

/*
 * Sanity checks every switch to a new position must pass, whoever proposed
 * it: mostly inside the frame (the border is where false matches pile up),
 * and standing out from its surroundings like the selection did (a "match"
 * on a patch of rock or forest blends in).
 */
static int candidate_plausible(const camtrak_tracker *t, const camtrak_frame *f, camtrak_box b)
{
    return inside_fraction(f, b) >= TT_MIN_INSIDE && tt_color_stands_out(&t->color, f, b);
}

/* Re-acquisition decision for a candidate found by the search. */
static int reacquire_ok(const camtrak_tracker *t, const camtrak_frame *f, tt_match m)
{
    float color;
    camtrak_box b;
    if (m.pk.psr < TT_REACQ_PSR_STRONG_COLOR) {
        return 0;
    }
    b = match_box(t, m);
    if (!candidate_plausible(t, f, b)) {
        return 0;
    }
    color = tt_color_match(&t->color, f, b);
    /* shape alone must be very sure; with a clear color match less is needed */
    return (m.pk.psr >= t->p.psr_reacquire && color_ok(color)) || color >= TT_COLOR_STRONG;
}

/* Keep the better candidate: an acceptable one always beats an unacceptable one. */
static void keep_best(camtrak_tracker *t, const camtrak_frame *f, tt_match *best, int *best_ok, tt_match m)
{
    const int ok = reacquire_ok(t, f, m);
    if ((ok && !*best_ok) || (ok == *best_ok && m.pk.psr > best->pk.psr)) {
        *best = m;
        *best_ok = ok;
    }
}

/*
 * Try the last good position (at every re-detection size) plus
 * `reacquire_budget` grid cells, continuing the spiral where the previous
 * frame stopped. Each full sweep of the frame uses the next size in
 * k_redetect_scales, so a target that came back bigger or smaller is found too.
 *
 * Frames alternate between the current model and the frozen lock-time anchor.
 * The current model knows the latest look of the target, but may have been
 * contaminated while the target was being covered up; the anchor cannot be.
 */
static tt_match redetect(camtrak_tracker *t, const camtrak_frame *f, int *found)
{
    tt_mosse *model = (t->lost_frames & 1) ? &t->anchor : &t->mosse;
    const float sweep_ratio = k_redetect_scales[t->sweep_scale];
    const float spacing = t->base_spacing * t->scale * sweep_ratio;
    const int frame_rings = (int)ceilf(fmaxf((float)f->width, (float)f->height) / spacing) + 1;
    /* Search radius grows with time lost: a target that just vanished is
     * almost certainly nearby, so re-check the neighborhood every frame first
     * and widen by one ring every two frames until the whole frame is covered. */
    const int reach = t->lost_frames / 2 + 2;
    const int max_ring = reach < frame_rings ? reach : frame_rings;
    const int max_attempts = t->p.reacquire_budget * TT_MAX_SKIPS_PER_WINDOW;
    tt_match best = detect_with(t, model, f, t->good_x, t->good_y, 1.0f);
    int tried = 0, attempts = 0, s;

    *found = reacquire_ok(t, f, best);
    for (s = 1; s < TT_N_REDETECT_SCALES; s++) {
        keep_best(t, f, &best, found, detect_with(t, model, f, t->good_x, t->good_y, k_redetect_scales[s]));
    }
    /* also the size the user originally selected, in case the size estimate drifted */
    if (fabsf(t->scale - 1.0f) > 0.03f) {
        keep_best(t, f, &best, found, detect_with(t, model, f, t->good_x, t->good_y, 1.0f / t->scale));
    }

    while (tried < t->p.reacquire_budget && attempts < max_attempts) {
        int i, j;
        float x, y;
        ring_cell(t->ring, t->ring_idx, &i, &j);
        if (++t->ring_idx >= 8 * t->ring) {
            t->ring_idx = 0;
            if (++t->ring > max_ring) {
                /* whole frame covered at this size: next sweep, next size */
                t->ring = 1;
                t->sweep_scale = (t->sweep_scale + 1) % TT_N_REDETECT_SCALES;
            }
        }
        attempts++;

        x = t->good_x + (float)i * spacing;
        y = t->good_y + (float)j * spacing;
        if (x < 0.0f || y < 0.0f || x > (float)f->width || y > (float)f->height) {
            continue;   /* off-frame cell: free to skip */
        }
        tried++;
        keep_best(t, f, &best, found, detect_with(t, model, f, x, y, sweep_ratio));
    }
    t->dbg.used_anchor = model == &t->anchor;
    t->dbg.search_rings = max_ring;
    t->dbg.windows_tried = tried + TT_N_REDETECT_SCALES + (fabsf(t->scale - 1.0f) > 0.03f);
    t->dbg.pred_x = t->good_x;
    t->dbg.pred_y = t->good_y;
    t->lost_frames++;
    return best;
}

static void update_lost(camtrak_tracker *t, const camtrak_frame *f, camtrak_result *out)
{
    const int used_anchor = t->lost_frames & 1;   /* same rule redetect() uses */
    int found;
    const tt_match m = redetect(t, f, &found);

    if (found) {
        /* Found by the clean anchor: the current model was the problem, replace it. */
        if (used_anchor) {
            tt_mosse_copy_model(&t->mosse, &t->anchor);
        }
        /* Strong, unambiguous match: restart tracking there, at the size it matched. */
        set_scale(t, f, t->scale * m.ratio);
        start_tracking_at(t, clampf(m.x, 0.0f, (float)f->width), clampf(m.y, 0.0f, (float)f->height));
        learn_at(t, f, t->cx, t->cy);
        t->dbg.learned = 1;
        write_result(t, t->cx, t->cy, m.pk.psr, CAMTRAK_TRACKING, out);
        return;
    }
    /* Stay honest: report the last known box, flagged LOST. */
    write_result(t, t->good_x, t->good_y, m.pk.psr, CAMTRAK_LOST, out);
}

/* ------------------------------------------------- TRACKING / COASTING */

static void update_tracking(camtrak_tracker *t, const camtrak_frame *f, camtrak_result *out)
{
    tt_match m;
    tt_peak pk;
    camtrak_status status;

    /* 1. predict where the target is now and search there, not at the old spot */
    tt_kalman_predict(&t->kf);
    t->dbg.pred_x = t->kf.x.p;
    t->dbg.pred_y = t->kf.y.p;
    t->dbg.windows_tried = 1;

    /* 2. correlate */
    m = detect_at(t, f, t->kf.x.p, t->kf.y.p, 1.0f);
    pk = m.pk;

    /* 3. a confident peak is a measurement; a weak one is ignored and we coast */
    if (pk.psr >= t->p.psr_track) {
        tt_kalman_correct(&t->kf, m.x, m.y);
        /* position known: now ask the scale filter how big the target is */
        if (t->use_scale) {
            const float factor = tt_scale_detect(&t->sc, f, &t->lut, t->kf.x.p, t->kf.y.p,
                                                 t->tw, t->th);
            t->dbg.scale_factor = factor;
            set_scale(t, f, t->scale * clampf(factor, 1.0f - TT_SCALE_MAX_CHANGE,
                                              1.0f + TT_SCALE_MAX_GROWTH));
        }
        t->misses = 0;
        status = CAMTRAK_TRACKING;
    } else {
        tt_kalman_damp(&t->kf, t->p.coast_damping);
        t->misses++;
        status = CAMTRAK_COASTING;
    }
    t->kf.x.p = t->cx = clampf(t->kf.x.p, 0.0f, (float)f->width);
    t->kf.y.p = t->cy = clampf(t->kf.y.p, 0.0f, (float)f->height);

    if (status == CAMTRAK_TRACKING) {
        t->good_x = t->cx;
        t->good_y = t->cy;
        /* 4. adapt the appearance model, but only on a confident match: learning
         *    from an occluder is how a tracker "becomes" the background. */
        if (pk.psr >= t->p.psr_update) {
            learn_at(t, f, t->cx, t->cy);
            t->dbg.learned = 1;
            if (t->use_scale) {
                tt_scale_train(&t->sc, f, &t->lut, t->cx, t->cy, t->tw, t->th);
            }
        }
    } else if (t->misses >= t->p.lost_after) {
        /* 5. coasted too long: the prediction is no longer worth reporting */
        enter_lost(t);
        write_result(t, t->good_x, t->good_y, pk.psr, CAMTRAK_LOST, out);
        return;
    }
    write_result(t, t->cx, t->cy, pk.psr, status, out);
}

camtrak_error camtrak_update(camtrak_tracker *t, const camtrak_frame *frame, camtrak_result *out)
{
    if (!t || !out || !tt_frame_valid(frame)) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    memset(&t->dbg, 0, sizeof(t->dbg));
    t->dbg.scale_factor = 1.0f;
    if (t->lost) {
        update_lost(t, frame, out);
    } else {
        update_tracking(t, frame, out);
    }
    t->dbg.vel_x = t->kf.x.v;
    t->dbg.vel_y = t->kf.y.v;
    t->dbg.scale = t->scale;
    t->dbg.channel_gain = t->channel_gain;
    t->dbg.misses = t->misses;
    t->dbg.lost_frames = t->lost ? t->lost_frames : 0;
    return CAMTRAK_OK;
}

/* Size of `box` relative to the current box (geometric mean of both sides). */
static float box_ratio(const camtrak_tracker *t, camtrak_box box)
{
    return sqrtf((box.w / t->tw) * (box.h / t->th));
}

camtrak_error camtrak_verify(camtrak_tracker *t, const camtrak_frame *frame, camtrak_box box,
                           camtrak_verdict *out)
{
    float cx, cy, ratio, best = 0.0f;
    float *psr = &best;
    tt_match a, b;

    if (!t || !out || !tt_frame_valid(frame) || !box_valid(frame, box)) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    cx = box.x + 0.5f * box.w;
    cy = box.y + 0.5f * box.h;
    ratio = box_ratio(t, box);
    a = detect_with(t, &t->mosse, frame, cx, cy, ratio);
    b = detect_with(t, &t->anchor, frame, cx, cy, ratio);
    /* the peak must also be near the box center, not somewhere else in the window */
    *psr = 0.0f;
    if (hypotf(a.x - cx, a.y - cy) <= 0.5f * fminf(box.w, box.h)) *psr = a.pk.psr;
    if (hypotf(b.x - cx, b.y - cy) <= 0.5f * fminf(box.w, box.h)) *psr = fmaxf(*psr, b.pk.psr);
    out->psr = best;
    out->color = tt_color_match(&t->color, frame, box);
    out->accept = color_ok(out->color) && candidate_plausible(t, frame, box)
        && best >= (out->color >= TT_COLOR_STRONG ? TT_VERIFY_PSR_STRONG_COLOR : TT_VERIFY_PSR);
    return CAMTRAK_OK;
}

camtrak_error camtrak_set_box(camtrak_tracker *t, const camtrak_frame *frame, camtrak_box box,
                              int relearn)
{
    if (!t || !tt_frame_valid(frame) || !box_valid(frame, box)) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    set_scale(t, frame, t->scale * box_ratio(t, box));
    /* move the position, keep the motion model's velocity */
    t->kf.x.p = t->cx = box.x + 0.5f * box.w;
    t->kf.y.p = t->cy = box.y + 0.5f * box.h;
    if (!t->lost) {
        t->good_x = t->cx;
        t->good_y = t->cy;
    }
    if (relearn) {
        learn_at(t, frame, t->cx, t->cy);
        if (t->use_scale) {
            tt_scale_train(&t->sc, frame, &t->lut, t->cx, t->cy, t->tw, t->th);
        }
    }
    return CAMTRAK_OK;
}

camtrak_error camtrak_reacquire(camtrak_tracker *t, const camtrak_frame *frame, camtrak_box box)
{
    if (!t || !tt_frame_valid(frame) || !box_valid(frame, box)) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    set_scale(t, frame, t->scale * box_ratio(t, box));
    start_tracking_at(t, box.x + 0.5f * box.w, box.y + 0.5f * box.h);
    /* The target was confirmed here but may look quite different now (new
     * shot, new pose). Learn this view strongly so the correlation filter can
     * take over on its own; the lock-time anchor stays untouched. */
    {
        const float rate = t->mosse.rate;
        t->mosse.rate = TT_REACQUIRE_RATE;
        learn_at(t, frame, t->cx, t->cy);
        t->mosse.rate = rate;
    }
    if (t->use_scale) {
        tt_scale_train(&t->sc, frame, &t->lut, t->cx, t->cy, t->tw, t->th);
    }
    return CAMTRAK_OK;
}

camtrak_error camtrak_get_debug(const camtrak_tracker *t, camtrak_debug *out)
{
    if (!t || !out) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    *out = t->dbg;
    return CAMTRAK_OK;
}

camtrak_error camtrak_scene_cut(camtrak_tracker *t)
{
    if (!t) {
        return CAMTRAK_ERR_INVALID_ARG;
    }
    if (!t->locked) {
        return CAMTRAK_ERR_NOT_LOCKED;
    }
    enter_lost(t);
    /* The size from the previous shot means nothing now (the target may have
     * gone from far away to close up). Search around the size the user drew,
     * the best prior we have; otherwise a too-small box matches a *part* of
     * something, e.g. the dark trousers of a different skier. */
    t->scale = 1.0f;
    t->step = t->base_step;
    t->tw = t->base_w;
    t->th = t->base_h;
    return CAMTRAK_OK;
}
