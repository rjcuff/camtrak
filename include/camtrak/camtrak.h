/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
/*
 * camtrak.h - public C API of the Camtrak visual object tracker.
 *
 * Usage:
 *   camtrak_tracker *t = camtrak_create(NULL);          // NULL = default params
 *   camtrak_lock(t, &first_frame, box);                // user-drawn box
 *   for each later frame:
 *       camtrak_result r;
 *       camtrak_update(t, &frame, &r);                 // r.box, r.confidence, r.status
 *   camtrak_destroy(t);
 *
 * Coordinates are continuous pixel coordinates: pixel (i, j) covers the square
 * [i, i+1) x [j, j+1). A box {x, y, w, h} covers [x, x+w) x [y, y+h).
 *
 * Memory: camtrak_lock() allocates everything the tracker needs. camtrak_update()
 * performs no heap allocation.
 */
#ifndef CAMTRAK_H
#define CAMTRAK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(CAMTRAK_SHARED)
#  ifdef CAMTRAK_BUILDING
#    define CAMTRAK_API __declspec(dllexport)
#  else
#    define CAMTRAK_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(CAMTRAK_SHARED)
#  define CAMTRAK_API __attribute__((visibility("default")))
#else
#  define CAMTRAK_API
#endif

/* Opaque tracker handle. */
typedef struct camtrak_tracker camtrak_tracker;

/* Axis-aligned box: top-left corner and size, in pixels. */
typedef struct {
    float x, y, w, h;
} camtrak_box;

/* Pixel format. The numeric value is the number of bytes per pixel. */
typedef enum {
    CAMTRAK_FORMAT_GRAY8 = 1, /* one byte per pixel */
    CAMTRAK_FORMAT_BGR8  = 3  /* OpenCV default channel order */
} camtrak_format;

/* A borrowed view of an image. Camtrak never keeps the pointer after a call returns. */
typedef struct {
    const uint8_t *data;
    int width;
    int height;
    int stride;              /* bytes between the starts of consecutive rows */
    camtrak_format format;
} camtrak_frame;

typedef enum {
    CAMTRAK_TRACKING = 0,  /* confident fix on the target */
    CAMTRAK_COASTING = 1,  /* low confidence; position predicted by motion model */
    CAMTRAK_LOST     = 2   /* target lost; box is the last known position */
} camtrak_status;

typedef struct {
    camtrak_box box;
    float confidence;     /* 0..1, derived from psr */
    float psr;            /* raw peak-to-sidelobe ratio of this frame's best match */
    camtrak_status status;
} camtrak_result;

typedef enum {
    CAMTRAK_OK              =  0,
    CAMTRAK_ERR_INVALID_ARG = -1,
    CAMTRAK_ERR_NO_MEMORY   = -2,
    CAMTRAK_ERR_NOT_LOCKED  = -3
} camtrak_error;

/* Tuning parameters. Start from camtrak_default_params() and change what you need. */
typedef struct {
    float padding;        /* search window = padding x target size (default 2.5) */
    int   template_size;  /* longest side of the internal patch, power of two 16..256 (default 64) */
    float learning_rate;  /* how fast the appearance model adapts, 0..1 (default 0.125) */
    float sigma_factor;   /* width of the desired Gaussian peak, relative to target size (default 0.1) */
    float lambda;         /* regularisation, as a fraction of average spectral energy (default 0.01) */
    float psr_update;     /* learn appearance only when PSR >= this (default 12) */
    float psr_track;      /* accept a measurement when PSR >= this, else COASTING (default 11) */
    float kalman_q;       /* motion model: expected acceleration variance, px^2 (default 0.5) */
    float kalman_r;       /* motion model: measurement noise variance, px^2 (default 1) */
    float coast_damping;  /* velocity multiplier per coasting frame (default 0.95) */
    int   lost_after;     /* consecutive misses before reporting LOST (default 20) */
    float psr_reacquire;  /* PSR needed to re-acquire a lost target (default 14) */
    int   reacquire_budget; /* search windows tried per LOST frame (default 24) */
    float scale_step;     /* size estimation: ratio between the 17 sizes the scale filter
                             compares (default 1.04; 0 or 1 = fixed box size) */
    int   adaptive_color; /* BGR input: 1 = at lock, track in the color projection that
                             best separates the target from its surroundings when that
                             beats plain brightness (a red car on a brown field);
                             0 = always brightness (default 1) */
} camtrak_params;

CAMTRAK_API camtrak_params camtrak_default_params(void);

/* Returns NULL if params are invalid or memory runs out. params may be NULL. */
CAMTRAK_API camtrak_tracker *camtrak_create(const camtrak_params *params);
CAMTRAK_API void camtrak_destroy(camtrak_tracker *t);

/* Start (or restart) tracking the object inside `box` on `frame`. */
CAMTRAK_API camtrak_error camtrak_lock(camtrak_tracker *t, const camtrak_frame *frame, camtrak_box box);

/* Track into the next frame. Writes the result to *out. */
CAMTRAK_API camtrak_error camtrak_update(camtrak_tracker *t, const camtrak_frame *frame, camtrak_result *out);

/*
 * Tell the tracker the next frame is a different shot (edited video, camera
 * switch). Motion prediction is meaningless across a cut, so the tracker goes
 * straight to LOST and starts searching, keeping the target's appearance.
 */
CAMTRAK_API camtrak_error camtrak_scene_cut(camtrak_tracker *t);

/* Verdict on a candidate box, see camtrak_verify. */
typedef struct {
    float psr;      /* shape match: best PSR of current and lock-time model (~15+ clear) */
    float color;    /* color match vs the selected target, 1 = as at selection;
                       -1 if unavailable (grayscale input) */
    int   accept;   /* 1 if both are good enough to switch to this candidate */
} camtrak_verdict;

/*
 * Hooks for an external detector or ML tracker (a "second opinion").
 *
 * camtrak_verify: does `box` on `frame` look like the selected target, in
 * shape (correlation) and color? Does not change the tracker.
 *
 * camtrak_reacquire: put the tracker on `box` (position and size), keeping the
 * learned appearance, and resume TRACKING from the next frame. Use after a
 * candidate passed camtrak_verify.
 */
CAMTRAK_API camtrak_error camtrak_verify(camtrak_tracker *t, const camtrak_frame *frame,
                                      camtrak_box box, camtrak_verdict *out);
CAMTRAK_API camtrak_error camtrak_reacquire(camtrak_tracker *t, const camtrak_frame *frame,
                                         camtrak_box box);

/*
 * Diagnostics: what the tracker did inside the last camtrak_update().
 * For logging and tuning only; fields may change between versions.
 */
typedef struct {
    float pred_x, pred_y;   /* where the motion model expected the target (search center) */
    float vel_x, vel_y;     /* motion model velocity, px/frame */
    float scale;            /* box size relative to the locked box */
    float scale_factor;     /* raw size change suggested by the scale filter (1 = none) */
    int   misses;           /* consecutive frames without a confident match */
    int   lost_frames;      /* frames spent LOST so far */
    int   search_rings;     /* LOST: current search radius, in grid rings */
    int   used_anchor;      /* LOST: 1 if this frame searched with the lock-time model */
    int   learned;          /* 1 if the appearance model was updated this frame */
    int   windows_tried;    /* correlation windows evaluated this frame */
    float channel_gain;     /* lock: how much better the chosen channel separates the
                               target than brightness (1 = brightness is used) */
} camtrak_debug;

CAMTRAK_API camtrak_error camtrak_get_debug(const camtrak_tracker *t, camtrak_debug *out);

CAMTRAK_API const char *camtrak_version(void);

#ifdef __cplusplus
}
#endif

#endif /* CAMTRAK_H */
