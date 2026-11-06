/*
 * tether.h - public C API of the Tether visual object tracker.
 *
 * Usage:
 *   tether_tracker *t = tether_create(NULL);          // NULL = default params
 *   tether_lock(t, &first_frame, box);                // user-drawn box
 *   for each later frame:
 *       tether_result r;
 *       tether_update(t, &frame, &r);                 // r.box, r.confidence, r.status
 *   tether_destroy(t);
 *
 * Coordinates are continuous pixel coordinates: pixel (i, j) covers the square
 * [i, i+1) x [j, j+1). A box {x, y, w, h} covers [x, x+w) x [y, y+h).
 *
 * Memory: tether_lock() allocates everything the tracker needs. tether_update()
 * performs no heap allocation.
 */
#ifndef TETHER_H
#define TETHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(TETHER_SHARED)
#  ifdef TETHER_BUILDING
#    define TETHER_API __declspec(dllexport)
#  else
#    define TETHER_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(TETHER_SHARED)
#  define TETHER_API __attribute__((visibility("default")))
#else
#  define TETHER_API
#endif

/* Opaque tracker handle. */
typedef struct tether_tracker tether_tracker;

/* Axis-aligned box: top-left corner and size, in pixels. */
typedef struct {
    float x, y, w, h;
} tether_box;

/* Pixel format. The numeric value is the number of bytes per pixel. */
typedef enum {
    TETHER_FORMAT_GRAY8 = 1, /* one byte per pixel */
    TETHER_FORMAT_BGR8  = 3  /* OpenCV default channel order */
} tether_format;

/* A borrowed view of an image. Tether never keeps the pointer after a call returns. */
typedef struct {
    const uint8_t *data;
    int width;
    int height;
    int stride;              /* bytes between the starts of consecutive rows */
    tether_format format;
} tether_frame;

typedef enum {
    TETHER_TRACKING = 0,  /* confident fix on the target */
    TETHER_COASTING = 1,  /* low confidence; position predicted by motion model */
    TETHER_LOST     = 2   /* target lost; box is the last known position */
} tether_status;

typedef struct {
    tether_box box;
    float confidence;     /* 0..1, derived from psr */
    float psr;            /* raw peak-to-sidelobe ratio of this frame's best match */
    tether_status status;
} tether_result;

typedef enum {
    TETHER_OK              =  0,
    TETHER_ERR_INVALID_ARG = -1,
    TETHER_ERR_NO_MEMORY   = -2,
    TETHER_ERR_NOT_LOCKED  = -3
} tether_error;

/* Tuning parameters. Start from tether_default_params() and change what you need. */
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
} tether_params;

TETHER_API tether_params tether_default_params(void);

/* Returns NULL if params are invalid or memory runs out. params may be NULL. */
TETHER_API tether_tracker *tether_create(const tether_params *params);
TETHER_API void tether_destroy(tether_tracker *t);

/* Start (or restart) tracking the object inside `box` on `frame`. */
TETHER_API tether_error tether_lock(tether_tracker *t, const tether_frame *frame, tether_box box);

/* Track into the next frame. Writes the result to *out. */
TETHER_API tether_error tether_update(tether_tracker *t, const tether_frame *frame, tether_result *out);

/*
 * Tell the tracker the next frame is a different shot (edited video, camera
 * switch). Motion prediction is meaningless across a cut, so the tracker goes
 * straight to LOST and starts searching, keeping the target's appearance.
 */
TETHER_API tether_error tether_scene_cut(tether_tracker *t);

/* Verdict on a candidate box, see tether_verify. */
typedef struct {
    float psr;      /* shape match: best PSR of current and lock-time model (~15+ clear) */
    float color;    /* colour match vs the selected target, 1 = as at selection;
                       -1 if unavailable (grayscale input) */
    int   accept;   /* 1 if both are good enough to switch to this candidate */
} tether_verdict;

/*
 * Hooks for an external detector or ML tracker (a "second opinion").
 *
 * tether_verify: does `box` on `frame` look like the selected target, in
 * shape (correlation) and colour? Does not change the tracker.
 *
 * tether_reacquire: put the tracker on `box` (position and size), keeping the
 * learned appearance, and resume TRACKING from the next frame. Use after a
 * candidate passed tether_verify.
 */
TETHER_API tether_error tether_verify(tether_tracker *t, const tether_frame *frame,
                                      tether_box box, tether_verdict *out);
TETHER_API tether_error tether_reacquire(tether_tracker *t, const tether_frame *frame,
                                         tether_box box);

/*
 * Diagnostics: what the tracker did inside the last tether_update().
 * For logging and tuning only; fields may change between versions.
 */
typedef struct {
    float pred_x, pred_y;   /* where the motion model expected the target (search centre) */
    float vel_x, vel_y;     /* motion model velocity, px/frame */
    float scale;            /* box size relative to the locked box */
    float scale_factor;     /* raw size change suggested by the scale filter (1 = none) */
    int   misses;           /* consecutive frames without a confident match */
    int   lost_frames;      /* frames spent LOST so far */
    int   search_rings;     /* LOST: current search radius, in grid rings */
    int   used_anchor;      /* LOST: 1 if this frame searched with the lock-time model */
    int   learned;          /* 1 if the appearance model was updated this frame */
    int   windows_tried;    /* correlation windows evaluated this frame */
} tether_debug;

TETHER_API tether_error tether_get_debug(const tether_tracker *t, tether_debug *out);

TETHER_API const char *tether_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TETHER_H */
