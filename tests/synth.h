/*
 * synth.h - deterministic synthetic video for testing trackers.
 *
 * A textured target moves along a smooth Lissajous path over a panning
 * textured background, with per-frame noise. The exact ground-truth box is
 * known for every frame. Occlusion and drop-out events come with the LOST /
 * re-acquire phase.
 */
#ifndef TT_SYNTH_H
#define TT_SYNTH_H

#include "tether/tether.h"

typedef struct {
    int       w, h;          /* frame size */
    int       tsz;           /* target side length */
    int       noise;         /* per-pixel noise amplitude (+/-) */
    uint8_t  *frame;         /* current frame, gray, stride = w */
    uint8_t  *bg;            /* background texture, (w + pan) x h */
    int       bg_w;
    uint8_t  *target;        /* tsz x tsz target texture */
    unsigned  rng;
    int       index;         /* index of the frame currently in `frame` */
    tether_box truth;        /* ground truth for `frame` */
    int       occ_start;     /* frames [occ_start, occ_end] hide the target; -1 = none */
    int       occ_end;
    int       occluded;      /* 1 if the target is hidden in `frame` */
    double    growth;        /* target size multiplier per frame (1 = constant size) */
} tt_synth;

int  tt_synth_init(tt_synth *s, int w, int h, int target_size, int noise, unsigned seed);
void tt_synth_free(tt_synth *s);
/* Hide the target completely behind a flat, noisy block during [start, end]. */
void tt_synth_set_occlusion(tt_synth *s, int start, int end);
/* Render frame number `index` into s->frame and set s->truth. */
void tt_synth_render(tt_synth *s, int index);
tether_frame tt_synth_frame(const tt_synth *s);

#endif
