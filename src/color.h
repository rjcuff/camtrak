/*
 * color.h - colour model of the selected target (mean-shift style, Comaniciu
 * et al. 2003; the colour half of Staple, Bertinetto et al. 2016).
 *
 * At lock time we take a coarse colour histogram (64 colours) of the middle
 * of the box, which is mostly the target and little background. A candidate
 * box is compared with the Bhattacharyya coefficient: 1 = same colours,
 * 0 = no colours in common.
 *
 * Colour ignores pose and shape completely, which is exactly what grayscale
 * correlation is bad at, so it is used to veto look-alikes (a red-jacketed
 * skier next to a black-jacketed one) when re-acquiring.
 */
#ifndef TT_COLOR_H
#define TT_COLOR_H

#include "tether/tether.h"

#define TT_COLOR_BINS 64

typedef struct {
    int   enabled;              /* 0 for grayscale input */
    float ref[TT_COLOR_BINS];   /* colour histogram of the selection */
    float ref_contrast;         /* how much the selection blended into its surroundings */
} tt_color;

/* Learn from the box on a BGR frame. Gray frames leave the model disabled. */
void  tt_color_learn(tt_color *c, const tether_frame *f, tether_box box);

/* Colour similarity of `box` to the selection, 0..1; -1 if unavailable. */
float tt_color_match(const tt_color *c, const tether_frame *f, tether_box box);

/* How alike the middle of `box` and its surroundings are, 0..1 (-1 if gray).
 * A skier on snow: low. A dark patch inside dark rocks: high. */
float tt_color_contrast(const tether_frame *f, tether_box box);

/* 1 if `box` stands out from its surroundings about as much as the selection
 * did. Real objects do; "matches" on a patch of background usually do not. */
int tt_color_stands_out(const tt_color *c, const tether_frame *f, tether_box box);

#endif
