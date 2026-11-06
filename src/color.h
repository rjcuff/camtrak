/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (c) 2026 Ryan Cuff. Commercial licenses: see LICENSING.md. */
/*
 * color.h - color model of the selected target (mean-shift style, Comaniciu
 * et al. 2003; the color half of Staple, Bertinetto et al. 2016).
 *
 * At lock time we take a coarse color histogram (64 colors) of the middle
 * of the box, which is mostly the target and little background. A candidate
 * box is compared with the Bhattacharyya coefficient: 1 = same colors,
 * 0 = no colors in common.
 *
 * Color ignores pose and shape completely, which is exactly what grayscale
 * correlation is bad at, so it is used to veto look-alikes (a red-jacketed
 * skier next to a black-jacketed one) when re-acquiring.
 */
#ifndef TT_COLOR_H
#define TT_COLOR_H

#include "camtrak/camtrak.h"

#define TT_COLOR_BINS 64

typedef struct {
    int   enabled;              /* 0 for grayscale input */
    float ref[TT_COLOR_BINS];   /* color histogram of the selection */
    float ref_contrast;         /* how much the selection blended into its surroundings */
} tt_color;

/* Learn from the box on a BGR frame. Gray frames leave the model disabled. */
void  tt_color_learn(tt_color *c, const camtrak_frame *f, camtrak_box box);

/* Color similarity of `box` to the selection, 0..1; -1 if unavailable. */
float tt_color_match(const tt_color *c, const camtrak_frame *f, camtrak_box box);

/* How alike the middle of `box` and its surroundings are, 0..1 (-1 if gray).
 * A skier on snow: low. A dark patch inside dark rocks: high. */
float tt_color_contrast(const camtrak_frame *f, camtrak_box box);

/* 1 if `box` stands out from its surroundings about as much as the selection
 * did. Real objects do; "matches" on a patch of background usually do not. */
int tt_color_stands_out(const tt_color *c, const camtrak_frame *f, camtrak_box box);

#endif
