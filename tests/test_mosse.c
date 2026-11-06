/*
 * test_mosse: phase-1 tracking accuracy and speed on synthetic video.
 *
 * Accuracy: 320x240, 24 px target, panning background, +/-12 noise, 300 frames,
 *           run on both GRAY8 and BGR8 input. Fails if the center error regresses.
 * Speed:    1280x720 frames; reports ms per camtrak_update(). In optimized
 *           builds it fails above the 0.9 ms/frame of the C++ prototype.
 */
#include "camtrak/camtrak.h"
#include "synth.h"
#include "test_util.h"

#include <math.h>
#include <stdlib.h>

#define FRAMES 300

static float center_error(camtrak_box a, camtrak_box b)
{
    const float dx = (a.x + 0.5f * a.w) - (b.x + 0.5f * b.w);
    const float dy = (a.y + 0.5f * a.h) - (b.y + 0.5f * b.h);
    return sqrtf(dx * dx + dy * dy);
}

/* Copy the gray synthetic frame into a BGR buffer (B=G=R) so the color path is exercised. */
static camtrak_frame to_bgr(const tt_synth *s, uint8_t *bgr)
{
    camtrak_frame f;
    int i;
    for (i = 0; i < s->w * s->h; i++) {
        bgr[3 * i] = bgr[3 * i + 1] = bgr[3 * i + 2] = s->frame[i];
    }
    f.data = bgr;
    f.width = s->w;
    f.height = s->h;
    f.stride = 3 * s->w;
    f.format = CAMTRAK_FORMAT_BGR8;
    return f;
}

static void accuracy(int use_bgr)
{
    tt_synth s;
    camtrak_tracker *t = camtrak_create(NULL);
    uint8_t *bgr = NULL;
    double sum_err = 0.0;
    float max_err = 0.0f;
    int within5 = 0, i;
    camtrak_frame f;

    CHECK(t != NULL);
    CHECK(tt_synth_init(&s, 320, 240, 24, 12, 1u) == 0);
    if (use_bgr) {
        bgr = malloc((size_t)3 * s.w * s.h);
        CHECK(bgr != NULL);
    }
    if (!t || (use_bgr && !bgr)) {
        camtrak_destroy(t);
        free(bgr);
        tt_synth_free(&s);
        return;
    }

    tt_synth_render(&s, 0);
    f = use_bgr ? to_bgr(&s, bgr) : tt_synth_frame(&s);
    CHECK(camtrak_lock(t, &f, s.truth) == CAMTRAK_OK);

    for (i = 1; i < FRAMES; i++) {
        camtrak_result r;
        float e;
        tt_synth_render(&s, i);
        f = use_bgr ? to_bgr(&s, bgr) : tt_synth_frame(&s);
        CHECK(camtrak_update(t, &f, &r) == CAMTRAK_OK);
        e = center_error(r.box, s.truth);
        sum_err += e;
        if (e > max_err) max_err = e;
        if (e <= 5.0f) within5++;
    }

    {
        const double mean = sum_err / (FRAMES - 1);
        const double prec5 = (double)within5 / (FRAMES - 1);
        printf("accuracy %-4s  mean err %.2f px  max err %.2f px  precision@5px %.1f%%\n",
               use_bgr ? "BGR" : "GRAY", mean, max_err, 100.0 * prec5);
        CHECK(mean < 1.5);
        CHECK(prec5 >= 0.98);
    }

    camtrak_destroy(t);
    free(bgr);
    tt_synth_free(&s);
}

/*
 * Occlusion: the target vanishes for 50 frames (longer than lost_after) and
 * reappears elsewhere. Gates:
 *   - accurate before the occlusion
 *   - honest: never claims TRACKING on the wrong spot while hidden, and says
 *     LOST before the occlusion ends
 *   - re-acquires within REACQ_MAX frames of the target reappearing
 *   - accurate again afterwards
 */
#define OCC_START  120
#define OCC_END    170
#define REACQ_MAX  30

static void occlusion(void)
{
    tt_synth s;
    camtrak_tracker *t = camtrak_create(NULL);
    double sum_before = 0.0;
    int n_before = 0, false_fix = 0, first_lost = -1, reacquired = -1;
    int n_after = 0, good_after = 0, i;
    double worst_ms = 0.0;
    camtrak_frame f;

    CHECK(t != NULL);
    CHECK(tt_synth_init(&s, 320, 240, 24, 12, 5u) == 0);
    if (!t) {
        tt_synth_free(&s);
        return;
    }
    tt_synth_set_occlusion(&s, OCC_START, OCC_END);
    tt_synth_render(&s, 0);
    f = tt_synth_frame(&s);
    CHECK(camtrak_lock(t, &f, s.truth) == CAMTRAK_OK);

    for (i = 1; i < FRAMES; i++) {
        camtrak_result r;
        double t0, ms;
        float e;
        tt_synth_render(&s, i);
        f = tt_synth_frame(&s);
        t0 = tt_now_ms();
        CHECK(camtrak_update(t, &f, &r) == CAMTRAK_OK);
        ms = tt_now_ms() - t0;
        if (ms > worst_ms) worst_ms = ms;
        e = center_error(r.box, s.truth);

        if (i < OCC_START) {
            sum_before += e;
            n_before++;
        } else if (s.occluded) {
            if (r.status == CAMTRAK_TRACKING && e > 10.0f) false_fix++;
            if (r.status == CAMTRAK_LOST && first_lost < 0) first_lost = i;
        } else {
            if (reacquired < 0 && r.status == CAMTRAK_TRACKING && e <= 5.0f) reacquired = i;
            if (reacquired >= 0) {
                n_after++;
                if (e <= 5.0f) good_after++;
            }
        }
    }

    printf("occlusion  before err %.2f px | false TRACKING while hidden %d | LOST at frame %d"
           " (hidden %d..%d) | re-acquired %d frames after reappearing | after precision@5px "
           "%.1f%% | worst update %.2f ms\n",
           sum_before / n_before, false_fix, first_lost, OCC_START, OCC_END,
           reacquired < 0 ? -1 : reacquired - OCC_END - 1,
           n_after ? 100.0 * good_after / n_after : 0.0, worst_ms);

    CHECK(sum_before / n_before < 1.5);
    CHECK(false_fix == 0);
    CHECK(first_lost >= OCC_START && first_lost <= OCC_END);
    CHECK(reacquired >= 0 && reacquired - OCC_END - 1 <= REACQ_MAX);
    CHECK(n_after > 0 && (double)good_after / n_after >= 0.95);

    camtrak_destroy(t);
    tt_synth_free(&s);
}

/*
 * Scale: the target grows from 24 px to ~50 px over the clip (like a skier
 * coming towards the drone). Gates: the box keeps up with the size, and
 * center accuracy holds.
 */
static void scale(void)
{
    tt_synth s;
    camtrak_tracker *t = camtrak_create(NULL);
    double size_err_sum = 0.0;
    int n_late = 0, within5 = 0, i;
    float last_w = 0.0f, last_truth = 0.0f;
    camtrak_frame f;

    CHECK(t != NULL);
    CHECK(tt_synth_init(&s, 320, 240, 24, 12, 9u) == 0);
    if (!t) {
        tt_synth_free(&s);
        return;
    }
    s.growth = 1.0025;   /* x2.1 over 300 frames */
    tt_synth_render(&s, 0);
    f = tt_synth_frame(&s);
    CHECK(camtrak_lock(t, &f, s.truth) == CAMTRAK_OK);

    for (i = 1; i < FRAMES; i++) {
        camtrak_result r;
        tt_synth_render(&s, i);
        f = tt_synth_frame(&s);
        CHECK(camtrak_update(t, &f, &r) == CAMTRAK_OK);
        if (center_error(r.box, s.truth) <= 5.0f) within5++;
        if (i >= FRAMES - 50) {
            size_err_sum += fabs(r.box.w / s.truth.w - 1.0);
            n_late++;
        }
        last_w = r.box.w;
        last_truth = s.truth.w;
        if (i % 50 == 0) {
            printf("  scale frame %3d: truth %.0f px, box %.1f px\n", i, s.truth.w, r.box.w);
        }
    }
    printf("scale      target 24 -> %.0f px, box ends at %.1f px | size error last 50 frames "
           "%.1f%% | precision@5px %.1f%%\n", last_truth, last_w, 100.0 * size_err_sum / n_late,
           100.0 * within5 / (FRAMES - 1));
    CHECK(size_err_sum / n_late < 0.05);
    CHECK((double)within5 / (FRAMES - 1) >= 0.95);

    camtrak_destroy(t);
    tt_synth_free(&s);
}

static void speed(void)
{
    tt_synth s;
    camtrak_tracker *t = camtrak_create(NULL);
    double total = 0.0, worst = 0.0;
    int i;
    camtrak_frame f;

    CHECK(t != NULL);
    CHECK(tt_synth_init(&s, 1280, 720, 32, 12, 7u) == 0);
    if (!t) {
        tt_synth_free(&s);
        return;
    }
    tt_synth_render(&s, 0);
    f = tt_synth_frame(&s);
    CHECK(camtrak_lock(t, &f, s.truth) == CAMTRAK_OK);

    for (i = 1; i < FRAMES; i++) {
        camtrak_result r;
        double t0, dt;
        tt_synth_render(&s, i);          /* not timed */
        f = tt_synth_frame(&s);
        t0 = tt_now_ms();
        camtrak_update(t, &f, &r);
        dt = tt_now_ms() - t0;
        total += dt;
        if (dt > worst) worst = dt;
    }
    printf("speed 1280x720  %.3f ms/frame avg  %.3f ms worst  (~%.0f fps)\n",
           total / (FRAMES - 1), worst, 1000.0 / (total / (FRAMES - 1)));
#ifdef NDEBUG
    CHECK(total / (FRAMES - 1) < 0.9);
#endif
    camtrak_destroy(t);
    tt_synth_free(&s);
}

int main(void)
{
    accuracy(0);
    accuracy(1);
    occlusion();
    scale();
    speed();
    if (tt_failures) {
        fprintf(stderr, "%d check(s) failed\n", tt_failures);
        return 1;
    }
    printf("test_mosse: OK\n");
    return 0;
}
