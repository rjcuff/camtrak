/* test_api: argument validation and lifecycle of the public C API. */
#include "tether/tether.h"
#include "test_util.h"

#include <string.h>

int main(void)
{
    static uint8_t pixels[64 * 48];
    tether_frame f = { pixels, 64, 48, 64, TETHER_FORMAT_GRAY8 };
    tether_box good = { 20.0f, 14.0f, 16.0f, 16.0f };
    tether_result r;
    tether_params p;
    tether_tracker *t;

    memset(pixels, 128, sizeof(pixels));

    /* invalid params are refused */
    p = tether_default_params();
    p.template_size = 48;           /* not a power of two */
    CHECK(tether_create(&p) == NULL);
    p = tether_default_params();
    p.learning_rate = 0.0f;
    CHECK(tether_create(&p) == NULL);

    t = tether_create(NULL);
    CHECK(t != NULL);
    if (!t) {
        return 1;
    }

    /* update before lock */
    CHECK(tether_update(t, &f, &r) == TETHER_ERR_NOT_LOCKED);

    /* bad frames and boxes */
    {
        tether_frame bad = f;
        tether_box tiny = { 10.0f, 10.0f, 2.0f, 2.0f };
        tether_box outside = { 500.0f, 10.0f, 16.0f, 16.0f };
        bad.stride = 10;
        CHECK(tether_lock(t, &bad, good) == TETHER_ERR_INVALID_ARG);
        bad = f;
        bad.data = NULL;
        CHECK(tether_lock(t, &bad, good) == TETHER_ERR_INVALID_ARG);
        CHECK(tether_lock(t, &f, tiny) == TETHER_ERR_INVALID_ARG);
        CHECK(tether_lock(t, &f, outside) == TETHER_ERR_INVALID_ARG);
        CHECK(tether_lock(NULL, &f, good) == TETHER_ERR_INVALID_ARG);
    }

    /* a flat frame must not crash or produce NaNs */
    CHECK(tether_lock(t, &f, good) == TETHER_OK);
    CHECK(tether_update(t, &f, &r) == TETHER_OK);
    CHECK(r.box.x == r.box.x && r.box.y == r.box.y);   /* not NaN */
    CHECK(tether_update(t, &f, NULL) == TETHER_ERR_INVALID_ARG);

    /* re-locking reuses the handle */
    CHECK(tether_lock(t, &f, good) == TETHER_OK);

    tether_destroy(t);
    tether_destroy(NULL);   /* allowed, like free(NULL) */

    if (tt_failures) {
        fprintf(stderr, "%d check(s) failed\n", tt_failures);
        return 1;
    }
    printf("test_api: OK (tether %s)\n", tether_version());
    return 0;
}
