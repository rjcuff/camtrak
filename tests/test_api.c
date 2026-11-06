/* test_api: argument validation and lifecycle of the public C API. */
#include "camtrak/camtrak.h"
#include "test_util.h"

#include <string.h>

int main(void)
{
    static uint8_t pixels[64 * 48];
    camtrak_frame f = { pixels, 64, 48, 64, CAMTRAK_FORMAT_GRAY8 };
    camtrak_box good = { 20.0f, 14.0f, 16.0f, 16.0f };
    camtrak_result r;
    camtrak_params p;
    camtrak_tracker *t;

    memset(pixels, 128, sizeof(pixels));

    /* invalid params are refused */
    p = camtrak_default_params();
    p.template_size = 48;           /* not a power of two */
    CHECK(camtrak_create(&p) == NULL);
    p = camtrak_default_params();
    p.learning_rate = 0.0f;
    CHECK(camtrak_create(&p) == NULL);

    t = camtrak_create(NULL);
    CHECK(t != NULL);
    if (!t) {
        return 1;
    }

    /* update before lock */
    CHECK(camtrak_update(t, &f, &r) == CAMTRAK_ERR_NOT_LOCKED);

    /* bad frames and boxes */
    {
        camtrak_frame bad = f;
        camtrak_box tiny = { 10.0f, 10.0f, 2.0f, 2.0f };
        camtrak_box outside = { 500.0f, 10.0f, 16.0f, 16.0f };
        bad.stride = 10;
        CHECK(camtrak_lock(t, &bad, good) == CAMTRAK_ERR_INVALID_ARG);
        bad = f;
        bad.data = NULL;
        CHECK(camtrak_lock(t, &bad, good) == CAMTRAK_ERR_INVALID_ARG);
        CHECK(camtrak_lock(t, &f, tiny) == CAMTRAK_ERR_INVALID_ARG);
        CHECK(camtrak_lock(t, &f, outside) == CAMTRAK_ERR_INVALID_ARG);
        CHECK(camtrak_lock(NULL, &f, good) == CAMTRAK_ERR_INVALID_ARG);
    }

    /* a flat frame must not crash or produce NaNs */
    CHECK(camtrak_lock(t, &f, good) == CAMTRAK_OK);
    CHECK(camtrak_update(t, &f, &r) == CAMTRAK_OK);
    CHECK(r.box.x == r.box.x && r.box.y == r.box.y);   /* not NaN */
    CHECK(camtrak_update(t, &f, NULL) == CAMTRAK_ERR_INVALID_ARG);

    /* re-locking reuses the handle */
    CHECK(camtrak_lock(t, &f, good) == CAMTRAK_OK);

    camtrak_destroy(t);
    camtrak_destroy(NULL);   /* allowed, like free(NULL) */

    if (tt_failures) {
        fprintf(stderr, "%d check(s) failed\n", tt_failures);
        return 1;
    }
    printf("test_api: OK (camtrak %s)\n", camtrak_version());
    return 0;
}
