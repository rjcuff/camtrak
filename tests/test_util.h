/* Minimal test helpers: no framework, the core has no dependencies and neither do its tests. */
#ifndef TT_TEST_UTIL_H
#define TT_TEST_UTIL_H

#include <stdio.h>
#include <time.h>

static int tt_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            tt_failures++;                                                   \
        }                                                                    \
    } while (0)

static inline double tt_now_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

#endif
