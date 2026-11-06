#!/bin/sh
# Build without CMake (macOS / Linux): shared library + C tests into build/.
# With CMake available, prefer:  cmake -S . -B build && cmake --build build && ctest --test-dir build
set -e
cd "$(dirname "$0")"
mkdir -p build
CC=${CC:-cc}
CFLAGS="-O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude -Isrc"
case "$(uname)" in Darwin) LIB=libtether_c.dylib ;; *) LIB=libtether_c.so ;; esac
$CC $CFLAGS -shared -fPIC -fvisibility=hidden -DTETHER_SHARED -DTETHER_BUILDING src/*.c -o build/$LIB -lm
$CC $CFLAGS -Itests tests/test_fft.c src/*.c -o build/test_fft -lm
$CC $CFLAGS -Itests tests/test_api.c src/*.c -o build/test_api -lm
$CC $CFLAGS -Itests tests/test_mosse.c tests/synth.c src/*.c -o build/test_mosse -lm
if [ "$1" = "test" ]; then
  ./build/test_fft | tail -1 && ./build/test_api | tail -1 && ./build/test_mosse
fi
