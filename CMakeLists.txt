cmake_minimum_required(VERSION 3.16)
project(tether VERSION 0.4.0 LANGUAGES C)

# ---------------------------------------------------------------------------
# Tether: CPU-only visual object tracker. Core is plain C11 + libm, nothing else.
# ---------------------------------------------------------------------------

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Default to an optimised build; the speed test is meaningless in Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TETHER_BUILD_TESTS  "Build unit tests and the synthetic benchmark" ON)
option(TETHER_BUILD_SHARED "Also build a shared library (used by the Python/Node bindings)" ON)

set(TETHER_SOURCES
  src/color.c
  src/fft.c
  src/kalman.c
  src/patch.c
  src/mosse.c
  src/scale.c
  src/tether.c
)

function(tether_configure target)
  target_include_directories(${target}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    # libm (sin/cos/exp/log). On macOS this is part of libSystem but linking is harmless.
    target_link_libraries(${target} PUBLIC m)
  endif()
endfunction()

# Static library: what C programs and the tests link against.
add_library(tether STATIC ${TETHER_SOURCES})
tether_configure(tether)

# Shared library: tether_c.dll / libtether_c.so / libtether_c.dylib.
# Only the TETHER_API functions are exported.
if(TETHER_BUILD_SHARED)
  add_library(tether_shared SHARED ${TETHER_SOURCES})
  tether_configure(tether_shared)
  set_target_properties(tether_shared PROPERTIES
    OUTPUT_NAME tether_c
    C_VISIBILITY_PRESET hidden)
  target_compile_definitions(tether_shared PRIVATE TETHER_SHARED TETHER_BUILDING)
endif()

if(TETHER_BUILD_TESTS)
  enable_testing()

  # Tests may poke at internal headers (fft.h etc.), hence the src include dir.
  function(tether_test name)
    add_executable(${name} tests/${name}.c ${ARGN})
    target_link_libraries(${name} PRIVATE tether)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                               ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  tether_test(test_fft)
  tether_test(test_api)
  tether_test(test_mosse tests/synth.c)
endif()
