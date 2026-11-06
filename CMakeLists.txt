cmake_minimum_required(VERSION 3.16)
project(camtrak VERSION 0.4.0 LANGUAGES C)

# ---------------------------------------------------------------------------
# Camtrak: CPU-only visual object tracker. Core is plain C11 + libm, nothing else.
# ---------------------------------------------------------------------------

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Default to an optimized build; the speed test is meaningless in Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CAMTRAK_BUILD_TESTS  "Build unit tests and the synthetic benchmark" ON)
option(CAMTRAK_BUILD_SHARED "Also build a shared library (used by the Python/Node bindings)" ON)

set(CAMTRAK_SOURCES
  src/color.c
  src/fft.c
  src/kalman.c
  src/patch.c
  src/mosse.c
  src/scale.c
  src/camtrak.c
)

function(camtrak_configure target)
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
add_library(camtrak STATIC ${CAMTRAK_SOURCES})
camtrak_configure(camtrak)

# Shared library: camtrak_c.dll / libcamtrak_c.so / libcamtrak_c.dylib.
# Only the CAMTRAK_API functions are exported.
if(CAMTRAK_BUILD_SHARED)
  add_library(camtrak_shared SHARED ${CAMTRAK_SOURCES})
  camtrak_configure(camtrak_shared)
  set_target_properties(camtrak_shared PROPERTIES
    OUTPUT_NAME camtrak_c
    C_VISIBILITY_PRESET hidden)
  target_compile_definitions(camtrak_shared PRIVATE CAMTRAK_SHARED CAMTRAK_BUILDING)
endif()

if(CAMTRAK_BUILD_TESTS)
  enable_testing()

  # Tests may poke at internal headers (fft.h etc.), hence the src include dir.
  function(camtrak_test name)
    add_executable(${name} tests/${name}.c ${ARGN})
    target_link_libraries(${name} PRIVATE camtrak)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                               ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  camtrak_test(test_fft)
  camtrak_test(test_api)
  camtrak_test(test_mosse tests/synth.c)
endif()
