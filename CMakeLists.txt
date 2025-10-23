cmake_minimum_required(VERSION 3.10)
project(spec_hud C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/deps)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/glfw)

# GLAD
add_library(glad STATIC
    deps/glad/src/glad.c
)
target_include_directories(glad PUBLIC deps/glad/include)

# Find required libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(SPA REQUIRED libspa-0.2)

# Create shared library for mel spectrogram generation
add_library(melspec SHARED
    src/melspec.c
)
target_include_directories(melspec PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${FFTW3F_INCLUDE_DIRS}
)
target_link_libraries(melspec
    ${FFTW3F_LIBRARIES}
    m
)

# Create shared library for audio streaming
add_library(audio_stream SHARED
    src/audio_stream.c
)
target_include_directories(audio_stream PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${PIPEWIRE_INCLUDE_DIRS}
    ${SPA_INCLUDE_DIRS}
)
target_link_libraries(audio_stream
    melspec
    ${PIPEWIRE_LIBRARIES}
    ${SPA_LIBRARIES}
    m
    pthread
)

# Create executable
add_executable(spec_hud
    src/spec_hud.c
)

# Link libraries
target_include_directories(spec_hud PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(spec_hud
    audio_stream
    glfw
    glad
    pthread
)
