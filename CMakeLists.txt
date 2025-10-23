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

# Add libsoundio as subdirectory
add_subdirectory(deps/libsoundio)
include_directories(${CMAKE_SOURCE_DIR}/deps/libsoundio)

# Find required libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3F REQUIRED fftw3f)

# Create executable (compile .c as C++ for fpng compatibility)
add_executable(spec_hud
    src/spec_hud.c
    deps/fpng/src/fpng.cpp
)
set_source_files_properties(src/spec_hud.c PROPERTIES LANGUAGE CXX)

# Link libraries
target_include_directories(spec_hud PRIVATE ${SNDFILE_INCLUDE_DIRS} ${FFTW3F_INCLUDE_DIRS})
target_link_libraries(spec_hud
    ${SNDFILE_LIBRARIES}
    ${FFTW3F_LIBRARIES}
    libsoundio_shared
    glfw
    glad
    m
    pthread
)
