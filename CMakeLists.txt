cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SpeakerEmbedding)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# LibTorch
# -----------------------------------------------------------------------------
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -----------------------------------------------------------------------------
# libsndfile et FFTW3
# -----------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3 REQUIRED fftw3)

# -----------------------------------------------------------------------------
# ONNX Runtime
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_DIR "/usr/local")

include_directories(
    ${SNDFILE_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${SNDFILE_LIBRARY_DIRS}
    ${FFTW3_LIBRARY_DIRS}
    ${ONNXRUNTIME_DIR}/lib
)

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------
add_executable(
    example-app
    main.cpp
)

# -----------------------------------------------------------------------------
# Liens des bibliothèques
# -----------------------------------------------------------------------------
target_link_libraries(
    example-app
    PRIVATE
        ${TORCH_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${FFTW3_LIBRARIES}
        onnxruntime
)