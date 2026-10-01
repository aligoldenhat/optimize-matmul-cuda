cmake_minimum_required(VERSION 3.24)

# RTX 3050 = sm_86. Must be set BEFORE project(): project() fills in nvcc's
# default (sm_75) if it's unset, and that default would win.
# Override with -DCMAKE_CUDA_ARCHITECTURES=native if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(optimize_matmul_cuda LANGUAGES CXX CUDA)

# ---- Options (toggled by presets or -D on the command line) --------------
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(ENABLE_SANITIZERS "Build host code with ASan + UBSan" OFF)

# ---- Global settings -----------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++20, not -std=gnu++20 (no GNU-only extensions)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile_commands.json for clangd / clang-tidy

find_package(CUDAToolkit REQUIRED) # provides CUDA::cudart, CUDA::cublas

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

# ---- Targets -------------------------------------------------------------
add_executable(sgemm src/main.cu)
target_include_directories(sgemm PRIVATE src)
target_link_libraries(sgemm PRIVATE CUDA::cudart CUDA::cublas)

# Non-Debug builds: -lineinfo maps SASS back to source lines in ncu /
# compute-sanitizer. Costs nothing at runtime, so it's always on there.
# (Debug builds use -G below, which already includes line info and conflicts
# with -lineinfo.)
target_compile_options(sgemm PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

# Debug builds: -G keeps device debug info and turns off device optimizations,
# so cuda-gdb can set breakpoints in kernels and show local variables.
# Kernels get MUCH slower with -G: never benchmark a Debug build.
target_compile_options(sgemm PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)

set_project_warnings(sgemm ${WARNINGS_AS_ERRORS})
enable_sanitizers(sgemm ${ENABLE_SANITIZERS})
