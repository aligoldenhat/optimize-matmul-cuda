# Compiler warnings: the C++ equivalent of a strict type checker.
#
# Important: nvcc splits a .cu file in two.
#   - host code   -> compiled by g++  (gets the flags below via -Xcompiler)
#   - device code -> compiled by NVIDIA's own compiler (cicc), which only
#                    understands nvcc's few warning flags
# So g++ warnings like -Wconversion never see your kernels. clangd /
# clang-tidy do parse device code, so they fill part of that gap.

function(set_project_warnings target warnings_as_errors)
  set(host_warnings
      -Wall
      -Wextra                # extra checks: unused parameters, etc.
      -Wshadow               # a variable hides one from an outer scope
      -Wconversion           # implicit narrowing: size_t -> int, double -> float
      -Wsign-conversion      # implicit signed <-> unsigned (common in index math)
      -Wdouble-promotion     # float silently promoted to double (slow on GPUs!)
      -Wnull-dereference
      -Wformat=2             # printf format string checks
      -Wimplicit-fallthrough # switch case without break
      -Wcast-align
      -Wunused)

  # Only for pure .cpp files: nvcc's generated host stubs trigger these.
  set(cxx_only_warnings -Wpedantic -Wold-style-cast -Wnon-virtual-dtor)

  # nvcc's own warnings (these do apply to device code).
  set(cuda_warnings -Wreorder --Wext-lambda-captures-this)

  if(warnings_as_errors)
    list(APPEND host_warnings -Werror)
    list(APPEND cuda_warnings --Werror=all-warnings)
  endif()

  # -Xcompiler wants one comma-separated list.
  list(JOIN host_warnings "," host_warnings_csv)

  target_compile_options(
    ${target}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${host_warnings} ${cxx_only_warnings}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${host_warnings_csv} ${cuda_warnings}>)
endfunction()
