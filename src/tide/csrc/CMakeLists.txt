# TIDE Backend CMakelists.txt

cmake_minimum_required(VERSION 3.18)
project(tide_backend LANGUAGES C CXX)

option(TIDE_ENABLE_CUDA "Enable CUDA backend" ON)

# Detect and enable CUDA if present and enabled
if(TIDE_ENABLE_CUDA)
    # Prefer a modern CUDA toolkit if present. This matters for newer ISA
    # features (e.g. FP8 convert instructions on SM89+).
    if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda")
        set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root" FORCE)
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        # On Windows, CUDA needs MSVC cl.exe as host compiler, even if C/C++ uses Clang
        if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
            # Find MSVC cl.exe for CUDA host compiler
            find_program(MSVC_CL_EXECUTABLE cl.exe
                PATHS "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
                      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC/*/bin/Hostx64/x64"
                      "C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/*/bin/Hostx64/x64"
                DOC "MSVC cl.exe for CUDA host compiler"
            )
            if(MSVC_CL_EXECUTABLE)
                set(CMAKE_CUDA_HOST_COMPILER "${MSVC_CL_EXECUTABLE}")
                message(STATUS "Using MSVC cl.exe for CUDA: ${CMAKE_CUDA_HOST_COMPILER}")
            else()
                message(WARNING "Could not find MSVC cl.exe for CUDA host compiler")
            endif()
        endif()
        enable_language(CUDA)
    else()
        message(WARNING "CUDA not found. Building without CUDA support.")
    endif()
else()
    message(STATUS "CUDA disabled (TIDE_ENABLE_CUDA=OFF).")
    set(CUDAToolkit_FOUND FALSE)
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# --- OpenMP Configuration (aligned with deepwave) ---
add_library(Tide_OpenMP_Interface INTERFACE)
set(OPENMP_CONFIGURED FALSE)

# On Windows with Clang, we need to manually configure OpenMP
if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    # Try static linking first to avoid runtime DLL conflicts
    find_library(LIBOMP_STATIC_LIB NAMES libomp_static libomp.a PATHS "C:/Program Files/LLVM/lib" NO_DEFAULT_PATH)
    if(LIBOMP_STATIC_LIB)
        target_link_libraries(Tide_OpenMP_Interface INTERFACE "${LIBOMP_STATIC_LIB}")
        # clang-cl uses MSVC-style flags: -openmp instead of -fopenmp
        target_compile_options(Tide_OpenMP_Interface INTERFACE -openmp)
        set(OPENMP_CONFIGURED TRUE)
        message(STATUS "OpenMP enabled (static linking, Clang/LLVM on Windows).")
    else()
        # Fall back to dynamic linking
        find_library(LIBOMP_LIB NAMES libomp omp PATHS "C:/Program Files/LLVM/lib" NO_DEFAULT_PATH)
        if(LIBOMP_LIB)
            target_link_libraries(Tide_OpenMP_Interface INTERFACE "${LIBOMP_LIB}")
            target_compile_options(Tide_OpenMP_Interface INTERFACE -openmp)
            set(OPENMP_CONFIGURED TRUE)
            message(STATUS "OpenMP enabled (dynamic linking, Clang/LLVM on Windows).")
        else()
            message(STATUS "OpenMP not found (libomp.lib not in LLVM/lib).")
        endif()
    endif()
else()
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(Tide_OpenMP_Interface INTERFACE OpenMP::OpenMP_C)
        set(OPENMP_CONFIGURED TRUE)
        message(STATUS "OpenMP enabled.")
    else()
        message(STATUS "OpenMP not found.")
    endif()
endif()

# --- Compiler Feature Detection and Flags ---
include(CheckCSourceCompiles)

# AVX2 detection
set(AVX2_TEST_CODE "
    #include <immintrin.h>
    int main() {
        __m256 vec = _mm256_set1_ps(42.0f);
        return 0;
    }")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    if(WIN32)
        set(C_AVX2_FLAG "/arch:AVX2")  # Clang-cl uses MSVC-style flags
    else()
        set(C_AVX2_FLAG "-mavx2")
    endif()
endif()

if(C_AVX2_FLAG)
    set(CMAKE_REQUIRED_FLAGS "${C_AVX2_FLAG}")
    check_c_source_compiles("${AVX2_TEST_CODE}" HAVE_AVX2)
    unset(CMAKE_REQUIRED_FLAGS)
else()
    set(HAVE_AVX2 FALSE)
endif()

if(HAVE_AVX2)
    message(STATUS "AVX2 is supported.")
else()
    message(STATUS "AVX2 is not supported.")
endif()

# Release flags (aligned with deepwave)
if(CMAKE_BUILD_TYPE MATCHES Release)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
        if(WIN32)
            # Clang-cl on Windows uses MSVC-style flags
            set(C_RELEASE_FLAGS "/O2" "/fp:fast")
        else()
            set(C_RELEASE_FLAGS "-Ofast")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast")
        endif()
    endif()
endif()

# --- Helper Macros for Object Libraries ---
macro(add_tide_cpu_object_library BASENAME ACCURACY DTYPE)
    set(TARGET_NAME "${BASENAME}_${ACCURACY}_${DTYPE}_cpu_obj")
    add_library(${TARGET_NAME} OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.c)
    target_compile_definitions(${TARGET_NAME} PRIVATE 
        TIDE_STENCIL=${ACCURACY} 
        TIDE_DTYPE=${DTYPE}
        TIDE_DEVICE=cpu
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    list(APPEND TIDE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
    list(APPEND CPU_TARGETS ${TARGET_NAME})

    # Set PIC for shared library objects
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if(C_RELEASE_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_RELEASE_FLAGS})
    endif()

    if(HAVE_AVX2 AND C_AVX2_FLAG)
        target_compile_options(${TARGET_NAME} PRIVATE ${C_AVX2_FLAG})
    endif()
endmacro()

if(CUDAToolkit_FOUND)
    # Use explicit architectures. Some CMake defaults resolve to 75 even on
    # Ada GPUs. Prefer "all but sm_4x" (and only those supported by nvcc)
    # by default unless the user explicitly sets it.
    if(NOT CMAKE_CUDA_ARCHITECTURES
       OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
       OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52"
       OR CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
        set(_TIDE_ARCH_CANDIDATES "50;52;53;60;61;62;70;72;75;80;86;87;89;90")
        set(_TIDE_SUPPORTED_ARCHES "")
        set(_TIDE_ARCH_PROBE "${CMAKE_BINARY_DIR}/tide_cuda_arch_probe.cu")
        file(WRITE "${_TIDE_ARCH_PROBE}" "__global__ void tide_arch_probe() {}\n")
        foreach(_arch IN LISTS _TIDE_ARCH_CANDIDATES)
            execute_process(
                COMMAND "${CMAKE_CUDA_COMPILER}" -arch=sm_${_arch} -x cu -c
                        "${_TIDE_ARCH_PROBE}"
                        -o "${CMAKE_BINARY_DIR}/tide_cuda_arch_probe_${_arch}.o"
                RESULT_VARIABLE _TIDE_ARCH_OK
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_TIDE_ARCH_OK EQUAL 0)
                list(APPEND _TIDE_SUPPORTED_ARCHES "${_arch}")
            endif()
        endforeach()
        if(_TIDE_SUPPORTED_ARCHES STREQUAL "")
            message(WARNING "Could not detect supported CUDA architectures; defaulting to 89.")
            set(_TIDE_SUPPORTED_ARCHES "89")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${_TIDE_SUPPORTED_ARCHES}"
            CACHE STRING "CUDA architectures" FORCE)
        unset(_TIDE_ARCH_CANDIDATES)
        unset(_TIDE_SUPPORTED_ARCHES)
        unset(_TIDE_ARCH_PROBE)
        unset(_TIDE_ARCH_OK)
    endif()
    set(TIDE_CUDA_ARCHES "${CMAKE_CUDA_ARCHITECTURES}")
    string(REPLACE ";" "," TIDE_CUDA_ARCHES_STR "${TIDE_CUDA_ARCHES}")

    # Detect whether FP8 instructions are available for the compiled arch set.
    set(TIDE_CUDA_FP8_ENABLED 0)
    if(TIDE_CUDA_ARCHES STREQUAL "native" OR TIDE_CUDA_ARCHES STREQUAL "all")
        set(TIDE_CUDA_FP8_ENABLED 1)
    else()
        foreach(_arch IN LISTS TIDE_CUDA_ARCHES)
            string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
            if(_arch_num AND _arch_num GREATER_EQUAL 89)
                set(TIDE_CUDA_FP8_ENABLED 1)
            endif()
        endforeach()
    endif()

    macro(add_tide_cuda_object_library BASENAME ACCURACY DTYPE)
        set(TARGET_NAME "${BASENAME}_${ACCURACY}_${DTYPE}_cuda_obj")
        add_library(${TARGET_NAME} OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.cu)
        # Set TIDE_DTYPE_FLOAT based on DTYPE (optimization 2.1)
        if(${DTYPE} STREQUAL "float")
            set(IS_FLOAT 1)
        else()
            set(IS_FLOAT 0)
        endif()
        target_compile_definitions(${TARGET_NAME} PRIVATE
            TIDE_STENCIL=${ACCURACY}
            TIDE_DTYPE=${DTYPE}
            TIDE_DEVICE=cuda
            TIDE_DTYPE_FLOAT=${IS_FLOAT}
            TIDE_CUDA_FP8_ENABLED=${TIDE_CUDA_FP8_ENABLED}
        )
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        list(APPEND TIDE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${TIDE_CUDA_ARCHES}")

        if(CMAKE_BUILD_TYPE MATCHES Release)
            target_compile_options(${TARGET_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:
                    --use_fast_math
                    -O3
                    --restrict
                    --maxrregcount=64
                    -Xptxas=-dlcm=ca
                >)
        endif()
    endmacro()
endif()

# Prepare lists that will collect the object files
set(TIDE_OBJECTS)
set(CPU_TARGETS)
set(ACCURACIES 2 4 6 8)
set(DTYPES float double)

# --- Storage utilities ---
set(STORAGE_UTILS_CPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/storage_utils.c)
set(STORAGE_UTILS_CUDA_SRC "")

# --- CPU object libraries ---
foreach(ACCURACY ${ACCURACIES})
    foreach(DTYPE ${DTYPES})
        add_tide_cpu_object_library(maxwell ${ACCURACY} ${DTYPE})
    endforeach()
endforeach()

if(OPENMP_CONFIGURED)
    foreach(CPU_TARGET ${CPU_TARGETS})
        target_link_libraries(${CPU_TARGET} PRIVATE Tide_OpenMP_Interface)
    endforeach()
endif()

# --- CUDA object libraries ---
if(CUDAToolkit_FOUND)
    if(CMAKE_BUILD_TYPE MATCHES Release)
        set(CUDA_RELEASE_OPTIONS --use_fast_math -O3 --restrict --maxrregcount=64 -Xptxas=-dlcm=ca)
    endif()
    if(NOT WIN32)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler=-fPIC)
    endif()

    foreach(ACCURACY ${ACCURACIES})
        foreach(DTYPE ${DTYPES})
            add_tide_cuda_object_library(maxwell ${ACCURACY} ${DTYPE})
        endforeach()
    endforeach()

    # --- Storage utilities ---
    set(STORAGE_UTILS_CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/storage_utils.cu)
    set_source_files_properties(${STORAGE_UTILS_CUDA_SRC} PROPERTIES CUDA_ARCHITECTURES "${TIDE_CUDA_ARCHES}")
endif()

# --- Final Library Build ---
# Combine all objects into a single shared library
add_library(tide_C SHARED ${TIDE_OBJECTS} ${STORAGE_UTILS_CUDA_SRC} ${STORAGE_UTILS_CPU_SRC})

if(WIN32)
    set_target_properties(tide_C PROPERTIES OUTPUT_NAME "libtide_C")
endif()

set_target_properties(tide_C PROPERTIES
    C_VISIBILITY_PRESET default
    CUDA_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(OPENMP_CONFIGURED)
    target_link_libraries(tide_C PRIVATE Tide_OpenMP_Interface)
endif()

if(HAVE_AVX2)
    target_compile_definitions(tide_C PRIVATE HAVE_AVX2)
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(tide_C PRIVATE
        TIDE_CUDA_FP8_ENABLED=${TIDE_CUDA_FP8_ENABLED}
        TIDE_CUDA_ARCHES_STR="${TIDE_CUDA_ARCHES_STR}"
    )
endif()

# Set output directory to the tide package directory
set_target_properties(tide_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# --- Print Configuration Summary ---
message(STATUS "")
message(STATUS "TIDE Backend Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP: ${OPENMP_CONFIGURED}")
message(STATUS "  AVX2: ${HAVE_AVX2}")
if(CUDAToolkit_FOUND)
    message(STATUS "  CUDA: ON")
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "  CUDA: OFF")
endif()
message(STATUS "  Output Directory: ${CMAKE_CURRENT_SOURCE_DIR}/..")
message(STATUS "")

# Install target
install(TARGETS tide_C
    LIBRARY DESTINATION tide
    ARCHIVE DESTINATION tide
    RUNTIME DESTINATION tide
)

# Bundle OpenMP runtime on Windows (Clang's libomp.dll)
if(WIN32)
    set(TIDE_LIBOMP_DLL "${CMAKE_CURRENT_SOURCE_DIR}/../libomp.dll")
    if(EXISTS "${TIDE_LIBOMP_DLL}")
        install(FILES "${TIDE_LIBOMP_DLL}" DESTINATION tide)
    endif()
endif()
