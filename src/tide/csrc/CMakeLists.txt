# TIDE Backend CMakelists.txt

cmake_minimum_required(VERSION 3.18)
project(tide_backend LANGUAGES C CXX)

option(TIDE_ENABLE_CUDA "Enable CUDA backend" ON)

# Detect and enable CUDA if present and enabled
if(TIDE_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
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

find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    target_link_libraries(Tide_OpenMP_Interface INTERFACE OpenMP::OpenMP_C)
    set(OPENMP_CONFIGURED TRUE)
    message(STATUS "OpenMP enabled.")
else()
    message(STATUS "OpenMP not found.")
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
    set(C_AVX2_FLAG "-mavx2")
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(C_AVX2_FLAG "/arch:AVX2")
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
        set(C_RELEASE_FLAGS "-Ofast")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast")
    elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
        set(C_RELEASE_FLAGS "/O2" "/fp:fast")
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
        )
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        list(APPEND TIDE_OBJECTS $<TARGET_OBJECTS:${TARGET_NAME}>)
        
        # Use "all" or specific architectures
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "89")
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        endif()

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
        add_tide_cpu_object_library(maxwell3d ${ACCURACY} ${DTYPE})
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
            add_tide_cuda_object_library(maxwell3d ${ACCURACY} ${DTYPE})
        endforeach()
    endforeach()

    # --- Storage utilities ---
    set(STORAGE_UTILS_CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/storage_utils.cu)
    set_source_files_properties(${STORAGE_UTILS_CUDA_SRC} PROPERTIES CUDA_ARCHITECTURES "89")
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

if(WIN32)
    set(TIDE_IOMP5_DLL "${CMAKE_CURRENT_SOURCE_DIR}/../libiomp5md.dll")
    if(EXISTS "${TIDE_IOMP5_DLL}")
        install(FILES "${TIDE_IOMP5_DLL}" DESTINATION tide)
    endif()
endif()
