# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CMake functions for HIP backend configuration, kernel compilation, and embedding

#==============================================================================
# Configuration
#==============================================================================

# Store the project root directory for later use
set(HIP_CTS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "HIP CTS root directory")
set(HIP_CTS_EMBED_BINARY_SCRIPT "${HIP_CTS_ROOT_DIR}/tools/embed_binary.py" CACHE INTERNAL "embed_binary.py script path")

# Backend configuration file
set(HIP_CTS_BACKEND_CONFIG "${HIP_CTS_ROOT_DIR}/cmake/hip_backend_config_amd.json" 
    CACHE FILEPATH "Path to HIP backend configuration JSON file")

# Directory for generated files
set(KERNEL_GEN_DIR "${hip_cts_BINARY_DIR}/generated/kernels")
set(CONFIG_GEN_DIR "${hip_cts_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${KERNEL_GEN_DIR})
file(MAKE_DIRECTORY ${CONFIG_GEN_DIR})

#==============================================================================
# Backend Configuration Loading
#==============================================================================

# Read and parse the backend configuration
function(_load_backend_config)
    if(NOT EXISTS "${HIP_CTS_BACKEND_CONFIG}")
        message(FATAL_ERROR "Backend config not found: ${HIP_CTS_BACKEND_CONFIG}")
    endif()
    
    file(READ "${HIP_CTS_BACKEND_CONFIG}" CONFIG_JSON)
    
    # Parse config name
    string(JSON CONFIG_NAME GET ${CONFIG_JSON} "name")
    message(STATUS "HIP backend config: ${CONFIG_NAME}")
    set(HIP_BACKEND_NAME "${CONFIG_NAME}" CACHE INTERNAL "HIP backend name")
    
    # Parse runtime settings
    string(JSON RUNTIME_DEFAULT_LIBRARY GET ${CONFIG_JSON} "runtime" "default_library")
    set(HIP_DEFAULT_LIBRARY "${RUNTIME_DEFAULT_LIBRARY}" CACHE INTERNAL "Default HIP runtime library")
    
    # Parse library search paths array
    string(JSON SEARCH_PATHS_LENGTH LENGTH ${CONFIG_JSON} "runtime" "library_search_paths")
    set(SEARCH_PATHS_LIST "")
    if(SEARCH_PATHS_LENGTH GREATER 0)
        math(EXPR SEARCH_PATHS_LAST "${SEARCH_PATHS_LENGTH} - 1")
        foreach(IDX RANGE ${SEARCH_PATHS_LAST})
            string(JSON SEARCH_PATH GET ${CONFIG_JSON} "runtime" "library_search_paths" ${IDX})
            list(APPEND SEARCH_PATHS_LIST "${SEARCH_PATH}")
        endforeach()
    endif()
    set(HIP_LIBRARY_SEARCH_PATHS "${SEARCH_PATHS_LIST}" CACHE INTERNAL "HIP library search paths")
    
    # Parse env variable hints for runtime
    string(JSON RUNTIME_ENV_PATH ERROR_VARIABLE _err GET ${CONFIG_JSON} "runtime" "env_library_path")
    string(JSON RUNTIME_ENV_SUBDIR ERROR_VARIABLE _err GET ${CONFIG_JSON} "runtime" "env_library_subdir")
    set(HIP_RUNTIME_ENV_PATH "${RUNTIME_ENV_PATH}" CACHE INTERNAL "Env var for HIP library path")
    set(HIP_RUNTIME_ENV_SUBDIR "${RUNTIME_ENV_SUBDIR}" CACHE INTERNAL "Subdir under env var for HIP library")
    
    # Parse compiler settings
    string(JSON COMPILER_EXECUTABLE GET ${CONFIG_JSON} "compiler" "executable")
    string(JSON COMPILER_HINT_ENV ERROR_VARIABLE _err GET ${CONFIG_JSON} "compiler" "find_hint_env")
    string(JSON COMPILER_HINT_SUBDIR ERROR_VARIABLE _err GET ${CONFIG_JSON} "compiler" "find_hint_subdir")
    
    # Parse compilation settings
    string(JSON OUTPUT_EXTENSION GET ${CONFIG_JSON} "compilation" "output_extension")
    
    # Parse command template array into a list
    string(JSON TEMPLATE_LENGTH LENGTH ${CONFIG_JSON} "compilation" "command_template")
    set(COMMAND_TEMPLATE_LIST "")
    math(EXPR TEMPLATE_LAST "${TEMPLATE_LENGTH} - 1")
    foreach(IDX RANGE ${TEMPLATE_LAST})
        string(JSON TEMPLATE_ITEM GET ${CONFIG_JSON} "compilation" "command_template" ${IDX})
        list(APPEND COMMAND_TEMPLATE_LIST "${TEMPLATE_ITEM}")
    endforeach()
    
    # Parse target settings
    string(JSON TARGET_FLAG_PREFIX GET ${CONFIG_JSON} "targets" "flag_prefix")
    
    # Parse default targets array into a list
    string(JSON TARGETS_LENGTH LENGTH ${CONFIG_JSON} "targets" "defaults")
    set(DEFAULT_TARGETS_LIST "")
    math(EXPR TARGETS_LAST "${TARGETS_LENGTH} - 1")
    foreach(IDX RANGE ${TARGETS_LAST})
        string(JSON TARGET_ITEM GET ${CONFIG_JSON} "targets" "defaults" ${IDX})
        list(APPEND DEFAULT_TARGETS_LIST "${TARGET_ITEM}")
    endforeach()
    
    # Export compiler settings to cache
    set(KERNEL_COMPILER_EXECUTABLE "${COMPILER_EXECUTABLE}" CACHE INTERNAL "Kernel compiler executable name")
    set(KERNEL_COMPILER_HINT_ENV "${COMPILER_HINT_ENV}" CACHE INTERNAL "Env var hint for finding compiler")
    set(KERNEL_COMPILER_HINT_SUBDIR "${COMPILER_HINT_SUBDIR}" CACHE INTERNAL "Subdir hint for finding compiler")
    set(KERNEL_OUTPUT_EXTENSION "${OUTPUT_EXTENSION}" CACHE INTERNAL "Kernel output file extension")
    set(KERNEL_COMMAND_TEMPLATE "${COMMAND_TEMPLATE_LIST}" CACHE INTERNAL "Kernel compilation command template")
    set(KERNEL_TARGET_FLAG_PREFIX "${TARGET_FLAG_PREFIX}" CACHE INTERNAL "Prefix for GPU target flags")
    set(KERNEL_DEFAULT_TARGETS "${DEFAULT_TARGETS_LIST}" CACHE INTERNAL "Default GPU targets")
endfunction()

# Load the configuration
_load_backend_config()

#==============================================================================
# Generate Backend Config Header
#==============================================================================

# Generate C++ header with backend configuration
function(_generate_backend_config_header)
    # Build the search paths as a C++ initializer list
    set(SEARCH_PATHS_CPP "")
    foreach(PATH ${HIP_LIBRARY_SEARCH_PATHS})
        if(SEARCH_PATHS_CPP)
            set(SEARCH_PATHS_CPP "${SEARCH_PATHS_CPP},\n        \"${PATH}\"")
        else()
            set(SEARCH_PATHS_CPP "\"${PATH}\"")
        endif()
    endforeach()
    
    # Generate the header file
    set(CONFIG_HEADER_CONTENT 
"// Copyright (c) 2024 HIP CTS Authors
// SPDX-License-Identifier: MIT
// 
// AUTO-GENERATED FILE - DO NOT EDIT
// Generated from: ${HIP_CTS_BACKEND_CONFIG}

#pragma once

#include <string>
#include <vector>

namespace hip_cts {
namespace config {

// Backend name
constexpr const char* kBackendName = \"${HIP_BACKEND_NAME}\";

// Default HIP runtime library to load
constexpr const char* kDefaultHipLibrary = \"${HIP_DEFAULT_LIBRARY}\";

// Library search paths (in priority order)
inline std::vector<std::string> getLibrarySearchPaths() {
    return {
        ${SEARCH_PATHS_CPP}
    };
}

// Environment variable that may contain HIP library path
constexpr const char* kLibraryEnvVar = \"${HIP_RUNTIME_ENV_PATH}\";
constexpr const char* kLibraryEnvSubdir = \"${HIP_RUNTIME_ENV_SUBDIR}\";

} // namespace config
} // namespace hip_cts
")
    
    file(WRITE "${CONFIG_GEN_DIR}/hip_backend_config.hpp" "${CONFIG_HEADER_CONTENT}")
    message(STATUS "Generated: ${CONFIG_GEN_DIR}/hip_backend_config.hpp")
endfunction()

_generate_backend_config_header()

#==============================================================================
# Find Kernel Compiler
#==============================================================================

# Find the kernel compiler based on config
if(KERNEL_COMPILER_HINT_ENV AND DEFINED ENV{${KERNEL_COMPILER_HINT_ENV}})
    set(_COMPILER_HINT_PATH "$ENV{${KERNEL_COMPILER_HINT_ENV}}/${KERNEL_COMPILER_HINT_SUBDIR}")
else()
    set(_COMPILER_HINT_PATH "")
endif()

find_program(KERNEL_COMPILER_PATH 
    NAMES ${KERNEL_COMPILER_EXECUTABLE}
    HINTS ${_COMPILER_HINT_PATH}
    DOC "Path to kernel compiler executable"
)

if(KERNEL_COMPILER_PATH)
    message(STATUS "Found kernel compiler: ${KERNEL_COMPILER_PATH}")
    # Also set HIPCC_EXECUTABLE for backward compatibility
    set(HIPCC_EXECUTABLE "${KERNEL_COMPILER_PATH}" CACHE FILEPATH "Path to kernel compiler (legacy)")
else()
    message(STATUS "Kernel compiler not found: ${KERNEL_COMPILER_EXECUTABLE}")
endif()

#==============================================================================
# Low-Level Functions
#==============================================================================

# Function to compile a kernel using the configured compiler
# Usage: compile_kernel(<kernel_name> <source_file> [GPU_TARGETS target1 target2 ...])
function(compile_kernel KERNEL_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "" "" "GPU_TARGETS" ${ARGN})
    
    if(NOT KERNEL_COMPILER_PATH)
        message(FATAL_ERROR "Kernel compiler not found - cannot compile kernel ${KERNEL_NAME}")
    endif()
    
    # Use provided targets or defaults from config
    if(NOT ARG_GPU_TARGETS)
        set(ARG_GPU_TARGETS ${KERNEL_DEFAULT_TARGETS})
    endif()
    
    # Build the target flags
    set(TARGET_FLAGS "")
    foreach(TARGET ${ARG_GPU_TARGETS})
        list(APPEND TARGET_FLAGS "${KERNEL_TARGET_FLAG_PREFIX}${TARGET}")
    endforeach()
    
    get_filename_component(SOURCE_ABS "${SOURCE_FILE}" ABSOLUTE)
    set(KERNEL_OUTPUT "${KERNEL_GEN_DIR}/${KERNEL_NAME}${KERNEL_OUTPUT_EXTENSION}")
    
    # Build command from template
    set(COMPILE_COMMAND "")
    foreach(TEMPLATE_PART ${KERNEL_COMMAND_TEMPLATE})
        # Substitute variables in template
        string(REPLACE "\${COMPILER}" "${KERNEL_COMPILER_PATH}" PART "${TEMPLATE_PART}")
        string(REPLACE "\${OUTPUT}" "${KERNEL_OUTPUT}" PART "${PART}")
        string(REPLACE "\${SOURCE}" "${SOURCE_ABS}" PART "${PART}")
        
        # Handle TARGET_FLAGS specially - it expands to multiple arguments
        if("${PART}" STREQUAL "\${TARGET_FLAGS}")
            list(APPEND COMPILE_COMMAND ${TARGET_FLAGS})
        else()
            list(APPEND COMPILE_COMMAND "${PART}")
        endif()
    endforeach()
    
    add_custom_command(
        OUTPUT ${KERNEL_OUTPUT}
        COMMAND ${COMPILE_COMMAND}
        DEPENDS ${SOURCE_ABS}
        COMMENT "Compiling kernel: ${KERNEL_NAME}"
        VERBATIM
    )
    
    # Create a target for this kernel
    add_custom_target(${KERNEL_NAME}_kernel DEPENDS ${KERNEL_OUTPUT})
    
    # Store the output path for later use
    set(${KERNEL_NAME}_KERNEL_PATH ${KERNEL_OUTPUT} PARENT_SCOPE)
endfunction()

# Legacy function name for backward compatibility
function(compile_hip_kernel KERNEL_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "" "" "GPU_TARGETS" ${ARGN})
    if(ARG_GPU_TARGETS)
        compile_kernel(${KERNEL_NAME} ${SOURCE_FILE} GPU_TARGETS ${ARG_GPU_TARGETS})
    else()
        compile_kernel(${KERNEL_NAME} ${SOURCE_FILE})
    endif()
    set(${KERNEL_NAME}_HSACO_PATH "${KERNEL_GEN_DIR}/${KERNEL_NAME}${KERNEL_OUTPUT_EXTENSION}" PARENT_SCOPE)
endfunction()

# Function to embed a binary file as C++ source
# Usage: embed_binary(<name> <input_file> [NAMESPACE <namespace>])
function(embed_binary NAME INPUT_FILE)
    cmake_parse_arguments(ARG "" "NAMESPACE" "" ${ARGN})
    
    set(OUTPUT_BASE "${KERNEL_GEN_DIR}/${NAME}")
    set(OUTPUT_HPP "${OUTPUT_BASE}.hpp")
    set(OUTPUT_CPP "${OUTPUT_BASE}.cpp")
    
    set(NS_ARG "")
    if(ARG_NAMESPACE)
        set(NS_ARG "--namespace=${ARG_NAMESPACE}")
    endif()
    
    add_custom_command(
        OUTPUT ${OUTPUT_HPP} ${OUTPUT_CPP}
        COMMAND ${Python3_EXECUTABLE}
            ${HIP_CTS_EMBED_BINARY_SCRIPT}
            ${INPUT_FILE}
            ${OUTPUT_BASE}
            ${NS_ARG}
        DEPENDS 
            ${INPUT_FILE}
            ${HIP_CTS_EMBED_BINARY_SCRIPT}
        COMMENT "Embedding binary: ${NAME}"
        VERBATIM
    )
    
    # Store output paths for parent scope
    set(${NAME}_EMBEDDED_HPP ${OUTPUT_HPP} PARENT_SCOPE)
    set(${NAME}_EMBEDDED_CPP ${OUTPUT_CPP} PARENT_SCOPE)
endfunction()

# Function to compile and embed a kernel in one step
# Usage: add_embedded_kernel(<name> <source_file> [NAMESPACE <namespace>] [GPU_TARGETS target1 ...])
function(add_embedded_kernel NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "" "NAMESPACE" "GPU_TARGETS" ${ARGN})
    
    # First compile the kernel
    if(ARG_GPU_TARGETS)
        compile_kernel(${NAME} ${SOURCE_FILE} GPU_TARGETS ${ARG_GPU_TARGETS})
    else()
        compile_kernel(${NAME} ${SOURCE_FILE})
    endif()
    
    # Get the kernel path from the function
    set(KERNEL_PATH "${KERNEL_GEN_DIR}/${NAME}${KERNEL_OUTPUT_EXTENSION}")
    
    # Then embed it
    if(ARG_NAMESPACE)
        embed_binary(${NAME} ${KERNEL_PATH} NAMESPACE ${ARG_NAMESPACE})
    else()
        embed_binary(${NAME} ${KERNEL_PATH})
    endif()
    
    # Propagate the output paths
    set(${NAME}_KERNEL_PATH ${KERNEL_PATH} PARENT_SCOPE)
    set(${NAME}_HSACO_PATH ${KERNEL_PATH} PARENT_SCOPE)  # Legacy name
    set(${NAME}_EMBEDDED_HPP "${KERNEL_GEN_DIR}/${NAME}.hpp" PARENT_SCOPE)
    set(${NAME}_EMBEDDED_CPP "${KERNEL_GEN_DIR}/${NAME}.cpp" PARENT_SCOPE)
endfunction()

#==============================================================================
# High-Level Test Function
#==============================================================================

# Create a HIP CTS test executable, optionally with embedded kernels
# Usage: add_hip_cts_test(<name> 
#            SOURCES src1.cpp src2.cpp ...
#            [KERNELS kernel1.hip kernel2.hip ...]
#            [NAMESPACE <namespace>]
#            [GPU_TARGETS target1 target2 ...])
#
# When KERNELS is specified, this function:
#   1. Compiles each kernel file using the configured compiler
#   2. Embeds each compiled kernel as C++ source
#   3. Creates a static library with the embedded kernels
#   4. Creates the test executable linking against the kernel library
#   5. Registers the test with CTest
#
# When KERNELS is not specified:
#   - Creates a simple test executable without kernel dependencies
#   - Does not require a kernel compiler
#
# Generated headers are named: <test_name>_<kernel_basename>.hpp
# e.g., for test "kernel_smoke" and kernel "vector_add.hip":
#       kernel_smoke_vector_add.hpp with struct kernel_smoke_vector_add_data
function(add_hip_cts_test TARGET_NAME)
    cmake_parse_arguments(ARG "" "NAMESPACE" "SOURCES;KERNELS;GPU_TARGETS" ${ARGN})
    
    # Validate arguments
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_hip_cts_test: SOURCES is required")
    endif()
    
    # If kernels specified, require kernel compiler
    if(ARG_KERNELS AND NOT KERNEL_COMPILER_PATH)
        message(STATUS "${TARGET_NAME} disabled (kernel compiler not found)")
        return()
    endif()
    
    # Default namespace for kernels
    if(NOT ARG_NAMESPACE)
        set(ARG_NAMESPACE "hip_cts::kernels")
    endif()
    
    # Process kernels if specified
    set(KERNEL_LIB_LINK "")
    if(ARG_KERNELS)
        set(KERNEL_CPPS "")
        set(KERNEL_TARGETS "")
        
        foreach(KERNEL_FILE ${ARG_KERNELS})
            # Get kernel name from filename (without extension)
            get_filename_component(KERNEL_BASENAME "${KERNEL_FILE}" NAME_WE)
            set(KERNEL_NAME "${TARGET_NAME}_${KERNEL_BASENAME}")
            
            # Get absolute path
            if(NOT IS_ABSOLUTE "${KERNEL_FILE}")
                set(KERNEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}")
            endif()
            
            # Compile and embed the kernel
            if(ARG_GPU_TARGETS)
                add_embedded_kernel(${KERNEL_NAME} ${KERNEL_FILE} 
                    NAMESPACE ${ARG_NAMESPACE}
                    GPU_TARGETS ${ARG_GPU_TARGETS})
            else()
                add_embedded_kernel(${KERNEL_NAME} ${KERNEL_FILE} 
                    NAMESPACE ${ARG_NAMESPACE})
            endif()
            
            # Collect generated files
            list(APPEND KERNEL_CPPS "${${KERNEL_NAME}_EMBEDDED_CPP}")
            list(APPEND KERNEL_TARGETS "${KERNEL_NAME}_kernel")
        endforeach()
        
        # Create kernel library
        set(KERNEL_LIB "${TARGET_NAME}_kernels")
        add_library(${KERNEL_LIB} STATIC ${KERNEL_CPPS})
        target_include_directories(${KERNEL_LIB} PUBLIC ${KERNEL_GEN_DIR})
        add_dependencies(${KERNEL_LIB} ${KERNEL_TARGETS})
        set(KERNEL_LIB_LINK ${KERNEL_LIB})
    endif()
    
    # Create test executable
    add_executable(${TARGET_NAME} ${ARG_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE 
        hip_cts_main
        Catch2::Catch2
        hip_cts_core
        ${KERNEL_LIB_LINK}
    )
    
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()
