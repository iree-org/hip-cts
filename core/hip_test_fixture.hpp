// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include "hip_loader.hpp"

// Global variable to store the selected HIP device (set from command line)
// Defined in main.cpp
extern int g_hip_device;

// Global variable to indicate if we're using the streaming backend
// Defined in main.cpp
extern bool g_is_streaming_backend;

// Test fixture that automatically sets the HIP device before each test
// Usage: TEST_CASE_METHOD(HipTestFixture, "test name", "[tags]")
class HipTestFixture {
public:
    HipTestFixture() {
        // Set the device at the start of each test
        hipError_t err = hip().hipSetDevice(g_hip_device);
        if (err != hipSuccess) {
            FAIL("Failed to set HIP device " << g_hip_device << ": " 
                 << hip().hipGetErrorString(err));
        }
    }

    ~HipTestFixture() {
        // Synchronize device at the end of each test to ensure clean state
        hip().hipDeviceSynchronize();
    }

    // Check if we're running on the streaming backend (IREE)
    bool is_streaming_backend() const {
        return g_is_streaming_backend;
    }
};

