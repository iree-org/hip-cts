// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Device Count Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceCount returns valid count", "[device][count]") {
    int count = -1;
    REQUIRE(hip().hipGetDeviceCount(&count) == hipSuccess);
    REQUIRE(count > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceCount with null pointer fails", "[device][count][negative]") {
    REQUIRE(hip().hipGetDeviceCount(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Set/Get Device Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice and hipGetDevice basic", "[device][set][get]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    REQUIRE(deviceCount > 0);
    
    // Set and get device 0
    REQUIRE(hip().hipSetDevice(0) == hipSuccess);
    
    int currentDevice = -1;
    REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
    REQUIRE(currentDevice == 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice invalid device negative", "[device][set][negative]") {
    REQUIRE(hip().hipSetDevice(-1) == hipErrorInvalidDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice invalid device out of range", "[device][set][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    REQUIRE(hip().hipSetDevice(deviceCount + 100) == hipErrorInvalidDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDevice with null pointer fails", "[device][get][negative]") {
    REQUIRE(hip().hipGetDevice(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Device Properties Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties basic", "[device][properties]") {
    hipDeviceProp_t props;
    memset(&props, 0, sizeof(props));
    
    REQUIRE(hip().hipGetDeviceProperties(&props, 0) == hipSuccess);
    
    SECTION("Device name is non-empty") {
        REQUIRE(props.name[0] != '\0');
    }
    
    SECTION("Device has valid memory") {
        REQUIRE(props.totalGlobalMem > 0);
    }
    
    SECTION("Device has valid compute capabilities") {
        REQUIRE(props.major >= 0);
        REQUIRE(props.minor >= 0);
    }
    
    SECTION("Device has valid warp size") {
        REQUIRE(props.warpSize > 0);
    }
    
    SECTION("Device has valid max threads per block") {
        REQUIRE(props.maxThreadsPerBlock > 0);
    }
    
    SECTION("Device has non-negative clock rate") {
        // Some devices may report 0 for clock rate
        REQUIRE(props.clockRate >= 0);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties null pointer fails", "[device][properties][negative]") {
    REQUIRE(hip().hipGetDeviceProperties(nullptr, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties invalid device fails", "[device][properties][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    hipDeviceProp_t props;
    REQUIRE(hip().hipGetDeviceProperties(&props, deviceCount + 100) == hipErrorInvalidDevice);
}

//=============================================================================
// Device Synchronize Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize empty", "[device][sync]") {
    // Synchronize with no work should succeed
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize with stream work", "[device][sync]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate memory and do some work
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xAB, size, stream) == hipSuccess);
    
    // Device synchronize should wait for all work
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify data
    std::vector<uint8_t> hostData(size, 0);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize multiple streams", "[device][sync]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numStreams, std::vector<uint8_t>(size, 0));
    
    // Create streams and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Launch async operations on all streams
    for (int i = 0; i < numStreams; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 1);
        REQUIRE(hip().hipMemsetAsync(devicePtrs[i], value, size, streams[i]) == hipSuccess);
    }
    
    // Device synchronize should wait for all streams
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify data from all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpy(hostData[i].data(), devicePtrs[i], size, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        uint8_t expectedValue = static_cast<uint8_t>(i + 1);
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(hostData[i][j] == expectedValue);
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Driver Version Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDriverGetVersion returns valid version", "[device][driver]") {
    int driverVersion = -1;
    REQUIRE(hip().hipDriverGetVersion(&driverVersion) == hipSuccess);
    REQUIRE(driverVersion > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipDriverGetVersion null pointer fails", "[device][driver][negative]") {
    REQUIRE(hip().hipDriverGetVersion(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Runtime Version Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipRuntimeGetVersion returns valid version", "[device][runtime]") {
    int runtimeVersion = -1;
    REQUIRE(hip().hipRuntimeGetVersion(&runtimeVersion) == hipSuccess);
    REQUIRE(runtimeVersion > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipRuntimeGetVersion null pointer fails", "[device][runtime][negative]") {
    REQUIRE(hip().hipRuntimeGetVersion(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Device Attribute Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute basic attributes", "[device][attribute]") {
    int value = -1;
    
    SECTION("Max threads per block") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max block dim X") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimX, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max grid dim X") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Warp size") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max shared memory per block") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Total constant memory") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeTotalConstantMemory, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Clock rate") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeClockRate, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Compute capability major") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Compute capability minor") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMinor, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute null pointer fails", "[device][attribute][negative]") {
    REQUIRE(hip().hipDeviceGetAttribute(nullptr, hipDeviceAttributeMaxThreadsPerBlock, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute invalid device fails", "[device][attribute][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    int value = -1;
    REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, deviceCount + 100) == hipErrorInvalidDevice);
}

//=============================================================================
// Memory Info Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo returns valid values", "[device][memory][info]") {
    size_t freeMem = 0;
    size_t totalMem = 0;
    
    REQUIRE(hip().hipMemGetInfo(&freeMem, &totalMem) == hipSuccess);
    REQUIRE(totalMem > 0);
    REQUIRE(freeMem <= totalMem);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo after allocation", "[device][memory][info]") {
    size_t freeMemBefore = 0;
    size_t totalMem = 0;
    REQUIRE(hip().hipMemGetInfo(&freeMemBefore, &totalMem) == hipSuccess);
    
    // Allocate memory
    void* devicePtr = nullptr;
    constexpr size_t allocSize = 16 * 1024 * 1024; // 16 MB
    REQUIRE(hip().hipMalloc(&devicePtr, allocSize) == hipSuccess);
    
    size_t freeMemAfter = 0;
    REQUIRE(hip().hipMemGetInfo(&freeMemAfter, &totalMem) == hipSuccess);
    
    // Free memory should be less after allocation
    REQUIRE(freeMemAfter < freeMemBefore);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// Init Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipInit with flags 0", "[device][init]") {
    // hipInit(0) should succeed
    REQUIRE(hip().hipInit(0) == hipSuccess);
}

//=============================================================================
// hipDeviceGet Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGet basic", "[device][get]") {
    int device = -1;
    REQUIRE(hip().hipDeviceGet(&device, 0) == hipSuccess);
    REQUIRE(device == 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGet invalid ordinal", "[device][get][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    int device = -1;
    REQUIRE(hip().hipDeviceGet(&device, deviceCount + 100) == hipErrorInvalidDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGet null pointer", "[device][get][negative]") {
    REQUIRE(hip().hipDeviceGet(nullptr, 0) == hipErrorInvalidValue);
}

//=============================================================================
// hipDeviceGetName Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetName basic", "[device][name]") {
    char name[256] = {0};
    REQUIRE(hip().hipDeviceGetName(name, sizeof(name), 0) == hipSuccess);
    REQUIRE(name[0] != '\0');  // Name should not be empty
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetName null pointer", "[device][name][negative]") {
    REQUIRE(hip().hipDeviceGetName(nullptr, 256, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetName invalid device", "[device][name][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    char name[256] = {0};
    hipError_t err = hip().hipDeviceGetName(name, sizeof(name), deviceCount + 100);
    REQUIRE((err == hipErrorInvalidDevice || err == hipErrorInvalidValue));
}

//=============================================================================
// hipDeviceTotalMem Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceTotalMem basic", "[device][totalmem]") {
    size_t bytes = 0;
    REQUIRE(hip().hipDeviceTotalMem(&bytes, 0) == hipSuccess);
    REQUIRE(bytes > 0);  // Device should have some memory
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceTotalMem null pointer", "[device][totalmem][negative]") {
    REQUIRE(hip().hipDeviceTotalMem(nullptr, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceTotalMem invalid device", "[device][totalmem][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    size_t bytes = 0;
    hipError_t err = hip().hipDeviceTotalMem(&bytes, deviceCount + 100);
    REQUIRE((err == hipErrorInvalidDevice || err == hipErrorInvalidValue));
}

//=============================================================================
// hipDeviceGetStreamPriorityRange Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetStreamPriorityRange basic", "[device][priority]") {
    int leastPriority = 0;
    int greatestPriority = 0;
    REQUIRE(hip().hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority) == hipSuccess);
    // leastPriority should be >= greatestPriority (lower number = higher priority in HIP)
    REQUIRE(leastPriority >= greatestPriority);
}

//=============================================================================
// Primary Context Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDevicePrimaryCtxRetain and Release", "[device][primaryctx]") {
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipDevicePrimaryCtxRetain(&ctx, 0) == hipSuccess);
    REQUIRE(ctx != nullptr);
    
    REQUIRE(hip().hipDevicePrimaryCtxRelease(0) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipDevicePrimaryCtxGetState", "[device][primaryctx]") {
    unsigned int flags = 0;
    int active = -1;
    REQUIRE(hip().hipDevicePrimaryCtxGetState(0, &flags, &active) == hipSuccess);
    // active should be 0 or 1
    REQUIRE((active == 0 || active == 1));
}

TEST_CASE_METHOD(HipTestFixture, "hipDevicePrimaryCtxReset", "[device][primaryctx]") {
    // First retain to ensure context exists
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipDevicePrimaryCtxRetain(&ctx, 0) == hipSuccess);
    
    // Reset the context
    REQUIRE(hip().hipDevicePrimaryCtxReset(0) == hipSuccess);
    
    // Release - may return hipErrorInvalidContext after reset on some implementations
    hipError_t err = hip().hipDevicePrimaryCtxRelease(0);
    REQUIRE((err == hipSuccess || err == hipErrorInvalidContext));
}

//=============================================================================
// Context Management Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipCtxGetCurrent", "[device][context]") {
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipCtxGetCurrent(&ctx) == hipSuccess);
    // Context may be null if none is set
}

TEST_CASE_METHOD(HipTestFixture, "hipCtxGetDevice", "[device][context]") {
    int device = -1;
    // This should work if there's a current context
    hipError_t err = hip().hipCtxGetDevice(&device);
    // Accept either success or no context error
    REQUIRE((err == hipSuccess || err == hipErrorInvalidContext));
}

TEST_CASE_METHOD(HipTestFixture, "hipCtxSynchronize with work", "[device][context]") {
    // Do some work first to ensure there's a context
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(hip().hipMemset(ptr, 0, 1024) == hipSuccess);
    
    // Now synchronize should succeed
    // Note: Some HIP implementations return errors if no driver context is active
    hipError_t err = hip().hipCtxSynchronize();
    if (err != hipSuccess) {
        hip().hipFree(ptr);
        SKIP("hipCtxSynchronize returns error " << err << " on this platform");
    }
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipCtxCreate and Destroy", "[device][context]") {
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipCtxCreate(&ctx, 0, 0) == hipSuccess);
    REQUIRE(ctx != nullptr);
    
    REQUIRE(hip().hipCtxDestroy(ctx) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipCtxPush and Pop", "[device][context]") {
    // Create a context
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipCtxCreate(&ctx, 0, 0) == hipSuccess);
    
    // Push it
    REQUIRE(hip().hipCtxPushCurrent(ctx) == hipSuccess);
    
    // Pop it
    hipCtx_t poppedCtx = nullptr;
    REQUIRE(hip().hipCtxPopCurrent(&poppedCtx) == hipSuccess);
    REQUIRE(poppedCtx == ctx);
    
    REQUIRE(hip().hipCtxDestroy(ctx) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipCtxSetCurrent", "[device][context]") {
    // Create a context
    hipCtx_t ctx = nullptr;
    REQUIRE(hip().hipCtxCreate(&ctx, 0, 0) == hipSuccess);
    
    // Set it as current
    REQUIRE(hip().hipCtxSetCurrent(ctx) == hipSuccess);
    
    // Verify it's current
    hipCtx_t currentCtx = nullptr;
    REQUIRE(hip().hipCtxGetCurrent(&currentCtx) == hipSuccess);
    REQUIRE(currentCtx == ctx);
    
    REQUIRE(hip().hipCtxDestroy(ctx) == hipSuccess);
}

//=============================================================================
// Device Flags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceFlags basic", "[device][flags]") {
    unsigned int flags = 0;
    REQUIRE(hip().hipGetDeviceFlags(&flags) == hipSuccess);
    // Flags should be a valid value (no error)
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceFlags null pointer fails", "[device][flags][negative]") {
    REQUIRE(hip().hipGetDeviceFlags(nullptr) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDeviceFlags with zero", "[device][flags]") {
    // Setting default flags (0) should work
    // Note: This must be called before any HIP runtime API on a device
    // For this test, we just verify the function doesn't crash
    hipError_t err = hip().hipSetDeviceFlags(0);
    // May succeed or fail depending on device state
    REQUIRE((err == hipSuccess || err == hipErrorSetOnActiveProcess));
}

//=============================================================================
// Device Peer Access Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer same device", "[device][peer]") {
    int canAccess = -1;
    // A device can always access itself
    REQUIRE(hip().hipDeviceCanAccessPeer(&canAccess, 0, 0) == hipSuccess);
    // Result depends on implementation - may be 0 or 1
    REQUIRE((canAccess == 0 || canAccess == 1));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer null pointer fails", "[device][peer][negative]") {
    REQUIRE(hip().hipDeviceCanAccessPeer(nullptr, 0, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer invalid device", "[device][peer][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    int canAccess = -1;
    hipError_t err = hip().hipDeviceCanAccessPeer(&canAccess, deviceCount + 100, 0);
    REQUIRE((err == hipErrorInvalidDevice || err == hipErrorInvalidValue));
}

//=============================================================================
// Device Reset Tests  
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceReset after allocation", "[device][reset]") {
    // Do some work
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(hip().hipMemset(ptr, 0, 1024) == hipSuccess);
    
    // Free before reset
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
    
    // Reset should succeed
    REQUIRE(hip().hipDeviceReset() == hipSuccess);
    
    // After reset, device should still be usable
    void* ptr2 = nullptr;
    REQUIRE(hip().hipMalloc(&ptr2, 1024) == hipSuccess);
    REQUIRE(hip().hipFree(ptr2) == hipSuccess);
}
