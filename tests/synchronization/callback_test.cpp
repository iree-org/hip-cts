// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <atomic>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Helper structures and callbacks
//=============================================================================

struct CallbackData {
    std::atomic<bool> called{false};
    std::atomic<int> value{0};
    hipStream_t stream{nullptr};
    hipError_t status{hipSuccess};
};

static void simpleCallback(hipStream_t stream, hipError_t status, void* userData) {
    CallbackData* data = static_cast<CallbackData*>(userData);
    data->called = true;
    data->stream = stream;
    data->status = status;
}

static void incrementCallback(hipStream_t stream, hipError_t status, void* userData) {
    CallbackData* data = static_cast<CallbackData*>(userData);
    data->value++;
    data->called = true;
}

static void simpleHostFunc(void* userData) {
    CallbackData* data = static_cast<CallbackData*>(userData);
    data->called = true;
}

static void incrementHostFunc(void* userData) {
    CallbackData* data = static_cast<CallbackData*>(userData);
    data->value++;
    data->called = true;
}

//=============================================================================
// hipStreamAddCallback Tests
// Note: hipStreamAddCallback may not be implemented in all backends
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamAddCallback basic", "[callback][stream]") {
    // Check if hipStreamAddCallback is available
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    hipStream_t stream = nullptr;
    CallbackData data;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Add callback
    REQUIRE(hip().hipStreamAddCallback(stream, simpleCallback, &data, 0) == hipSuccess);
    
    // Sync to ensure callback executes
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify callback was called
    REQUIRE(data.called == true);
    REQUIRE(data.stream == stream);
    REQUIRE(data.status == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamAddCallback on null stream", "[callback][stream]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    CallbackData data;
    
    // Add callback to null stream (default stream)
    REQUIRE(hip().hipStreamAddCallback(nullptr, simpleCallback, &data, 0) == hipSuccess);
    
    // Sync device
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify callback was called
    REQUIRE(data.called == true);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamAddCallback after work", "[callback][stream]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    hipStream_t stream = nullptr;
    void* devicePtr = nullptr;
    CallbackData data;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Do some work first
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xAB, size, stream) == hipSuccess);
    
    // Then add callback
    REQUIRE(hip().hipStreamAddCallback(stream, simpleCallback, &data, 0) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify callback was called after work completed
    REQUIRE(data.called == true);
    
    // Also verify work was done
    std::vector<uint8_t> hostData(size);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamAddCallback multiple callbacks", "[callback][stream]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    hipStream_t stream = nullptr;
    constexpr int numCallbacks = 5;
    std::vector<CallbackData> data(numCallbacks);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Add multiple callbacks
    for (int i = 0; i < numCallbacks; ++i) {
        REQUIRE(hip().hipStreamAddCallback(stream, incrementCallback, &data[i], 0) == hipSuccess);
    }
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify all callbacks were called
    for (int i = 0; i < numCallbacks; ++i) {
        REQUIRE(data[i].called == true);
        REQUIRE(data[i].value == 1);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamAddCallback null callback fails", "[callback][stream][negative]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamAddCallback(stream, nullptr, nullptr, 0) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipLaunchHostFunc Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipLaunchHostFunc basic", "[callback][hostfunc]") {
    hipStream_t stream = nullptr;
    CallbackData data;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Launch host function
    REQUIRE(hip().hipLaunchHostFunc(stream, simpleHostFunc, &data) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify function was called
    REQUIRE(data.called == true);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipLaunchHostFunc on null stream", "[callback][hostfunc]") {
    CallbackData data;
    
    // Launch on null stream
    REQUIRE(hip().hipLaunchHostFunc(nullptr, simpleHostFunc, &data) == hipSuccess);
    
    // Sync device
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify function was called
    REQUIRE(data.called == true);
}

TEST_CASE_METHOD(HipTestFixture, "hipLaunchHostFunc after work", "[callback][hostfunc]") {
    hipStream_t stream = nullptr;
    void* devicePtr = nullptr;
    CallbackData data;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Do some work first
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xCD, size, stream) == hipSuccess);
    
    // Then launch host function
    REQUIRE(hip().hipLaunchHostFunc(stream, simpleHostFunc, &data) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify function was called after work completed
    REQUIRE(data.called == true);
    
    // Also verify work was done
    std::vector<uint8_t> hostData(size);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipLaunchHostFunc multiple functions", "[callback][hostfunc]") {
    hipStream_t stream = nullptr;
    constexpr int numFunctions = 5;
    std::vector<CallbackData> data(numFunctions);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Launch multiple host functions
    for (int i = 0; i < numFunctions; ++i) {
        REQUIRE(hip().hipLaunchHostFunc(stream, incrementHostFunc, &data[i]) == hipSuccess);
    }
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify all functions were called
    for (int i = 0; i < numFunctions; ++i) {
        REQUIRE(data[i].called == true);
        REQUIRE(data[i].value == 1);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipLaunchHostFunc null function fails", "[callback][hostfunc][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipLaunchHostFunc(stream, nullptr, nullptr) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Mixed Callback and Work Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Interleaved work and callbacks", "[callback][stream]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    hipStream_t stream = nullptr;
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<CallbackData> callbackData(3);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Work -> Callback -> Work -> Callback -> Work -> Callback
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x11, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamAddCallback(stream, incrementCallback, &callbackData[0], 0) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x22, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamAddCallback(stream, incrementCallback, &callbackData[1], 0) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x33, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamAddCallback(stream, incrementCallback, &callbackData[2], 0) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify all callbacks were called
    for (int i = 0; i < 3; ++i) {
        REQUIRE(callbackData[i].called == true);
    }
    
    // Final data should be from last memset
    std::vector<uint8_t> hostData(size);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x33);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Callbacks on multiple streams", "[callback][stream]") {
    if (!hip().hipStreamAddCallback) {
        SKIP("hipStreamAddCallback not available");
    }
    
    constexpr int numStreams = 4;
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<CallbackData> callbackData(numStreams);
    
    // Create streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
    }
    
    // Add callbacks to each stream
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamAddCallback(streams[i], simpleCallback, &callbackData[i], 0) == hipSuccess);
    }
    
    // Sync all
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Verify all callbacks were called with correct stream
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(callbackData[i].called == true);
        REQUIRE(callbackData[i].stream == streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}
