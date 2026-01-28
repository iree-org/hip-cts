// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <thread>
#include <chrono>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Stream Creation Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreate creates valid stream", "[stream][create]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(stream != nullptr);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreate with null pointer fails", "[stream][create][negative]") {
    REQUIRE(hip().hipStreamCreate(nullptr) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreate multiple streams", "[stream][create]") {
    constexpr int numStreams = 10;
    std::vector<hipStream_t> streams(numStreams, nullptr);
    
    // Create multiple streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(streams[i] != nullptr);
    }
    
    // Verify all streams are unique
    for (int i = 0; i < numStreams; ++i) {
        for (int j = i + 1; j < numStreams; ++j) {
            REQUIRE(streams[i] != streams[j]);
        }
    }
    
    // Destroy all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Stream Creation with Flags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreateWithFlags default", "[stream][create][flags]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreateWithFlags(&stream, hipStreamDefault) == hipSuccess);
    REQUIRE(stream != nullptr);
    
    unsigned int flags = 0;
    REQUIRE(hip().hipStreamGetFlags(stream, &flags) == hipSuccess);
    REQUIRE((flags & hipStreamNonBlocking) == 0);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreateWithFlags non-blocking", "[stream][create][flags]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
    REQUIRE(stream != nullptr);
    
    unsigned int flags = 0;
    REQUIRE(hip().hipStreamGetFlags(stream, &flags) == hipSuccess);
    REQUIRE((flags & hipStreamNonBlocking) != 0);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreateWithFlags null pointer fails", "[stream][create][flags][negative]") {
    REQUIRE(hip().hipStreamCreateWithFlags(nullptr, hipStreamDefault) == hipErrorInvalidValue);
}

//=============================================================================
// Stream Destruction Tests  
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamDestroy valid stream", "[stream][destroy]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamDestroy null stream fails", "[stream][destroy][negative]") {
    // Destroying null stream (default stream) should return an error
    // The default stream is not a user-created stream and cannot be destroyed
    REQUIRE(hip().hipStreamDestroy(nullptr) == hipErrorInvalidResourceHandle);
}

//=============================================================================
// Stream Query Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamQuery empty stream returns success", "[stream][query]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Empty stream should be ready
    REQUIRE(hip().hipStreamQuery(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamQuery null stream returns success", "[stream][query]") {
    // Null stream (default stream) with no work should be ready
    REQUIRE(hip().hipStreamQuery(nullptr) == hipSuccess);
}

//=============================================================================
// Stream Synchronize Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize empty stream", "[stream][sync]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Synchronizing empty stream should succeed immediately
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize null stream", "[stream][sync]") {
    // Synchronizing null stream should succeed
    REQUIRE(hip().hipStreamSynchronize(nullptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize with work", "[stream][sync]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate memory
    void* d_ptr = nullptr;
    constexpr size_t size = 1024;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Do an async memset
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0, size, stream) == hipSuccess);
    
    // Synchronize should wait for memset to complete
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // After sync, stream should be ready
    REQUIRE(hip().hipStreamQuery(stream) == hipSuccess);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Stream Flags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamGetFlags returns correct flags", "[stream][flags]") {
    hipStream_t stream = nullptr;
    unsigned int flags = 0;
    
    SECTION("Default stream flags") {
        REQUIRE(hip().hipStreamCreateWithFlags(&stream, hipStreamDefault) == hipSuccess);
        REQUIRE(hip().hipStreamGetFlags(stream, &flags) == hipSuccess);
        REQUIRE((flags & hipStreamNonBlocking) == 0);
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    }
    
    SECTION("Non-blocking stream flags") {
        REQUIRE(hip().hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
        REQUIRE(hip().hipStreamGetFlags(stream, &flags) == hipSuccess);
        REQUIRE((flags & hipStreamNonBlocking) != 0);
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamGetFlags null pointer fails", "[stream][flags][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamGetFlags(stream, nullptr) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Stream Priority Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreateWithPriority default priority", "[stream][priority]") {
    hipStream_t stream = nullptr;
    // Use priority 0 (default)
    REQUIRE(hip().hipStreamCreateWithPriority(&stream, hipStreamDefault, 0) == hipSuccess);
    REQUIRE(stream != nullptr);
    
    int retrievedPriority = -1;
    REQUIRE(hip().hipStreamGetPriority(stream, &retrievedPriority) == hipSuccess);
    REQUIRE(retrievedPriority == 0);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamGetPriority null pointer fails", "[stream][priority][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamGetPriority(stream, nullptr) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Stream Wait Event Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamWaitEvent basic", "[stream][event][wait]") {
    hipStream_t stream1 = nullptr;
    hipStream_t stream2 = nullptr;
    hipEvent_t event = nullptr;
    
    REQUIRE(hip().hipStreamCreate(&stream1) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream2) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    // Record event on stream1
    REQUIRE(hip().hipEventRecord(event, stream1) == hipSuccess);
    
    // Make stream2 wait for the event
    REQUIRE(hip().hipStreamWaitEvent(stream2, event, 0) == hipSuccess);
    
    // Synchronize both streams
    REQUIRE(hip().hipStreamSynchronize(stream1) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream2) == hipSuccess);
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream2) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream1) == hipSuccess);
}

//=============================================================================
// Stream with Memory Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Stream memcpy operations", "[stream][memory]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostSrc(size, 0xAB);
    std::vector<uint8_t> hostDst(size, 0);
    
    void* devicePtr = nullptr;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Copy to device async
    REQUIRE(hip().hipMemcpyAsync(devicePtr, hostSrc.data(), size, 
                                  hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Copy back to host async
    REQUIRE(hip().hipMemcpyAsync(hostDst.data(), devicePtr, size,
                                  hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify data
    REQUIRE(hostDst == hostSrc);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Stream memset operations", "[stream][memory]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    void* devicePtr = nullptr;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Memset async
    uint8_t value = 0xCD;
    REQUIRE(hip().hipMemsetAsync(devicePtr, value, size, stream) == hipSuccess);
    
    // Copy back to verify
    REQUIRE(hip().hipMemcpyAsync(hostData.data(), devicePtr, size,
                                  hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify all bytes are set correctly
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == value);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Multiple Streams Concurrent Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multiple streams concurrent operations", "[stream][concurrent]") {
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
        REQUIRE(hip().hipMemcpyAsync(hostData[i].data(), devicePtrs[i], size,
                                      hipMemcpyDeviceToHost, streams[i]) == hipSuccess);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Verify data
    for (int i = 0; i < numStreams; ++i) {
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
// hipGetStreamDeviceId Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetStreamDeviceId for created stream", "[stream][device]") {
    int currentDevice = -1;
    REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
    
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    int streamDevice = hip().hipGetStreamDeviceId(stream);
    REQUIRE(streamDevice == currentDevice);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetStreamDeviceId for null stream", "[stream][device]") {
    int currentDevice = -1;
    REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
    
    int streamDevice = hip().hipGetStreamDeviceId(nullptr);
    REQUIRE(streamDevice == currentDevice);
}
