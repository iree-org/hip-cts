// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cmath>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Event Creation Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventCreate creates valid event", "[event][create]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventCreate with null pointer fails", "[event][create][negative]") {
    REQUIRE(hip().hipEventCreate(nullptr) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventCreate multiple events", "[event][create]") {
    constexpr int numEvents = 10;
    std::vector<hipEvent_t> events(numEvents, nullptr);
    
    // Create multiple events
    for (int i = 0; i < numEvents; ++i) {
        REQUIRE(hip().hipEventCreate(&events[i]) == hipSuccess);
        REQUIRE(events[i] != nullptr);
    }
    
    // Verify all events are unique
    for (int i = 0; i < numEvents; ++i) {
        for (int j = i + 1; j < numEvents; ++j) {
            REQUIRE(events[i] != events[j]);
        }
    }
    
    // Destroy all events
    for (int i = 0; i < numEvents; ++i) {
        REQUIRE(hip().hipEventDestroy(events[i]) == hipSuccess);
    }
}

//=============================================================================
// Event Creation with Flags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventCreateWithFlags default", "[event][create][flags]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreateWithFlags(&event, hipEventDefault) == hipSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventCreateWithFlags blocking sync", "[event][create][flags]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreateWithFlags(&event, hipEventBlockingSync) == hipSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventCreateWithFlags disable timing", "[event][create][flags]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreateWithFlags(&event, hipEventDisableTiming) == hipSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventCreateWithFlags null pointer fails", "[event][create][flags][negative]") {
    REQUIRE(hip().hipEventCreateWithFlags(nullptr, hipEventDefault) == hipErrorInvalidValue);
}

//=============================================================================
// Event Destruction Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventDestroy valid event", "[event][destroy]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Event Record Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventRecord on null stream", "[event][record]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(event, nullptr) == hipSuccess);
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventRecord on created stream", "[event][record]") {
    hipEvent_t event = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventRecord multiple times", "[event][record]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    // Record event multiple times
    for (int i = 0; i < 5; ++i) {
        REQUIRE(hip().hipEventRecord(event, nullptr) == hipSuccess);
    }
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Event Query Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventQuery after record and sync", "[event][query]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(event, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    // After synchronization, event should be complete
    REQUIRE(hip().hipEventQuery(event) == hipSuccess);
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventQuery unrecorded event", "[event][query]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    // Query on unrecorded event - behavior may vary
    // Some implementations return success, others return an error
    hipError_t result = hip().hipEventQuery(event);
    REQUIRE((result == hipSuccess || result == hipErrorNotReady || 
             result == hipErrorInvalidResourceHandle));
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Event Synchronize Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventSynchronize after record", "[event][sync]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(event, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventSynchronize unrecorded event", "[event][sync]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    // Synchronize on unrecorded event should return immediately
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventSynchronize with stream work", "[event][sync]") {
    hipEvent_t event = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate and do some work
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xAB, size, stream) == hipSuccess);
    
    // Record event after work
    REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
    
    // Synchronize on event should wait for work to complete
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    // Event should now be complete
    REQUIRE(hip().hipEventQuery(event) == hipSuccess);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Event Elapsed Time Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventElapsedTime basic", "[event][elapsed]") {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    
    REQUIRE(hip().hipEventCreate(&start) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&stop) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(start, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(start) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(stop, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(stop) == hipSuccess);
    
    float elapsedMs = -1.0f;
    REQUIRE(hip().hipEventElapsedTime(&elapsedMs, start, stop) == hipSuccess);
    REQUIRE(elapsedMs >= 0.0f);
    
    REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventElapsedTime with work", "[event][elapsed]") {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipEventCreate(&start) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&stop) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate memory for work
    void* devicePtr = nullptr;
    constexpr size_t size = 1024 * 1024; // 1 MB
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(start, stream) == hipSuccess);
    
    // Do some work
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0, size, stream) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(stop, stream) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(stop) == hipSuccess);
    
    float elapsedMs = -1.0f;
    REQUIRE(hip().hipEventElapsedTime(&elapsedMs, start, stop) == hipSuccess);
    REQUIRE(elapsedMs >= 0.0f);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventElapsedTime null pointer fails", "[event][elapsed][negative]") {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    
    REQUIRE(hip().hipEventCreate(&start) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&stop) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(start, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventRecord(stop, nullptr) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(stop) == hipSuccess);
    
    // Null ms pointer should fail
    REQUIRE(hip().hipEventElapsedTime(nullptr, start, stop) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventElapsedTime with disable timing flag fails", "[event][elapsed][negative]") {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    
    REQUIRE(hip().hipEventCreateWithFlags(&start, hipEventDisableTiming) == hipSuccess);
    REQUIRE(hip().hipEventCreateWithFlags(&stop, hipEventDisableTiming) == hipSuccess);
    
    float elapsedMs = 0.0f;
    REQUIRE(hip().hipEventElapsedTime(&elapsedMs, start, stop) == hipErrorInvalidHandle);
    
    REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
}

//=============================================================================
// Stream Wait Event Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamWaitEvent basic dependency", "[event][stream][wait]") {
    hipEvent_t event = nullptr;
    hipStream_t stream1 = nullptr;
    hipStream_t stream2 = nullptr;
    
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream1) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream2) == hipSuccess);
    
    // Allocate memory
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Stream 1: memset
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xCD, size, stream1) == hipSuccess);
    
    // Record event on stream 1
    REQUIRE(hip().hipEventRecord(event, stream1) == hipSuccess);
    
    // Stream 2 waits for event from stream 1
    REQUIRE(hip().hipStreamWaitEvent(stream2, event, 0) == hipSuccess);
    
    // Stream 2: copy to host (should see memset data)
    REQUIRE(hip().hipMemcpyAsync(hostData.data(), devicePtr, size, 
                                  hipMemcpyDeviceToHost, stream2) == hipSuccess);
    
    // Synchronize stream 2
    REQUIRE(hip().hipStreamSynchronize(stream2) == hipSuccess);
    
    // Verify data
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream2) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream1) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Event with Memory Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Event timing for memcpy operations", "[event][memory][timing]") {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipEventCreate(&start) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&stop) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr size_t size = 64 * 1024; // 64 KB
    std::vector<uint8_t> hostSrc(size, 0xAB);
    std::vector<uint8_t> hostDst(size, 0);
    
    void* devicePtr = nullptr;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Time the H2D -> D2H round trip
    REQUIRE(hip().hipEventRecord(start, stream) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyAsync(devicePtr, hostSrc.data(), size, 
                                  hipMemcpyHostToDevice, stream) == hipSuccess);
    REQUIRE(hip().hipMemcpyAsync(hostDst.data(), devicePtr, size,
                                  hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    REQUIRE(hip().hipEventRecord(stop, stream) == hipSuccess);
    REQUIRE(hip().hipEventSynchronize(stop) == hipSuccess);
    
    float elapsedMs = -1.0f;
    REQUIRE(hip().hipEventElapsedTime(&elapsedMs, start, stop) == hipSuccess);
    REQUIRE(elapsedMs >= 0.0f);
    
    // Verify data integrity
    REQUIRE(hostDst == hostSrc);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
}

//=============================================================================
// Multiple Events on Multiple Streams Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multiple events on multiple streams", "[event][stream][concurrent]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<hipEvent_t> events(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numStreams, std::vector<uint8_t>(size, 0));
    
    // Create streams, events, and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipEventCreate(&events[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Launch operations and record events
    for (int i = 0; i < numStreams; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 1);
        REQUIRE(hip().hipMemsetAsync(devicePtrs[i], value, size, streams[i]) == hipSuccess);
        REQUIRE(hip().hipEventRecord(events[i], streams[i]) == hipSuccess);
    }
    
    // Wait for all events
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipEventSynchronize(events[i]) == hipSuccess);
        REQUIRE(hip().hipEventQuery(events[i]) == hipSuccess);
    }
    
    // Copy data back and verify
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
        REQUIRE(hip().hipEventDestroy(events[i]) == hipSuccess);
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}
