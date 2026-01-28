// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include <thread>
#include <chrono>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Basic Synchronization Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize after memset", "[sync][device]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Async memset
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xAB, size, nullptr) == hipSuccess);
    
    // Device sync
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify data is ready
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize after memset", "[sync][stream]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Async memset on stream
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xCD, size, stream) == hipSuccess);
    
    // Stream sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify data is ready
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Event Synchronization Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventSynchronize after memset", "[sync][event]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Async memset on stream
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xEF, size, stream) == hipSuccess);
    
    // Record event
    REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
    
    // Event sync
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    // Verify data is ready
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamWaitEvent cross-stream sync", "[sync][event][stream]") {
    void* devicePtr = nullptr;
    hipStream_t stream1 = nullptr;
    hipStream_t stream2 = nullptr;
    hipEvent_t event = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream1) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream2) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Memset on stream1
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x12, size, stream1) == hipSuccess);
    
    // Record event on stream1
    REQUIRE(hip().hipEventRecord(event, stream1) == hipSuccess);
    
    // Stream2 waits for event
    REQUIRE(hip().hipStreamWaitEvent(stream2, event, 0) == hipSuccess);
    
    // Memcpy on stream2 should wait for memset to complete
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMemcpyAsync(hostPtr, devicePtr, size, hipMemcpyDeviceToHost, stream2) == hipSuccess);
    
    // Sync stream2
    REQUIRE(hip().hipStreamSynchronize(stream2) == hipSuccess);
    
    // Verify
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostBytes[i] == 0x12);
    }
    
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream1) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream2) == hipSuccess);
}

//=============================================================================
// Multiple Stream Synchronization Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize waits for all streams", "[sync][device][stream]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numStreams);
    
    // Create streams and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        hostData[i].resize(size, 0);
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Launch async memsets on all streams
    for (int i = 0; i < numStreams; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 1);
        REQUIRE(hip().hipMemsetAsync(devicePtrs[i], value, size, streams[i]) == hipSuccess);
    }
    
    // Single device sync should wait for all
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify all are complete
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

TEST_CASE_METHOD(HipTestFixture, "Independent streams execute independently", "[sync][stream]") {
    hipStream_t stream1 = nullptr;
    hipStream_t stream2 = nullptr;
    void* devicePtr1 = nullptr;
    void* devicePtr2 = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData1(size, 0);
    std::vector<uint8_t> hostData2(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream1) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream2) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr1, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr2, size) == hipSuccess);
    
    // Launch operations on both streams
    REQUIRE(hip().hipMemsetAsync(devicePtr1, 0xAA, size, stream1) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr2, 0xBB, size, stream2) == hipSuccess);
    
    // Sync stream1 only
    REQUIRE(hip().hipStreamSynchronize(stream1) == hipSuccess);
    
    // Verify stream1 data is ready
    REQUIRE(hip().hipMemcpy(hostData1.data(), devicePtr1, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData1[i] == 0xAA);
    }
    
    // Now sync stream2
    REQUIRE(hip().hipStreamSynchronize(stream2) == hipSuccess);
    
    // Verify stream2 data
    REQUIRE(hip().hipMemcpy(hostData2.data(), devicePtr2, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData2[i] == 0xBB);
    }
    
    REQUIRE(hip().hipFree(devicePtr1) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr2) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream1) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream2) == hipSuccess);
}

//=============================================================================
// Stream Query Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamQuery after sync returns success", "[sync][stream][query]") {
    hipStream_t stream = nullptr;
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Do some work
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0, size, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Query should return success
    REQUIRE(hip().hipStreamQuery(stream) == hipSuccess);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventQuery after sync returns success", "[sync][event][query]") {
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Do some work
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0, size, stream) == hipSuccess);
    
    // Record event
    REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    
    // Query should return success
    REQUIRE(hip().hipEventQuery(event) == hipSuccess);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Data Dependency Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Memcpy H2D then D2D then D2H chain", "[sync][memcpy]") {
    void* devicePtr1 = nullptr;
    void* devicePtr2 = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    
    std::vector<uint8_t> hostSrc(size);
    std::vector<uint8_t> hostDst(size, 0);
    
    // Initialize source
    for (size_t i = 0; i < size; ++i) {
        hostSrc[i] = static_cast<uint8_t>(i);
    }
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr1, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr2, size) == hipSuccess);
    
    // Chain: H2D -> D2D -> D2H
    void* hostPinned = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPinned, size, hipHostMallocDefault) == hipSuccess);
    memcpy(hostPinned, hostSrc.data(), size);
    
    REQUIRE(hip().hipMemcpyAsync(devicePtr1, hostPinned, size, hipMemcpyHostToDevice, stream) == hipSuccess);
    REQUIRE(hip().hipMemcpyAsync(devicePtr2, devicePtr1, size, hipMemcpyDeviceToDevice, stream) == hipSuccess);
    REQUIRE(hip().hipMemcpyAsync(hostPinned, devicePtr2, size, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    memcpy(hostDst.data(), hostPinned, size);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipHostFree(hostPinned) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr1) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr2) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multiple memsets then single memcpy", "[sync][memset][memcpy]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Multiple memsets - last one wins
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x11, size, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x22, size, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x33, size, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x44, size, stream) == hipSuccess);
    
    // Sync and verify
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x44);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Null Stream Synchronization Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Null stream sync with hipDeviceSynchronize", "[sync][stream]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Use null stream (default)
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x99, size, nullptr) == hipSuccess);
    
    // Device sync
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x99);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize on null stream", "[sync][stream]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Use null stream
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x77, size, nullptr) == hipSuccess);
    
    // Sync null stream
    REQUIRE(hip().hipStreamSynchronize(nullptr) == hipSuccess);
    
    // Verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x77);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}
