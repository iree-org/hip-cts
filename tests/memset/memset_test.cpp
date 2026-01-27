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
// hipMemset Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset basic", "[memset][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Memset to a value
    REQUIRE(hip().hipMemset(devicePtr, 0xAB, size) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset zero value", "[memset][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0xFF);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Fill with non-zero first
    REQUIRE(hip().hipMemset(devicePtr, 0xFF, size) == hipSuccess);
    
    // Then zero it
    REQUIRE(hip().hipMemset(devicePtr, 0, size) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset partial buffer", "[memset][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    constexpr size_t partialSize = 1024;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Zero entire buffer
    REQUIRE(hip().hipMemset(devicePtr, 0, size) == hipSuccess);
    
    // Memset only first part
    REQUIRE(hip().hipMemset(devicePtr, 0xCD, partialSize) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // First part should be 0xCD
    for (size_t i = 0; i < partialSize; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    // Rest should be 0
    for (size_t i = partialSize; i < size; ++i) {
        REQUIRE(hostData[i] == 0);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset null pointer fails", "[memset][sync][negative]") {
    REQUIRE(hip().hipMemset(nullptr, 0, 4096) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset different sizes", "[memset][sync]") {
    std::vector<size_t> sizes = {1, 64, 256, 1024, 4096, 65536};
    
    for (size_t size : sizes) {
        void* devicePtr = nullptr;
        std::vector<uint8_t> hostData(size, 0);
        
        REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
        
        uint8_t value = static_cast<uint8_t>(size & 0xFF);
        REQUIRE(hip().hipMemset(devicePtr, value, size) == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(hostData[i] == value);
        }
        
        REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    }
}

//=============================================================================
// hipMemsetAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync basic", "[memset][async]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Async memset
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xEF, size, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync on null stream", "[memset][async]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Async memset on null stream (default stream)
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x12, size, nullptr) == hipSuccess);
    
    // Synchronize device
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x12);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync multiple on same stream", "[memset][async]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Multiple async memsets - last one should win
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x11, size, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x22, size, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0x33, size, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify (should be 0x33)
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x33);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync null pointer fails", "[memset][async][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(nullptr, 0, 4096, stream) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipMemsetD8 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD8 basic", "[memset][d8]") {
    void* devicePtr = nullptr;
    constexpr size_t count = 4096;
    std::vector<uint8_t> hostData(count, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, count) == hipSuccess);
    
    // D8 memset
    REQUIRE(hip().hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                               0xAB, count) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, count, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// hipMemsetD16 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD16 basic", "[memset][d16]") {
    void* devicePtr = nullptr;
    constexpr size_t count = 2048; // Number of 16-bit elements
    constexpr size_t sizeBytes = count * sizeof(uint16_t);
    std::vector<uint16_t> hostData(count, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, sizeBytes) == hipSuccess);
    
    // D16 memset
    uint16_t value = 0xABCD;
    REQUIRE(hip().hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                value, count) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == value);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// hipMemsetD32 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32 basic", "[memset][d32]") {
    void* devicePtr = nullptr;
    constexpr size_t count = 1024; // Number of 32-bit elements
    constexpr size_t sizeBytes = count * sizeof(uint32_t);
    std::vector<uint32_t> hostData(count, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, sizeBytes) == hipSuccess);
    
    // D32 memset
    int value = 0xDEADBEEF;
    REQUIRE(hip().hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                value, count) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == static_cast<uint32_t>(value));
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// hipMemsetD8Async Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD8Async basic", "[memset][d8][async]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t count = 4096;
    std::vector<uint8_t> hostData(count, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, count) == hipSuccess);
    
    // D8 async memset
    REQUIRE(hip().hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                    0xCD, count, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, count, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipMemsetD16Async Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD16Async basic", "[memset][d16][async]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t count = 2048;
    constexpr size_t sizeBytes = count * sizeof(uint16_t);
    std::vector<uint16_t> hostData(count, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, sizeBytes) == hipSuccess);
    
    // D16 async memset
    uint16_t value = 0x1234;
    REQUIRE(hip().hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                     value, count, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == value);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipMemsetD32Async Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32Async basic", "[memset][d32][async]") {
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t count = 1024;
    constexpr size_t sizeBytes = count * sizeof(uint32_t);
    std::vector<uint32_t> hostData(count, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, sizeBytes) == hipSuccess);
    
    // D32 async memset
    int value = 0xCAFEBABE;
    REQUIRE(hip().hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                     value, count, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, sizeBytes, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        REQUIRE(hostData[i] == static_cast<uint32_t>(value));
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Multiple Buffers Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset multiple buffers", "[memset][sync]") {
    constexpr int numBuffers = 4;
    constexpr size_t size = 4096;
    
    std::vector<void*> devicePtrs(numBuffers, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numBuffers, std::vector<uint8_t>(size, 0));
    
    // Allocate buffers
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Memset each buffer with different value
    for (int i = 0; i < numBuffers; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 1);
        REQUIRE(hip().hipMemset(devicePtrs[i], value, size) == hipSuccess);
    }
    
    // Copy back and verify each
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipMemcpy(hostData[i].data(), devicePtrs[i], size, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        uint8_t expectedValue = static_cast<uint8_t>(i + 1);
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(hostData[i][j] == expectedValue);
        }
    }
    
    // Cleanup
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync multiple buffers on different streams", "[memset][async]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numStreams, std::vector<uint8_t>(size, 0));
    
    // Create streams and allocate buffers
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Async memset on each stream
    for (int i = 0; i < numStreams; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 10);
        REQUIRE(hip().hipMemsetAsync(devicePtrs[i], value, size, streams[i]) == hipSuccess);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Copy back and verify
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpy(hostData[i].data(), devicePtrs[i], size, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        uint8_t expectedValue = static_cast<uint8_t>(i + 10);
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
