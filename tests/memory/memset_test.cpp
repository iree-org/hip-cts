// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for hipMemset APIs
// Migrated from TheRock: rocm-systems/projects/hip-tests/catch/unit/memory/hipMemset*.cc

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include <vector>
#include <cstring>
#include <cstdint>
#include <array>

//=============================================================================
// Helper constants
//=============================================================================
static constexpr size_t kPageSize = 4096;

//=============================================================================
// hipMemset Basic Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset basic", "[memory][memset]") {
    auto size = GENERATE(
        size_t(64),
        size_t(1024),
        size_t(kPageSize),
        size_t(kPageSize * 4)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        REQUIRE(hip().hipMemset(d_ptr, 0xCD, size) == hipSuccess);
        
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == 0xCD);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset zero pattern", "[memory][memset]") {
    constexpr size_t size = 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // First fill with non-zero
    REQUIRE(hip().hipMemset(d_ptr, 0xFF, size) == hipSuccess);
    
    // Then zero it out
    REQUIRE(hip().hipMemset(d_ptr, 0, size) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0xFF);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset various patterns", "[memory][memset]") {
    constexpr size_t size = 1024;
    
    auto pattern = GENERATE(
        0x00,
        0x42,
        0x5A,
        0xA6,
        0xFF
    );
    
    DYNAMIC_SECTION("Pattern: 0x" << std::hex << pattern) {
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        REQUIRE(hip().hipMemset(d_ptr, pattern, size) == hipSuccess);
        
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == static_cast<unsigned char>(pattern));
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset small sizes", "[memory][memset]") {
    auto size = GENERATE(
        size_t(1),
        size_t(2),
        size_t(3),
        size_t(7),
        size_t(15),
        size_t(16),
        size_t(17)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        REQUIRE(hip().hipMemset(d_ptr, 0x24, size) == hipSuccess);
        
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == 0x24);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset large size", "[memory][memset]") {
    // 16 MB
    constexpr size_t size = 16 * 1024 * 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    REQUIRE(hip().hipMemset(d_ptr, 0x77, size) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Sample verification
    for (size_t i = 0; i < size; i += 1000) {
        REQUIRE(h_dst[i] == 0x77);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset zero size is valid", "[memory][memset]") {
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, 64) == hipSuccess);
    
    // Zero-size memset should succeed
    REQUIRE(hip().hipMemset(d_ptr, 0xAB, 0) == hipSuccess);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemsetAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync basic", "[memory][memset][async]") {
    constexpr size_t size = 4096;
    
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0xEE, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0xEE);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync null stream", "[memory][memset][async]") {
    constexpr size_t size = 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0x33, size, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x33);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync multiple streams", "[memory][memset][async]") {
    constexpr size_t size = 1024;
    constexpr int numStreams = 4;
    
    std::vector<hipStream_t> streams(numStreams);
    std::vector<void*> d_ptrs(numStreams, nullptr);
    
    // Create streams and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&d_ptrs[i], size) == hipSuccess);
    }
    
    // Issue async memsets on each stream with different patterns
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemsetAsync(d_ptrs[i], static_cast<unsigned char>(i * 10), 
                                      size, streams[i]) == hipSuccess);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Verify
    for (int i = 0; i < numStreams; ++i) {
        std::vector<unsigned char> h_dst(size, 0xFF);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptrs[i], size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(h_dst[j] == static_cast<unsigned char>(i * 10));
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
        REQUIRE(hip().hipFree(d_ptrs[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync followed by memcpy", "[memory][memset][async]") {
    constexpr size_t size = 4096;
    
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Memset async
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0xAB, size, stream) == hipSuccess);
    
    // Synchronize before copying back
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back synchronously (memcpyAsync to pageable memory has limitations)
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, 
                             hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0xAB);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemsetD8 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD8 basic", "[memory][memset][d8]") {
    auto numElements = GENERATE(
        size_t(64),
        size_t(1024),
        size_t(kPageSize)
    );
    
    DYNAMIC_SECTION("Elements: " << numElements) {
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, numElements) == hipSuccess);
        
        REQUIRE(hip().hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                   0xDE, numElements) == hipSuccess);
        
        std::vector<unsigned char> h_dst(numElements, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, numElements, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < numElements; ++i) {
            REQUIRE(h_dst[i] == 0xDE);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD8Async basic", "[memory][memset][d8][async]") {
    constexpr size_t numElements = 1024;
    
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, numElements) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                    0xCA, numElements, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    std::vector<unsigned char> h_dst(numElements, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, numElements, 
                             hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < numElements; ++i) {
        REQUIRE(h_dst[i] == 0xCA);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemsetD16 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD16 basic", "[memory][memset][d16]") {
    auto numElements = GENERATE(
        size_t(64),
        size_t(1024),
        size_t(kPageSize / 2)
    );
    
    DYNAMIC_SECTION("Elements: " << numElements) {
        void* d_ptr = nullptr;
        size_t sizeBytes = numElements * sizeof(int16_t);
        REQUIRE(hip().hipMalloc(&d_ptr, sizeBytes) == hipSuccess);
        
        int16_t pattern = 0xDEAD;
        REQUIRE(hip().hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                    pattern, numElements) == hipSuccess);
        
        std::vector<int16_t> h_dst(numElements, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, sizeBytes, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < numElements; ++i) {
            REQUIRE(h_dst[i] == pattern);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD16Async basic", "[memory][memset][d16][async]") {
    constexpr size_t numElements = 1024;
    
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    size_t sizeBytes = numElements * sizeof(int16_t);
    
    REQUIRE(hip().hipMalloc(&d_ptr, sizeBytes) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    int16_t pattern = static_cast<int16_t>(0xCAFE);
    REQUIRE(hip().hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                     pattern, numElements, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    std::vector<int16_t> h_dst(numElements, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, sizeBytes, 
                             hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < numElements; ++i) {
        REQUIRE(h_dst[i] == pattern);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemsetD32 Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32 basic", "[memory][memset][d32]") {
    auto numElements = GENERATE(
        size_t(64),
        size_t(1024),
        size_t(kPageSize / 4)
    );
    
    DYNAMIC_SECTION("Elements: " << numElements) {
        void* d_ptr = nullptr;
        size_t sizeBytes = numElements * sizeof(int32_t);
        REQUIRE(hip().hipMalloc(&d_ptr, sizeBytes) == hipSuccess);
        
        int32_t pattern = static_cast<int32_t>(0xDEADBEEF);
        REQUIRE(hip().hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                    pattern, numElements) == hipSuccess);
        
        std::vector<int32_t> h_dst(numElements, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, sizeBytes, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < numElements; ++i) {
            REQUIRE(h_dst[i] == pattern);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32Async basic", "[memory][memset][d32][async]") {
    constexpr size_t numElements = 1024;
    
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    size_t sizeBytes = numElements * sizeof(int32_t);
    
    REQUIRE(hip().hipMalloc(&d_ptr, sizeBytes) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    int32_t pattern = static_cast<int32_t>(0xCAFEBABE);
    REQUIRE(hip().hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                     pattern, numElements, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    std::vector<int32_t> h_dst(numElements, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, sizeBytes, 
                             hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < numElements; ++i) {
        REQUIRE(h_dst[i] == pattern);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32 with kernel pattern", "[memory][memset][d32]") {
    // Test a pattern that represents a float value (0x3fe00000 = 1.75f)
    constexpr size_t numElements = 1024;
    
    void* d_ptr = nullptr;
    size_t sizeBytes = numElements * sizeof(float);
    
    REQUIRE(hip().hipMalloc(&d_ptr, sizeBytes) == hipSuccess);
    
    // 0x3fe00000 is the IEEE 754 representation of 1.75f
    REQUIRE(hip().hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_ptr), 
                                0x3fe00000, numElements) == hipSuccess);
    
    std::vector<float> h_dst(numElements, 0.0f);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, sizeBytes, 
                             hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < numElements; ++i) {
        REQUIRE(h_dst[i] == 1.75f);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset null pointer returns error", "[memory][memset][negative]") {
    REQUIRE(hip().hipMemset(nullptr, 0, 64) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetAsync null pointer returns error", "[memory][memset][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipMemsetAsync(nullptr, 0, 64, stream) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD8 null pointer returns error", "[memory][memset][d8][negative]") {
    REQUIRE(hip().hipMemsetD8(0, 0, 64) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD16 null pointer returns error", "[memory][memset][d16][negative]") {
    REQUIRE(hip().hipMemsetD16(0, 0, 64) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32 null pointer returns error", "[memory][memset][d32][negative]") {
    REQUIRE(hip().hipMemsetD32(0, 0, 64) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemsetD32 zero size is valid", "[memory][memset][d32]") {
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, 64) == hipSuccess);
    
    REQUIRE(hip().hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d_ptr), 0xDEADBEEF, 0) == hipSuccess);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Sequential Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset multiple sequential operations", "[memory][memset]") {
    constexpr size_t size = 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Multiple memsets, each should overwrite the previous
    REQUIRE(hip().hipMemset(d_ptr, 0x11, size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_ptr, 0x22, size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_ptr, 0x33, size) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Should have the last pattern
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x33);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset partial buffer", "[memory][memset]") {
    constexpr size_t totalSize = 1024;
    constexpr size_t partialSize = 512;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, totalSize) == hipSuccess);
    
    // Fill entire buffer with one pattern
    REQUIRE(hip().hipMemset(d_ptr, 0xAA, totalSize) == hipSuccess);
    
    // Fill first half with different pattern
    REQUIRE(hip().hipMemset(d_ptr, 0xBB, partialSize) == hipSuccess);
    
    std::vector<unsigned char> h_dst(totalSize, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, totalSize, hipMemcpyDeviceToHost) == hipSuccess);
    
    // First half should be 0xBB
    for (size_t i = 0; i < partialSize; ++i) {
        REQUIRE(h_dst[i] == 0xBB);
    }
    
    // Second half should still be 0xAA
    for (size_t i = partialSize; i < totalSize; ++i) {
        REQUIRE(h_dst[i] == 0xAA);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset with offset", "[memory][memset]") {
    constexpr size_t totalSize = 1024;
    constexpr size_t offset = 256;
    constexpr size_t fillSize = 512;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, totalSize) == hipSuccess);
    
    // Fill entire buffer with one pattern
    REQUIRE(hip().hipMemset(d_ptr, 0x00, totalSize) == hipSuccess);
    
    // Fill middle section with different pattern
    void* offset_ptr = static_cast<char*>(d_ptr) + offset;
    REQUIRE(hip().hipMemset(offset_ptr, 0xFF, fillSize) == hipSuccess);
    
    std::vector<unsigned char> h_dst(totalSize, 0xAB);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, totalSize, hipMemcpyDeviceToHost) == hipSuccess);
    
    // First part should be 0x00
    for (size_t i = 0; i < offset; ++i) {
        REQUIRE(h_dst[i] == 0x00);
    }
    
    // Middle part should be 0xFF
    for (size_t i = offset; i < offset + fillSize; ++i) {
        REQUIRE(h_dst[i] == 0xFF);
    }
    
    // Last part should be 0x00
    for (size_t i = offset + fillSize; i < totalSize; ++i) {
        REQUIRE(h_dst[i] == 0x00);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Mixed Async Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset and hipMemsetD32 async interleaved", "[memory][memset][async]") {
    constexpr size_t size1 = 1024;
    constexpr size_t numElements = 256;
    
    void* d_ptr1 = nullptr;
    void* d_ptr2 = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr1, size1) == hipSuccess);
    REQUIRE(hip().hipMalloc(&d_ptr2, numElements * sizeof(int32_t)) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Interleave byte memset and 32-bit memset
    REQUIRE(hip().hipMemsetAsync(d_ptr1, 0, size1, stream) == hipSuccess);
    REQUIRE(hip().hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d_ptr2), 
                                     0x3fe00000, numElements, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify d_ptr1 (bytes)
    std::vector<unsigned char> h_dst1(size1);
    REQUIRE(hip().hipMemcpy(h_dst1.data(), d_ptr1, size1, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size1; ++i) {
        REQUIRE(h_dst1[i] == 0);
    }
    
    // Verify d_ptr2 (floats as 1.75f)
    std::vector<float> h_dst2(numElements);
    REQUIRE(hip().hipMemcpy(h_dst2.data(), d_ptr2, numElements * sizeof(float), 
                             hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < numElements; ++i) {
        REQUIRE(h_dst2[i] == 1.75f);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr1) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr2) == hipSuccess);
}
