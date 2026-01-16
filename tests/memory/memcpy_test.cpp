// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for hipMemcpy APIs
// Migrated from TheRock: rocm-systems/projects/hip-tests/catch/unit/memory/hipMemcpy*.cc

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>

//=============================================================================
// Helper constants
//=============================================================================
constexpr size_t kPageSize = 4096;

//=============================================================================
// hipMemcpy Basic Tests - Host to Device
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy host to device basic", "[memory][memcpy]") {
    auto size = GENERATE(
        size_t(kPageSize / 2),
        size_t(kPageSize),
        size_t(kPageSize * 2)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        // Create source data on host
        std::vector<unsigned char> h_src(size);
        for (size_t i = 0; i < size; ++i) {
            h_src[i] = static_cast<unsigned char>(i % 256);
        }
        
        // Allocate device memory
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        // Copy to device
        REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        
        // Copy back and verify
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == h_src[i]);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy device to host basic", "[memory][memcpy]") {
    auto size = GENERATE(
        size_t(kPageSize / 2),
        size_t(kPageSize),
        size_t(kPageSize * 2)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        // Create source data on host and transfer to device via memset
        void* d_ptr = nullptr;
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        // Memset device memory to a pattern
        REQUIRE(hip().hipMemset(d_ptr, 0x42, size) == hipSuccess);
        
        // Copy back to host
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        // Verify
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == 0x42);
        }
        
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy device to device basic", "[memory][memcpy]") {
    auto size = GENERATE(
        size_t(kPageSize / 2),
        size_t(kPageSize),
        size_t(kPageSize * 2)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        // Create source data on host
        std::vector<unsigned char> h_src(size);
        for (size_t i = 0; i < size; ++i) {
            h_src[i] = static_cast<unsigned char>(i % 256);
        }
        
        // Allocate source and destination device memory
        void* d_src = nullptr;
        void* d_dst = nullptr;
        REQUIRE(hip().hipMalloc(&d_src, size) == hipSuccess);
        REQUIRE(hip().hipMalloc(&d_dst, size) == hipSuccess);
        
        // Initialize source device memory
        REQUIRE(hip().hipMemcpy(d_src, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        
        // Device to device copy
        REQUIRE(hip().hipMemcpy(d_dst, d_src, size, hipMemcpyDeviceToDevice) == hipSuccess);
        
        // Copy result back to host and verify
        std::vector<unsigned char> h_dst(size, 0);
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_dst, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == h_src[i]);
        }
        
        REQUIRE(hip().hipFree(d_src) == hipSuccess);
        REQUIRE(hip().hipFree(d_dst) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy host to host basic", "[memory][memcpy]") {
    auto size = GENERATE(
        size_t(kPageSize / 2),
        size_t(kPageSize),
        size_t(kPageSize * 2)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        std::vector<unsigned char> h_src(size);
        for (size_t i = 0; i < size; ++i) {
            h_src[i] = static_cast<unsigned char>(i % 256);
        }
        
        std::vector<unsigned char> h_dst(size, 0);
        
        REQUIRE(hip().hipMemcpy(h_dst.data(), h_src.data(), size, hipMemcpyHostToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == h_src[i]);
        }
    }
}

//=============================================================================
// hipMemcpy with hipMemcpyDefault
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy with hipMemcpyDefault host to device", "[memory][memcpy]") {
    constexpr size_t size = 1024;
    
    std::vector<unsigned char> h_src(size, 0x55);
    void* d_ptr = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyDefault) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDefault) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x55);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemcpy Zero Size
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy zero size is valid", "[memory][memcpy]") {
    void* d_ptr = nullptr;
    unsigned char h_src = 0x42;
    unsigned char h_dst = 0;
    
    REQUIRE(hip().hipMalloc(&d_ptr, 64) == hipSuccess);
    
    // Zero-size memcpy should succeed
    REQUIRE(hip().hipMemcpy(d_ptr, &h_src, 0, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hip().hipMemcpy(&h_dst, d_ptr, 0, hipMemcpyDeviceToHost) == hipSuccess);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemcpy Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy null dst returns error", "[memory][memcpy][negative]") {
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, 64) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(nullptr, d_ptr, 64, hipMemcpyDeviceToHost) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy null src returns error", "[memory][memcpy][negative]") {
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, 64) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(d_ptr, nullptr, 64, hipMemcpyHostToDevice) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemcpyHtoD/DtoH/DtoD Tests (Driver API style)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyHtoD basic", "[memory][memcpy]") {
    constexpr size_t size = 1024;
    
    std::vector<unsigned char> h_src(size, 0x77);
    void* d_ptr = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d_ptr), h_src.data(), size) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpyDtoH(h_dst.data(), reinterpret_cast<hipDeviceptr_t>(d_ptr), size) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x77);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoH basic", "[memory][memcpy]") {
    constexpr size_t size = 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_ptr, 0x88, size) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpyDtoH(h_dst.data(), reinterpret_cast<hipDeviceptr_t>(d_ptr), size) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x88);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoD basic", "[memory][memcpy]") {
    constexpr size_t size = 1024;
    
    std::vector<unsigned char> h_src(size, 0x99);
    void* d_src = nullptr;
    void* d_dst = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_src, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&d_dst, size) == hipSuccess);
    
    // Initialize source
    REQUIRE(hip().hipMemcpy(d_src, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // D2D copy
    REQUIRE(hip().hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d_dst), 
                                 reinterpret_cast<hipDeviceptr_t>(d_src), size) == hipSuccess);
    
    // Verify
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_dst, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x99);
    }
    
    REQUIRE(hip().hipFree(d_src) == hipSuccess);
    REQUIRE(hip().hipFree(d_dst) == hipSuccess);
}

//=============================================================================
// hipMemcpyAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync host to device", "[memory][memcpy][async]") {
    constexpr size_t size = 4096;
    
    std::vector<unsigned char> h_src(size, 0xAA);
    std::vector<unsigned char> h_dst(size, 0);
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Async copy to device
    REQUIRE(hip().hipMemcpyAsync(d_ptr, h_src.data(), size, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Async copy back
    REQUIRE(hip().hipMemcpyAsync(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0xAA);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync null stream uses default stream", "[memory][memcpy][async]") {
    constexpr size_t size = 1024;
    
    std::vector<unsigned char> h_src(size, 0xBB);
    std::vector<unsigned char> h_dst(size, 0);
    void* d_ptr = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Async copy with null stream
    REQUIRE(hip().hipMemcpyAsync(d_ptr, h_src.data(), size, hipMemcpyHostToDevice, nullptr) == hipSuccess);
    REQUIRE(hip().hipMemcpyAsync(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost, nullptr) == hipSuccess);
    
    // Synchronize default stream
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0xBB);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Large Transfer Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy large transfer", "[memory][memcpy]") {
    // 16 MB transfer
    constexpr size_t size = 16 * 1024 * 1024;
    
    std::vector<unsigned char> h_src(size);
    for (size_t i = 0; i < size; ++i) {
        h_src[i] = static_cast<unsigned char>(i % 256);
    }
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    std::vector<unsigned char> h_dst(size, 0);
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Sample verification (checking every 1000th byte)
    for (size_t i = 0; i < size; i += 1000) {
        REQUIRE(h_dst[i] == h_src[i]);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// hipMemset Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset basic", "[memory][memset]") {
    auto size = GENERATE(
        size_t(64),
        size_t(1024),
        size_t(4096)
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

//=============================================================================
// hipMemset Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset null pointer returns error", "[memory][memset][negative]") {
    REQUIRE(hip().hipMemset(nullptr, 0, 64) == hipErrorInvalidValue);
}

//=============================================================================
// Multiple Stream Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync multiple streams", "[memory][memcpy][async]") {
    constexpr size_t size = 1024;
    constexpr int numStreams = 4;
    
    std::vector<hipStream_t> streams(numStreams);
    std::vector<void*> d_ptrs(numStreams);
    std::vector<std::vector<unsigned char>> h_srcs(numStreams);
    std::vector<std::vector<unsigned char>> h_dsts(numStreams);
    
    // Create streams and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&d_ptrs[i], size) == hipSuccess);
        h_srcs[i].resize(size, static_cast<unsigned char>(i * 10));
        h_dsts[i].resize(size, 0);
    }
    
    // Issue async copies on each stream
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpyAsync(d_ptrs[i], h_srcs[i].data(), size, 
                                      hipMemcpyHostToDevice, streams[i]) == hipSuccess);
    }
    
    // Issue async copies back
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpyAsync(h_dsts[i].data(), d_ptrs[i], size,
                                      hipMemcpyDeviceToHost, streams[i]) == hipSuccess);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Verify
    for (int i = 0; i < numStreams; ++i) {
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(h_dsts[i][j] == h_srcs[i][j]);
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
        REQUIRE(hip().hipFree(d_ptrs[i]) == hipSuccess);
    }
}
