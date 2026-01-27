// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include <numeric>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// hipMemcpy Host to Device Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy H2D basic", "[memcpy][h2d][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostSrc(size);
    std::vector<uint8_t> hostDst(size, 0);
    
    // Initialize source with pattern
    std::iota(hostSrc.begin(), hostSrc.end(), 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Copy to device
    REQUIRE(hip().hipMemcpy(devicePtr, hostSrc.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy back
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy D2H basic", "[memcpy][d2h][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Fill device memory with memset
    REQUIRE(hip().hipMemset(devicePtr, 0xAB, size) == hipSuccess);
    
    // Copy to host
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy D2D basic", "[memcpy][d2d][sync]") {
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&deviceSrc, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceDst, size) == hipSuccess);
    
    // Fill source
    REQUIRE(hip().hipMemset(deviceSrc, 0xCD, size) == hipSuccess);
    
    // Zero destination
    REQUIRE(hip().hipMemset(deviceDst, 0, size) == hipSuccess);
    
    // Device to device copy
    REQUIRE(hip().hipMemcpy(deviceDst, deviceSrc, size, hipMemcpyDeviceToDevice) == hipSuccess);
    
    // Copy to host and verify
    REQUIRE(hip().hipMemcpy(hostData.data(), deviceDst, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(deviceSrc) == hipSuccess);
    REQUIRE(hip().hipFree(deviceDst) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy H2H basic", "[memcpy][h2h][sync]") {
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostSrc(size);
    std::vector<uint8_t> hostDst(size, 0);
    
    // Initialize source
    std::iota(hostSrc.begin(), hostSrc.end(), 0);
    
    // Host to host copy
    REQUIRE(hip().hipMemcpy(hostDst.data(), hostSrc.data(), size, hipMemcpyHostToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy null dst fails", "[memcpy][sync][negative]") {
    std::vector<uint8_t> hostData(4096);
    REQUIRE(hip().hipMemcpy(nullptr, hostData.data(), 4096, hipMemcpyHostToDevice) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy null src fails", "[memcpy][sync][negative]") {
    void* devicePtr = nullptr;
    REQUIRE(hip().hipMalloc(&devicePtr, 4096) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(devicePtr, nullptr, 4096, hipMemcpyHostToDevice) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy different sizes", "[memcpy][sync]") {
    std::vector<size_t> sizes = {1, 64, 256, 1024, 4096, 65536};
    
    for (size_t size : sizes) {
        void* devicePtr = nullptr;
        std::vector<uint8_t> hostSrc(size);
        std::vector<uint8_t> hostDst(size, 0);
        
        std::iota(hostSrc.begin(), hostSrc.end(), static_cast<uint8_t>(size & 0xFF));
        
        REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
        REQUIRE(hip().hipMemcpy(devicePtr, hostSrc.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(hostDst[i] == hostSrc[i]);
        }
        
        REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    }
}

//=============================================================================
// hipMemcpyAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync H2D basic", "[memcpy][h2d][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostDst(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Initialize pinned host memory
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < size; ++i) {
        hostBytes[i] = static_cast<uint8_t>(i);
    }
    
    // Async H2D
    REQUIRE(hip().hipMemcpyAsync(devicePtr, hostPtr, size, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == static_cast<uint8_t>(i));
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync D2H basic", "[memcpy][d2h][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Fill device
    REQUIRE(hip().hipMemset(devicePtr, 0xEF, size) == hipSuccess);
    
    // Async D2H
    REQUIRE(hip().hipMemcpyAsync(hostPtr, devicePtr, size, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostBytes[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync D2D basic", "[memcpy][d2d][async]") {
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceSrc, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceDst, size) == hipSuccess);
    
    // Fill source
    REQUIRE(hip().hipMemset(deviceSrc, 0x12, size) == hipSuccess);
    
    // Async D2D
    REQUIRE(hip().hipMemcpyAsync(deviceDst, deviceSrc, size, hipMemcpyDeviceToDevice, stream) == hipSuccess);
    
    // Sync
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    REQUIRE(hip().hipMemcpy(hostData.data(), deviceDst, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x12);
    }
    
    REQUIRE(hip().hipFree(deviceSrc) == hipSuccess);
    REQUIRE(hip().hipFree(deviceDst) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync on null stream", "[memcpy][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostDst(size, 0);
    
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Initialize
    memset(hostPtr, 0x34, size);
    
    // Async on default stream
    REQUIRE(hip().hipMemcpyAsync(devicePtr, hostPtr, size, hipMemcpyHostToDevice, nullptr) == hipSuccess);
    
    // Sync device
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == 0x34);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

//=============================================================================
// hipMemcpyHtoD / hipMemcpyDtoH / hipMemcpyDtoD Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyHtoD basic", "[memcpy][htod][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostSrc(size);
    std::vector<uint8_t> hostDst(size, 0);
    
    std::iota(hostSrc.begin(), hostSrc.end(), 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(devicePtr), 
                                 hostSrc.data(), size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoH basic", "[memcpy][dtoh][sync]") {
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    REQUIRE(hip().hipMemset(devicePtr, 0x56, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyDtoH(hostData.data(), 
                                 reinterpret_cast<hipDeviceptr_t>(devicePtr), size) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x56);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoD basic", "[memcpy][dtod][sync]") {
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&deviceSrc, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceDst, size) == hipSuccess);
    
    REQUIRE(hip().hipMemset(deviceSrc, 0x78, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(deviceDst),
                                 reinterpret_cast<hipDeviceptr_t>(deviceSrc), size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(hostData.data(), deviceDst, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0x78);
    }
    
    REQUIRE(hip().hipFree(deviceSrc) == hipSuccess);
    REQUIRE(hip().hipFree(deviceDst) == hipSuccess);
}

//=============================================================================
// hipMemcpyHtoDAsync / hipMemcpyDtoHAsync / hipMemcpyDtoDAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyHtoDAsync basic", "[memcpy][htod][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostDst(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    memset(hostPtr, 0x9A, size);
    
    REQUIRE(hip().hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(devicePtr),
                                      hostPtr, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == 0x9A);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoHAsync basic", "[memcpy][dtoh][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    REQUIRE(hip().hipMemset(devicePtr, 0xBC, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyDtoHAsync(hostPtr, reinterpret_cast<hipDeviceptr_t>(devicePtr),
                                      size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostBytes[i] == 0xBC);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyDtoDAsync basic", "[memcpy][dtod][async]") {
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceSrc, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&deviceDst, size) == hipSuccess);
    
    REQUIRE(hip().hipMemset(deviceSrc, 0xDE, size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(deviceDst),
                                      reinterpret_cast<hipDeviceptr_t>(deviceSrc),
                                      size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(hostData.data(), deviceDst, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xDE);
    }
    
    REQUIRE(hip().hipFree(deviceSrc) == hipSuccess);
    REQUIRE(hip().hipFree(deviceDst) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipMemcpyWithStream Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyWithStream H2D", "[memcpy][withstream]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostDst(size, 0);
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    memset(hostPtr, 0xF0, size);
    
    REQUIRE(hip().hipMemcpyWithStream(devicePtr, hostPtr, size, hipMemcpyHostToDevice, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == 0xF0);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// Multiple Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy roundtrip multiple buffers", "[memcpy][sync]") {
    constexpr int numBuffers = 4;
    constexpr size_t size = 4096;
    
    std::vector<void*> devicePtrs(numBuffers, nullptr);
    std::vector<std::vector<uint8_t>> srcData(numBuffers);
    std::vector<std::vector<uint8_t>> dstData(numBuffers);
    
    // Allocate and init
    for (int i = 0; i < numBuffers; ++i) {
        srcData[i].resize(size);
        dstData[i].resize(size, 0);
        std::fill(srcData[i].begin(), srcData[i].end(), static_cast<uint8_t>(i + 1));
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Copy H2D
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipMemcpy(devicePtrs[i], srcData[i].data(), size, hipMemcpyHostToDevice) == hipSuccess);
    }
    
    // Copy D2H
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipMemcpy(dstData[i].data(), devicePtrs[i], size, hipMemcpyDeviceToHost) == hipSuccess);
    }
    
    // Verify
    for (int i = 0; i < numBuffers; ++i) {
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(dstData[i][j] == srcData[i][j]);
        }
    }
    
    // Cleanup
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync multiple streams concurrent", "[memcpy][async]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<void*> hostPtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> dstData(numStreams);
    
    // Create resources
    for (int i = 0; i < numStreams; ++i) {
        dstData[i].resize(size, 0);
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipHostMalloc(&hostPtrs[i], size, hipHostMallocDefault) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
        memset(hostPtrs[i], i + 10, size);
    }
    
    // Issue concurrent H2D copies
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpyAsync(devicePtrs[i], hostPtrs[i], size, 
                                      hipMemcpyHostToDevice, streams[i]) == hipSuccess);
    }
    
    // Sync all
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }
    
    // Verify
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpy(dstData[i].data(), devicePtrs[i], size, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(dstData[i][j] == static_cast<uint8_t>(i + 10));
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
        REQUIRE(hip().hipHostFree(hostPtrs[i]) == hipSuccess);
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy chain D2D", "[memcpy][d2d][sync]") {
    constexpr int numBuffers = 4;
    constexpr size_t size = 4096;
    
    std::vector<void*> devicePtrs(numBuffers, nullptr);
    std::vector<uint8_t> hostSrc(size);
    std::vector<uint8_t> hostDst(size, 0);
    
    std::iota(hostSrc.begin(), hostSrc.end(), 0);
    
    // Allocate
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Copy to first buffer
    REQUIRE(hip().hipMemcpy(devicePtrs[0], hostSrc.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Chain D2D copies
    for (int i = 1; i < numBuffers; ++i) {
        REQUIRE(hip().hipMemcpy(devicePtrs[i], devicePtrs[i-1], size, hipMemcpyDeviceToDevice) == hipSuccess);
    }
    
    // Copy back from last
    REQUIRE(hip().hipMemcpy(hostDst.data(), devicePtrs[numBuffers-1], size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    // Cleanup
    for (int i = 0; i < numBuffers; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
    }
}
