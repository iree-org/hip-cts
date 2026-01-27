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
// hipMemcpy2D Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2D H2D and D2H basic", "[memory2d][memcpy2d]") {
    void* devicePtr = nullptr;
    size_t devicePitch = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &devicePitch, width, height) == hipSuccess);
    REQUIRE(devicePitch >= width);
    
    std::vector<uint8_t> hostSrc(width * height);
    std::vector<uint8_t> hostDst(width * height, 0);
    
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        hostSrc[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    REQUIRE(hip().hipMemcpy2D(devicePtr, devicePitch, hostSrc.data(), width,
                               width, height, hipMemcpyHostToDevice) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy2D(hostDst.data(), width, devicePtr, devicePitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2D D2D", "[memory2d][memcpy2d]") {
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    size_t pitchSrc = 0;
    size_t pitchDst = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipMallocPitch(&deviceSrc, &pitchSrc, width, height) == hipSuccess);
    REQUIRE(hip().hipMallocPitch(&deviceDst, &pitchDst, width, height) == hipSuccess);
    
    std::vector<uint8_t> hostSrc(width * height);
    std::vector<uint8_t> hostDst(width * height, 0);
    
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        hostSrc[i] = static_cast<uint8_t>((i * 7) & 0xFF);
    }
    
    REQUIRE(hip().hipMemcpy2D(deviceSrc, pitchSrc, hostSrc.data(), width,
                               width, height, hipMemcpyHostToDevice) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy2D(deviceDst, pitchDst, deviceSrc, pitchSrc,
                               width, height, hipMemcpyDeviceToDevice) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy2D(hostDst.data(), width, deviceDst, pitchDst,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipFree(deviceSrc) == hipSuccess);
    REQUIRE(hip().hipFree(deviceDst) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2D null dst fails", "[memory2d][memcpy2d][negative]") {
    std::vector<uint8_t> hostData(1024);
    REQUIRE(hip().hipMemcpy2D(nullptr, 256, hostData.data(), 256,
                               256, 4, hipMemcpyHostToDevice) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2D null src fails", "[memory2d][memcpy2d][negative]") {
    void* devicePtr = nullptr;
    size_t pitch = 0;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, 256, 4) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy2D(devicePtr, pitch, nullptr, 256,
                               256, 4, hipMemcpyHostToDevice) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2D various sizes", "[memory2d][memcpy2d]") {
    struct TestCase {
        size_t width;
        size_t height;
    };
    
    std::vector<TestCase> cases = {
        {64, 64},
        {128, 256},
        {512, 128},
        {100, 100},
        {1024, 16},
        {16, 1024},
    };
    
    for (const auto& tc : cases) {
        void* devicePtr = nullptr;
        size_t pitch = 0;
        
        REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, tc.width, tc.height) == hipSuccess);
        
        std::vector<uint8_t> hostSrc(tc.width * tc.height);
        std::vector<uint8_t> hostDst(tc.width * tc.height, 0);
        
        for (size_t i = 0; i < hostSrc.size(); ++i) {
            hostSrc[i] = static_cast<uint8_t>(i);
        }
        
        REQUIRE(hip().hipMemcpy2D(devicePtr, pitch, hostSrc.data(), tc.width,
                                   tc.width, tc.height, hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(hip().hipMemcpy2D(hostDst.data(), tc.width, devicePtr, pitch,
                                   tc.width, tc.height, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < hostSrc.size(); ++i) {
            REQUIRE(hostDst[i] == hostSrc[i]);
        }
        
        REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    }
}

//=============================================================================
// hipMemcpy2DAsync Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2DAsync basic", "[memory2d][memcpy2d][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t devicePitch = 0;
    hipStream_t stream = nullptr;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    constexpr size_t hostPitch = width;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMallocPitch(&devicePtr, &devicePitch, width, height) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, width * height, hipHostMallocDefault) == hipSuccess);
    
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < width * height; ++i) {
        hostBytes[i] = static_cast<uint8_t>(i);
    }
    
    REQUIRE(hip().hipMemcpy2DAsync(devicePtr, devicePitch, hostPtr, hostPitch,
                                    width, height, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    memset(hostPtr, 0, width * height);
    REQUIRE(hip().hipMemcpy2DAsync(hostPtr, hostPitch, devicePtr, devicePitch,
                                    width, height, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    for (size_t i = 0; i < width * height; ++i) {
        REQUIRE(hostBytes[i] == static_cast<uint8_t>(i));
    }
    
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy2DAsync on null stream", "[memory2d][memcpy2d][async]") {
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t devicePitch = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 64;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &devicePitch, width, height) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, width * height, hipHostMallocDefault) == hipSuccess);
    
    memset(hostPtr, 0xAB, width * height);
    
    REQUIRE(hip().hipMemcpy2DAsync(devicePtr, devicePitch, hostPtr, width,
                                    width, height, hipMemcpyHostToDevice, nullptr) == hipSuccess);
    
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<uint8_t> hostDst(width * height, 0);
    REQUIRE(hip().hipMemcpy2D(hostDst.data(), width, devicePtr, devicePitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < hostDst.size(); ++i) {
        REQUIRE(hostDst[i] == 0xAB);
    }
    
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// hipMemset2D Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset2D basic", "[memory2d][memset2d]") {
    void* devicePtr = nullptr;
    size_t pitch = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, width, height) == hipSuccess);
    
    REQUIRE(hip().hipMemset2D(devicePtr, pitch, 0xCD, width, height) == hipSuccess);
    
    std::vector<uint8_t> hostData(width * height, 0);
    REQUIRE(hip().hipMemcpy2D(hostData.data(), width, devicePtr, pitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < hostData.size(); ++i) {
        REQUIRE(hostData[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset2D null dst fails", "[memory2d][memset2d][negative]") {
    REQUIRE(hip().hipMemset2D(nullptr, 256, 0, 256, 4) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset2DAsync basic", "[memory2d][memset2d][async]") {
    void* devicePtr = nullptr;
    size_t pitch = 0;
    hipStream_t stream = nullptr;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, width, height) == hipSuccess);
    
    REQUIRE(hip().hipMemset2DAsync(devicePtr, pitch, 0xEF, width, height, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    std::vector<uint8_t> hostData(width * height, 0);
    REQUIRE(hip().hipMemcpy2D(hostData.data(), width, devicePtr, pitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < hostData.size(); ++i) {
        REQUIRE(hostData[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipMalloc3D Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc3D basic", "[memory2d][malloc3d]") {
    hipPitchedPtr pitchedPtr;
    memset(&pitchedPtr, 0, sizeof(pitchedPtr));
    
    hipExtent extent = {64, 64, 16};
    
    REQUIRE(hip().hipMalloc3D(&pitchedPtr, extent) == hipSuccess);
    REQUIRE(pitchedPtr.ptr != nullptr);
    REQUIRE(pitchedPtr.pitch >= 64);
    
    REQUIRE(hip().hipFree(pitchedPtr.ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc3D null pointer fails", "[memory2d][malloc3d][negative]") {
    hipExtent extent = {64, 64, 16};
    REQUIRE(hip().hipMalloc3D(nullptr, extent) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc3D various sizes", "[memory2d][malloc3d]") {
    struct TestCase {
        size_t width;
        size_t height;
        size_t depth;
    };
    
    std::vector<TestCase> cases = {
        {32, 32, 8},
        {64, 64, 16},
        {128, 128, 4},
        {256, 64, 8},
    };
    
    for (const auto& tc : cases) {
        hipPitchedPtr pitchedPtr;
        memset(&pitchedPtr, 0, sizeof(pitchedPtr));
        
        hipExtent extent = {tc.width, tc.height, tc.depth};
        
        REQUIRE(hip().hipMalloc3D(&pitchedPtr, extent) == hipSuccess);
        REQUIRE(pitchedPtr.ptr != nullptr);
        REQUIRE(pitchedPtr.pitch >= tc.width);
        
        REQUIRE(hip().hipFree(pitchedPtr.ptr) == hipSuccess);
    }
}

//=============================================================================
// Pitched Memory Roundtrip Test
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Pitched memory full roundtrip", "[memory2d][integration]") {
    void* devicePtr = nullptr;
    size_t pitch = 0;
    constexpr size_t width = 512;
    constexpr size_t height = 256;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, width, height) == hipSuccess);
    
    std::vector<uint8_t> hostSrc(width * height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            hostSrc[y * width + x] = static_cast<uint8_t>((x ^ y) & 0xFF);
        }
    }
    
    REQUIRE(hip().hipMemcpy2D(devicePtr, pitch, hostSrc.data(), width,
                               width, height, hipMemcpyHostToDevice) == hipSuccess);
    
    std::vector<uint8_t> hostDst(width * height, 0);
    REQUIRE(hip().hipMemcpy2D(hostDst.data(), width, devicePtr, pitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            REQUIRE(hostDst[idx] == static_cast<uint8_t>((x ^ y) & 0xFF));
        }
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}
