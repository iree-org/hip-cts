// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HIP device configuration and limits

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Device Limit Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetLimit stack size", "[device][limit]") {
    size_t value = 0;
    hipError_t err = hip().hipDeviceGetLimit(&value, hipLimitStackSize);
    // May not be supported on all devices
    REQUIRE((err == hipSuccess || err == hipErrorUnsupportedLimit));
    if (err == hipSuccess) {
        INFO("Stack size limit: " << value);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetLimit malloc heap size", "[device][limit]") {
    size_t value = 0;
    hipError_t err = hip().hipDeviceGetLimit(&value, hipLimitMallocHeapSize);
    // May not be supported on all devices
    REQUIRE((err == hipSuccess || err == hipErrorUnsupportedLimit));
    if (err == hipSuccess) {
        INFO("Malloc heap size limit: " << value);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetLimit null pointer fails", "[device][limit][error]") {
    hipError_t err = hip().hipDeviceGetLimit(nullptr, hipLimitStackSize);
    REQUIRE(err == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetLimit stack size", "[device][limit]") {
    // Try to set stack size - this may not be supported
    hipError_t err = hip().hipDeviceSetLimit(hipLimitStackSize, 8192);
    REQUIRE((err == hipSuccess || err == hipErrorUnsupportedLimit));
}

//=============================================================================
// Device Cache Config Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetCacheConfig", "[device][cache]") {
    hipFuncCache_t config;
    REQUIRE(hip().hipDeviceGetCacheConfig(&config) == hipSuccess);
    INFO("Current cache config: " << static_cast<int>(config));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetCacheConfig prefer none", "[device][cache]") {
    hipError_t err = hip().hipDeviceSetCacheConfig(hipFuncCachePreferNone);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetCacheConfig prefer shared", "[device][cache]") {
    hipError_t err = hip().hipDeviceSetCacheConfig(hipFuncCachePreferShared);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetCacheConfig prefer L1", "[device][cache]") {
    hipError_t err = hip().hipDeviceSetCacheConfig(hipFuncCachePreferL1);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetCacheConfig prefer equal", "[device][cache]") {
    hipError_t err = hip().hipDeviceSetCacheConfig(hipFuncCachePreferEqual);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

//=============================================================================
// Device Shared Memory Config Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetSharedMemConfig", "[device][shared]") {
    hipSharedMemConfig config;
    REQUIRE(hip().hipDeviceGetSharedMemConfig(&config) == hipSuccess);
    INFO("Current shared mem config: " << static_cast<int>(config));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetSharedMemConfig default", "[device][shared]") {
    hipError_t err = hip().hipDeviceSetSharedMemConfig(hipSharedMemBankSizeDefault);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetSharedMemConfig four byte", "[device][shared]") {
    hipError_t err = hip().hipDeviceSetSharedMemConfig(hipSharedMemBankSizeFourByte);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSetSharedMemConfig eight byte", "[device][shared]") {
    hipError_t err = hip().hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);
    REQUIRE((err == hipSuccess || err == hipErrorNotSupported));
}

//=============================================================================
// Peer Access Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer single device", "[device][peer]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);

    if (deviceCount >= 2) {
        int canAccess = -1;
        REQUIRE(hip().hipDeviceCanAccessPeer(&canAccess, 0, 1) == hipSuccess);
        INFO("Device 0 can access device 1: " << canAccess);
        // Just verify the call works, value may be 0 or 1
        REQUIRE((canAccess == 0 || canAccess == 1));
    } else {
        SKIP("Need at least 2 devices for peer access tests");
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer same device", "[device][peer]") {
    int device = 0;
    REQUIRE(hip().hipGetDevice(&device) == hipSuccess);

    int canAccess = -1;
    // Self-access query
    hipError_t err = hip().hipDeviceCanAccessPeer(&canAccess, device, device);
    // Querying self may return error or success depending on implementation
    if (err == hipSuccess) {
        // Self-access is typically not allowed/meaningful
        INFO("Self-access result: " << canAccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceCanAccessPeer null pointer fails", "[device][peer][error]") {
    hipError_t err = hip().hipDeviceCanAccessPeer(nullptr, 0, 0);
    REQUIRE(err == hipErrorInvalidValue);
}

//=============================================================================
// Device PCI Bus ID Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetPCIBusId", "[device][pci]") {
    int device = 0;
    REQUIRE(hip().hipGetDevice(&device) == hipSuccess);

    char pciBusId[64] = {0};
    REQUIRE(hip().hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device) == hipSuccess);

    INFO("PCI Bus ID: " << pciBusId);
    REQUIRE(strlen(pciBusId) > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetPCIBusId null pointer fails", "[device][pci][error]") {
    hipError_t err = hip().hipDeviceGetPCIBusId(nullptr, 64, 0);
    REQUIRE(err == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetPCIBusId zero length fails", "[device][pci][error]") {
    char pciBusId[64] = {0};
    hipError_t err = hip().hipDeviceGetPCIBusId(pciBusId, 0, 0);
    REQUIRE(err == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetByPCIBusId round trip", "[device][pci]") {
    int originalDevice = 0;
    REQUIRE(hip().hipGetDevice(&originalDevice) == hipSuccess);

    // Get PCI bus ID
    char pciBusId[64] = {0};
    REQUIRE(hip().hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), originalDevice) == hipSuccess);

    // Get device by PCI bus ID
    int foundDevice = -1;
    REQUIRE(hip().hipDeviceGetByPCIBusId(&foundDevice, pciBusId) == hipSuccess);

    // Should be the same device
    REQUIRE(foundDevice == originalDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetByPCIBusId null device fails", "[device][pci][error]") {
    char pciBusId[64] = "0000:00:00.0";  // Dummy bus ID
    hipError_t err = hip().hipDeviceGetByPCIBusId(nullptr, pciBusId);
    REQUIRE(err == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetByPCIBusId null string fails", "[device][pci][error]") {
    int device = -1;
    hipError_t err = hip().hipDeviceGetByPCIBusId(&device, nullptr);
    REQUIRE(err == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetByPCIBusId invalid string behavior", "[device][pci]") {
    int device = -1;
    hipError_t err = hip().hipDeviceGetByPCIBusId(&device, "invalid:bus:id");
    // Note: Behavior varies by implementation.
    // Some return hipErrorInvalidValue, others may return hipSuccess with device=-1 or 0.
    // Just verify it doesn't crash.
    if (err == hipSuccess) {
        // If success, behavior is undefined but shouldn't crash
    } else {
        // Error is expected on most implementations
        REQUIRE((err == hipErrorInvalidValue || err == hipErrorNotFound || err == hipErrorInvalidDevice));
    }
}
