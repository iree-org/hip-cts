// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// hipGetLastError Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetLastError after success returns hipSuccess", "[error][getlast]") {
    // Ensure no previous error
    hip().hipGetLastError();
    
    // Successful operation
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
    
    // hipGetLastError should return success
    REQUIRE(hip().hipGetLastError() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetLastError after failure returns error code", "[error][getlast]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause an error - null pointer for malloc output
    REQUIRE(hip().hipMalloc(nullptr, 1024) == hipErrorInvalidValue);
    
    // hipGetLastError should return the error
    REQUIRE(hip().hipGetLastError() == hipErrorInvalidValue);
    
    // Calling again should return success (error is cleared)
    REQUIRE(hip().hipGetLastError() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetLastError clears error state", "[error][getlast]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause an error
    REQUIRE(hip().hipSetDevice(-1) == hipErrorInvalidDevice);
    
    // First call should return error
    REQUIRE(hip().hipGetLastError() == hipErrorInvalidDevice);
    
    // Second call should return success
    REQUIRE(hip().hipGetLastError() == hipSuccess);
}

//=============================================================================
// hipPeekAtLastError Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipPeekAtLastError after success returns hipSuccess", "[error][peek]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Successful operation
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
    
    // hipPeekAtLastError should return success
    REQUIRE(hip().hipPeekAtLastError() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPeekAtLastError after failure returns error code", "[error][peek]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause an error
    REQUIRE(hip().hipMalloc(nullptr, 1024) == hipErrorInvalidValue);
    
    // hipPeekAtLastError should return the error
    REQUIRE(hip().hipPeekAtLastError() == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipPeekAtLastError does not clear error state", "[error][peek]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause an error
    REQUIRE(hip().hipSetDevice(-1) == hipErrorInvalidDevice);
    
    // First peek should return error
    REQUIRE(hip().hipPeekAtLastError() == hipErrorInvalidDevice);
    
    // Second peek should also return error (not cleared)
    REQUIRE(hip().hipPeekAtLastError() == hipErrorInvalidDevice);
    
    // hipGetLastError clears the error
    REQUIRE(hip().hipGetLastError() == hipErrorInvalidDevice);
    
    // Now both should return success
    REQUIRE(hip().hipPeekAtLastError() == hipSuccess);
}

//=============================================================================
// hipGetErrorName Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorName for hipSuccess", "[error][name]") {
    const char* name = hip().hipGetErrorName(hipSuccess);
    REQUIRE(name != nullptr);
    // Name should contain "success" (case insensitive check not needed - just verify non-empty)
    REQUIRE(strlen(name) > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorName for common errors", "[error][name]") {
    SECTION("hipErrorInvalidValue") {
        const char* name = hip().hipGetErrorName(hipErrorInvalidValue);
        REQUIRE(name != nullptr);
        REQUIRE(strlen(name) > 0);
    }
    
    SECTION("hipErrorOutOfMemory") {
        const char* name = hip().hipGetErrorName(hipErrorOutOfMemory);
        REQUIRE(name != nullptr);
        REQUIRE(strlen(name) > 0);
    }
    
    SECTION("hipErrorInvalidDevice") {
        const char* name = hip().hipGetErrorName(hipErrorInvalidDevice);
        REQUIRE(name != nullptr);
        REQUIRE(strlen(name) > 0);
    }
    
    SECTION("hipErrorInvalidResourceHandle") {
        const char* name = hip().hipGetErrorName(hipErrorInvalidResourceHandle);
        REQUIRE(name != nullptr);
        REQUIRE(strlen(name) > 0);
    }
    
    SECTION("hipErrorNotReady") {
        const char* name = hip().hipGetErrorName(hipErrorNotReady);
        REQUIRE(name != nullptr);
        REQUIRE(strlen(name) > 0);
    }
}

//=============================================================================
// hipGetErrorString Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorString for hipSuccess", "[error][string]") {
    const char* str = hip().hipGetErrorString(hipSuccess);
    REQUIRE(str != nullptr);
    // Should contain "success" or similar
    REQUIRE(strlen(str) > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorString for common errors", "[error][string]") {
    SECTION("hipErrorInvalidValue") {
        const char* str = hip().hipGetErrorString(hipErrorInvalidValue);
        REQUIRE(str != nullptr);
        REQUIRE(strlen(str) > 0);
    }
    
    SECTION("hipErrorOutOfMemory") {
        const char* str = hip().hipGetErrorString(hipErrorOutOfMemory);
        REQUIRE(str != nullptr);
        REQUIRE(strlen(str) > 0);
    }
    
    SECTION("hipErrorInvalidDevice") {
        const char* str = hip().hipGetErrorString(hipErrorInvalidDevice);
        REQUIRE(str != nullptr);
        REQUIRE(strlen(str) > 0);
    }
}

//=============================================================================
// Error Handling Integration Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Error state independent across operations", "[error][integration]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause an error
    REQUIRE(hip().hipMalloc(nullptr, 1024) == hipErrorInvalidValue);
    
    // Successful operations should not clear error state
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    
    // Error should still be recorded
    REQUIRE(hip().hipGetLastError() == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Error state updates with new errors", "[error][integration]") {
    // Clear any previous error
    hip().hipGetLastError();
    
    // Cause first error
    REQUIRE(hip().hipMalloc(nullptr, 1024) == hipErrorInvalidValue);
    
    // Cause second error (different type)
    REQUIRE(hip().hipSetDevice(-1) == hipErrorInvalidDevice);
    
    // hipGetLastError should return the most recent error
    REQUIRE(hip().hipGetLastError() == hipErrorInvalidDevice);
    
    // Error should be cleared
    REQUIRE(hip().hipGetLastError() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multiple error-check cycles", "[error][integration]") {
    for (int i = 0; i < 5; ++i) {
        // Clear error state
        hip().hipGetLastError();
        
        // Cause an error
        REQUIRE(hip().hipMalloc(nullptr, 1024) == hipErrorInvalidValue);
        
        // Check error
        REQUIRE(hip().hipGetLastError() == hipErrorInvalidValue);
        
        // Should be cleared
        REQUIRE(hip().hipGetLastError() == hipSuccess);
    }
}
