// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Helper Functions
//=============================================================================

static void createThenDestroyStreams(int iterations, int burstSize) {
    std::vector<hipStream_t> streams(burstSize);
    
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < burstSize; ++j) {
            REQUIRE(hip().hipStreamCreate(&streams[j]) == hipSuccess);
        }
        for (int j = 0; j < burstSize; ++j) {
            REQUIRE(hip().hipStreamDestroy(streams[j]) == hipSuccess);
        }
    }
}

static void waitStreams(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    }
}

//=============================================================================
// Basic Multi-thread Stream Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multi-thread stream create destroy serial", "[multithread][stream]") {
    createThenDestroyStreams(10, 10);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread serial pyramid", "[multithread][stream]") {
    constexpr int iters = 3;
    constexpr int maxBurstSize = 40;
    
    std::thread t1(createThenDestroyStreams, iters * 1, maxBurstSize);
    t1.join();
    
    std::thread t2(createThenDestroyStreams, iters * 10, 10);
    t2.join();
    
    std::thread t3(createThenDestroyStreams, iters * 100, 1);
    t3.join();
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread parallel pyramid", "[multithread][stream]") {
    constexpr int iters = 3;
    constexpr int maxBurstSize = 40;
    
    std::thread t1(createThenDestroyStreams, iters * 1, maxBurstSize);
    std::thread t2(createThenDestroyStreams, iters * 10, 10);
    std::thread t3(createThenDestroyStreams, iters * 100, 1);
    
    t1.join();
    t2.join();
    t3.join();
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread near zero streams", "[multithread][stream]") {
    constexpr int iters = 100;
    
    std::thread t1(createThenDestroyStreams, iters, 1);
    std::thread t2(createThenDestroyStreams, iters, 1);
    std::thread t3(waitStreams, iters * 5);
    
    t1.join();
    t2.join();
    t3.join();
}

//=============================================================================
// Multi-thread Memory Operations
//=============================================================================

static void memsetThread(int threadId, hipStream_t stream, size_t numIterations) {
    constexpr size_t size = 1024;
    void* devicePtr = nullptr;
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    std::vector<uint8_t> hostData(size);
    
    for (size_t iter = 0; iter < numIterations; ++iter) {
        uint8_t value = static_cast<uint8_t>((threadId * 17 + iter) & 0xFF);
        
        REQUIRE(hip().hipMemsetAsync(devicePtr, value, size, stream) == hipSuccess);
        REQUIRE(hip().hipMemcpyAsync(hostData.data(), devicePtr, size,
                                      hipMemcpyDeviceToHost, stream) == hipSuccess);
        REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(hostData[i] == value);
        }
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread memset own streams", "[multithread][memory]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 10;
    
    auto memsetThreadOwn = [](int threadId, size_t numIter) {
        hipStream_t stream = nullptr;
        REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
        memsetThread(threadId, stream, numIter);
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(memsetThreadOwn, i, numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread memset different streams", "[multithread][memory]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 10;
    
    std::vector<hipStream_t> streams(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
    }
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(memsetThread, i, streams[i], numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Multi-thread Allocation Tests
//=============================================================================

static void allocFreeThread(int threadId, size_t numIterations) {
    std::mt19937 rng(threadId);
    std::uniform_int_distribution<size_t> sizeDist(1, 1024 * 1024);
    
    for (size_t iter = 0; iter < numIterations; ++iter) {
        size_t size = sizeDist(rng);
        void* devicePtr = nullptr;
        
        REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
        REQUIRE(devicePtr != nullptr);
        REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread malloc free parallel", "[multithread][memory]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 50;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(allocFreeThread, i, numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

//=============================================================================
// Multi-thread Event Tests
//=============================================================================

static void eventThreadOwnStream(int threadId, size_t numIterations) {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    for (size_t iter = 0; iter < numIterations; ++iter) {
        hipEvent_t event = nullptr;
        REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
        REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
        REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
        REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread events own streams", "[multithread][event]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 20;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(eventThreadOwnStream, i, numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

static void eventThread(int threadId, hipStream_t stream, size_t numIterations) {
    for (size_t iter = 0; iter < numIterations; ++iter) {
        hipEvent_t event = nullptr;
        REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
        REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
        REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
        REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread events different streams", "[multithread][event]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 20;
    
    std::vector<hipStream_t> streams(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
    }
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(eventThread, i, streams[i], numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Multi-thread Memcpy Tests
//=============================================================================

static void memcpyThread(int threadId, hipStream_t stream, size_t numIterations) {
    constexpr size_t size = 4096;
    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    std::vector<uint8_t> verifyBuffer(size);
    
    for (size_t iter = 0; iter < numIterations; ++iter) {
        uint8_t pattern = static_cast<uint8_t>((threadId * 31 + iter) & 0xFF);
        
        memset(hostBytes, pattern, size);
        
        REQUIRE(hip().hipMemcpyAsync(devicePtr, hostPtr, size, 
                                      hipMemcpyHostToDevice, stream) == hipSuccess);
        REQUIRE(hip().hipMemcpyAsync(verifyBuffer.data(), devicePtr, size,
                                      hipMemcpyDeviceToHost, stream) == hipSuccess);
        REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(verifyBuffer[i] == pattern);
        }
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread memcpy own streams", "[multithread][memcpy]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 10;
    
    auto memcpyThreadOwn = [](int threadId, size_t numIter) {
        hipStream_t stream = nullptr;
        REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
        memcpyThread(threadId, stream, numIter);
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(memcpyThreadOwn, i, numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread memcpy different streams", "[multithread][memcpy]") {
    constexpr int numThreads = 4;
    constexpr size_t numIterations = 10;
    
    std::vector<hipStream_t> streams(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
    }
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(memcpyThread, i, streams[i], numIterations);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    for (int i = 0; i < numThreads; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Mixed Operations Multi-thread Test
//=============================================================================

static void mixedOperationsThread(int threadId) {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr size_t size = 2048;
    void* devicePtr = nullptr;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    std::vector<uint8_t> hostData(size);
    
    for (int iter = 0; iter < 5; ++iter) {
        uint8_t value = static_cast<uint8_t>((threadId * 13 + iter) & 0xFF);
        
        REQUIRE(hip().hipMemsetAsync(devicePtr, value, size, stream) == hipSuccess);
        
        hipEvent_t event = nullptr;
        REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
        REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
        
        REQUIRE(hip().hipMemcpyAsync(hostData.data(), devicePtr, size,
                                      hipMemcpyDeviceToHost, stream) == hipSuccess);
        
        REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
        REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(hostData[i] == value);
        }
        
        REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread mixed operations", "[multithread][integration]") {
    constexpr int numThreads = 8;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(mixedOperationsThread, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

//=============================================================================
// Stress Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multi-thread stress stream operations", "[multithread][stress]") {
    constexpr int numThreads = 8;
    constexpr int numIterations = 20;
    
    std::atomic<int> completedThreads{0};
    
    auto threadFunc = [&completedThreads](int threadId) {
        for (int iter = 0; iter < numIterations; ++iter) {
            hipStream_t stream = nullptr;
            REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
            
            void* ptr = nullptr;
            REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
            REQUIRE(hip().hipMemsetAsync(ptr, threadId, 1024, stream) == hipSuccess);
            REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
            REQUIRE(hip().hipFree(ptr) == hipSuccess);
            
            REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
        }
        completedThreads++;
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(completedThreads == numThreads);
}

TEST_CASE_METHOD(HipTestFixture, "Multi-thread device synchronize stress", "[multithread][stress]") {
    constexpr int numThreads = 4;
    constexpr int numIterations = 50;
    
    std::atomic<int> syncCount{0};
    
    auto threadFunc = [&syncCount](int threadId) {
        for (int iter = 0; iter < numIterations; ++iter) {
            REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
            syncCount++;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(syncCount == numThreads * numIterations);
}
