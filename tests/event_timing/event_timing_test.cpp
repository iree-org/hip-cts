// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// hipEventElapsedTime must report GPU-reach duration around real device work
// (a strictly positive number), not host enqueue spacing. GPU required.

#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

TEST_CASE_METHOD(HipTestFixture,
                 "hipEventElapsedTime is strictly positive around GPU work",
                 "[event][nightly]") {
  const size_t kBytes = 64 * 1024 * 1024;  // large enough to be measurable
  void* src = nullptr;
  void* dst = nullptr;
  REQUIRE(hip().hipMalloc(&src, kBytes) == hipSuccess);
  REQUIRE(hip().hipMalloc(&dst, kBytes) == hipSuccess);

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  REQUIRE(hip().hipEventCreate(&start) == hipSuccess);
  REQUIRE(hip().hipEventCreate(&stop) == hipSuccess);

  REQUIRE(hip().hipEventRecord(start, nullptr) == hipSuccess);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(hip().hipMemcpy(dst, src, kBytes, hipMemcpyDeviceToDevice) ==
            hipSuccess);
  }
  REQUIRE(hip().hipEventRecord(stop, nullptr) == hipSuccess);
  REQUIRE(hip().hipEventSynchronize(stop) == hipSuccess);

  float ms = -1.0f;
  REQUIRE(hip().hipEventElapsedTime(&ms, start, stop) == hipSuccess);
  REQUIRE(ms > 0.0f);      // host-enqueue spacing would measure ~0
  REQUIRE(ms < 10000.0f);  // sane upper bound

  REQUIRE(hip().hipEventDestroy(start) == hipSuccess);
  REQUIRE(hip().hipEventDestroy(stop) == hipSuccess);
  REQUIRE(hip().hipFree(src) == hipSuccess);
  REQUIRE(hip().hipFree(dst) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture,
                 "hipEventElapsedTime rejects disable-timing events",
                 "[event][nightly]") {
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  REQUIRE(hip().hipEventCreateWithFlags(&a, hipEventDisableTiming) ==
          hipSuccess);
  REQUIRE(hip().hipEventCreateWithFlags(&b, hipEventDisableTiming) ==
          hipSuccess);
  REQUIRE(hip().hipEventRecord(a, nullptr) == hipSuccess);
  REQUIRE(hip().hipEventRecord(b, nullptr) == hipSuccess);
  REQUIRE(hip().hipEventSynchronize(b) == hipSuccess);

  // Timing disabled on both — elapsed time must be rejected, never a silent
  // value.
  float ms = 0.0f;
  REQUIRE(hip().hipEventElapsedTime(&ms, a, b) != hipSuccess);

  REQUIRE(hip().hipEventDestroy(a) == hipSuccess);
  REQUIRE(hip().hipEventDestroy(b) == hipSuccess);
}
