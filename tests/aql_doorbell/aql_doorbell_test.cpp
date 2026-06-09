// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sustained host<->device copy integrity. Many back-to-back H2D/D2H roundtrips
// of a multi-MiB buffer, each followed by a full element-wise comparison, so a
// byte dropped or reordered on any transfer fails the test. Each hipMemcpy also
// submits to the stream's queue, so the loop doubles as a stress test of the
// queue-submission path under sustained use. GPU required.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

TEST_CASE_METHOD(HipTestFixture,
                 "Repeated H2D/D2H roundtrips preserve data integrity",
                 "[memcpy][doorbell][nightly]") {
  const size_t kCount = 1u << 20;  // 1M uint32 = 4 MiB
  const size_t kBytes = kCount * sizeof(uint32_t);
  std::vector<uint32_t> host_in(kCount);
  std::vector<uint32_t> host_out(kCount, 0u);
  for (size_t i = 0; i < kCount; ++i) {
    host_in[i] = static_cast<uint32_t>(i * 2654435761u);
  }

  void* dev = nullptr;
  REQUIRE(hip().hipMalloc(&dev, kBytes) == hipSuccess);

  // 32 iterations, each an H2D copy followed by a D2H copy of the same buffer.
  // A conformant runtime returns the bytes intact on every round trip.
  for (int iter = 0; iter < 32; ++iter) {
    REQUIRE(hip().hipMemcpy(dev, host_in.data(), kBytes,
                            hipMemcpyHostToDevice) == hipSuccess);
    std::fill(host_out.begin(), host_out.end(), 0u);
    REQUIRE(hip().hipMemcpy(host_out.data(), dev, kBytes,
                            hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(host_out == host_in);
  }

  REQUIRE(hip().hipFree(dev) == hipSuccess);
}
