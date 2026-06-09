// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// hipModuleLoadData must reject an unrecognized or unresolvable code-object
// image with a defined hipError_t — never crash, never report success. This
// feeds it a structurally-plausible but bogus image: a fat-binary wrapper
// bearing an out-of-band "HIPK" magic (recent ROCm builds pack some code
// objects in a sibling archive named by the wrapper's metadata) with a non-NULL
// payload that resolves to nothing.
//
// A conformant runtime fails it (unknown format, or archive-not-found) and
// stays alive; the test only requires that the call returns without success.
// GPU required for module loading.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

namespace {
struct HipFatBinaryWrapper {
  uint32_t magic;
  uint32_t version;
  void* binary;
  void* reserved;
};
constexpr uint32_t kHipkMagic = 0x4b504948u;  // "HIPK", little-endian
}  // namespace

TEST_CASE_METHOD(HipTestFixture,
                 "hipModuleLoadData rejects a bogus code-object image",
                 "[module][kpack][nightly]") {
  // Non-NULL payload so the loader gets past any NULL-image guard and actually
  // inspects the wrapper; the buffer is zeroed and names no real archive.
  std::vector<uint8_t> fake_metadata(4096, 0);
  HipFatBinaryWrapper wrapper = {kHipkMagic, 1u, fake_metadata.data(), nullptr};

  hipModule_t module = nullptr;
  hipError_t err = hip().hipModuleLoadData(&module, &wrapper);

  // Reaching this assertion proves the loader did not crash on the bogus image;
  // the result must be a failure (never hipSuccess for an image that resolves
  // to nothing).
  REQUIRE(err != hipSuccess);
}
