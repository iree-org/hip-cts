// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ABI-surface check: a libamdhip64 targeting the recent HIP ABI must export
// these extended entry points. They were added across HIP versions
// (hipDrvLaunchKernelEx is versioned @hip_6.5) and are absent from older
// builds, so a runtime or a downstream library linked against the newer
// symbols fails to load if any is missing.
//
// Presence only: the test does not call them. Their argument contracts differ
// and a real implementation would dereference the (here absent) arguments, so
// invoking them generically would be undefined behavior. This is GPU-free: it
// dlopens the library under test and resolves each symbol.

#include <dlfcn.h>

#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"

namespace {
const char* const kExtendedSymbols[] = {
    "hipDrvLaunchKernelEx",  // versioned @hip_6.5
    "hipGetFuncBySymbol",
    "hipStreamBatchMemOp",
    "hipMemGetHandleForAddressRange",
    "hipMemcpyBatchAsync",
    "hipImportExternalMemory",
    "hipExternalMemoryGetMappedBuffer",
    "hipDestroyExternalMemory",
};
}  // namespace

TEST_CASE("libamdhip64 exports the extended HIP entry points",
          "[symbols][nightly]") {
  // Re-open the same binary the loader resolved (RTLD_NOW forces every
  // relocation).
  const std::string& lib = HipLoader::instance().libraryPath();
  void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
  REQUIRE(handle != nullptr);

  for (const char* name : kExtendedSymbols) {
    INFO("symbol: " << name);
    // A -fvisibility regression or a version script that re-hides these would
    // fail here.
    REQUIRE(dlsym(handle, name) != nullptr);
  }

  dlclose(handle);
}
