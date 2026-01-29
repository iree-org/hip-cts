# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test subdirectories - organized by category:
#
# Core functionality:
add_subdirectory(api_smoke)
add_subdirectory(device)
add_subdirectory(error)

# Memory operations (malloc, memcpy, memset, host memory, 2D/3D):
add_subdirectory(memory)

# Synchronization (streams, events, callbacks, sync primitives):
add_subdirectory(synchronization)

# Kernel execution:
add_subdirectory(kernel_smoke)

# Concurrency:
add_subdirectory(multithread)

# Graph API:
add_subdirectory(graph)