# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_hip_cts_test(synchronization_stream_test
    SOURCES stream_test.cpp
)

add_hip_cts_test(synchronization_event_test
    SOURCES event_test.cpp
)

add_hip_cts_test(synchronization_sync_test
    SOURCES sync_test.cpp
)

add_hip_cts_test(synchronization_callback_test
    SOURCES callback_test.cpp
)
