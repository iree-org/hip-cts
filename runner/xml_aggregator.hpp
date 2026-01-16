// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "test_runner.hpp"
#include <string>
#include <ostream>

namespace hip_cts {
namespace runner {

// Aggregates test results into JUnit-compatible XML format
class XmlAggregator {
public:
    // Generate aggregated XML from results
    static std::string generate(const AggregatedResults& results);
    
    // Write aggregated XML to stream
    static void write(const AggregatedResults& results, std::ostream& out);
    
    // Write aggregated XML to file
    static bool writeToFile(const AggregatedResults& results, 
                            const std::string& filename);
};

} // namespace runner
} // namespace hip_cts
