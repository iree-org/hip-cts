// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "test_runner.hpp"
#include <ostream>
#include <string>
#include <vector>

namespace hip_cts {
namespace runner {

// Generates a standalone HTML report for test results.
// The HTML embeds the XML data as a compressed (gzipped) base64 blob,
// which is decompressed and rendered client-side using JavaScript.
class HtmlReporter {
public:
    struct Options {
        std::string title;
        bool include_timestamps;
        bool include_commands;
        bool include_output;        // Include stdout/stderr
        bool collapsible_failures;  // Make failure details collapsible
        
        Options() 
            : title("HIP CTS Test Results")
            , include_timestamps(true)
            , include_commands(true)
            , include_output(true)
            , collapsible_failures(true) {}
    };
    
    explicit HtmlReporter(const Options& options = Options());
    
    // Generate HTML string with embedded compressed XML
    std::string generate(const AggregatedResults& results);
    
    // Write report to stream
    void write(const AggregatedResults& results, std::ostream& out);
    
    // Write report to file
    bool writeToFile(const AggregatedResults& results, const std::string& filename);
    
private:
    // Compress data using deflate (miniz)
    std::vector<uint8_t> compressData(const std::string& data);
    
    // Base64 encode binary data
    std::string base64Encode(const std::vector<uint8_t>& data);
    
    std::string escapeHtml(const std::string& str);
    std::string formatDuration(double seconds);
    
    Options options_;
};

} // namespace runner
} // namespace hip_cts
