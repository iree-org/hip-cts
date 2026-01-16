// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "test_runner.hpp"
#include <ostream>
#include <string>

namespace hip_cts {
namespace runner {

// Generates Catch2-like text output for test results
class TextReporter {
public:
    struct Options {
        bool use_color;
        bool show_passed;       // Show individual passed tests
        bool show_timing;       // Show timing information
        bool verbose;           // Show more details
        
        Options() : use_color(true), show_passed(false), show_timing(true), verbose(false) {}
    };
    
    explicit TextReporter(const Options& options = Options());
    
    // Write report to stream
    void write(const AggregatedResults& results, std::ostream& out);
    
    // Write report to file
    bool writeToFile(const AggregatedResults& results, const std::string& filename);
    
private:
    void writeSuiteHeader(const TestSuiteResult& suite, std::ostream& out);
    void writeTestCase(const TestCaseResult& tc, std::ostream& out);
    void writeSuiteSummary(const TestSuiteResult& suite, std::ostream& out);
    void writeTotalSummary(const AggregatedResults& results, std::ostream& out);
    void writeSeparator(std::ostream& out, char ch = '=');
    
    // ANSI color codes
    std::string colorReset() const;
    std::string colorGreen() const;
    std::string colorRed() const;
    std::string colorYellow() const;
    std::string colorCyan() const;
    std::string colorBold() const;
    std::string colorDim() const;
    
    Options options_;
};

} // namespace runner
} // namespace hip_cts
