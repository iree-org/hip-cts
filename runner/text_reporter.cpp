// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "text_reporter.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace hip_cts {
namespace runner {

TextReporter::TextReporter(const Options& options) : options_(options) {}

std::string TextReporter::colorReset() const {
    return options_.use_color ? "\033[0m" : "";
}

std::string TextReporter::colorGreen() const {
    return options_.use_color ? "\033[32m" : "";
}

std::string TextReporter::colorRed() const {
    return options_.use_color ? "\033[31m" : "";
}

std::string TextReporter::colorYellow() const {
    return options_.use_color ? "\033[33m" : "";
}

std::string TextReporter::colorCyan() const {
    return options_.use_color ? "\033[36m" : "";
}

std::string TextReporter::colorBold() const {
    return options_.use_color ? "\033[1m" : "";
}

std::string TextReporter::colorDim() const {
    return options_.use_color ? "\033[2m" : "";
}

void TextReporter::writeSeparator(std::ostream& out, char ch) {
    out << std::string(78, ch) << "\n";
}

void TextReporter::writeSuiteHeader(const TestSuiteResult& suite, std::ostream& out) {
    out << "\n";
    writeSeparator(out, '-');
    out << colorBold() << colorCyan() << "Test Suite: " << colorReset() 
        << colorBold() << suite.name << colorReset() << "\n";
    if (options_.verbose) {
        out << colorDim() << "  Executable: " << suite.executable_path << colorReset() << "\n";
        out << colorDim() << "  Command: " << suite.command_line << colorReset() << "\n";
    }
    writeSeparator(out, '-');
}

void TextReporter::writeTestCase(const TestCaseResult& tc, std::ostream& out) {
    if (tc.passed && !tc.skipped && !options_.show_passed) {
        return;  // Skip passed tests unless verbose
    }
    
    std::string status;
    std::string status_color;
    
    if (tc.skipped) {
        status = "SKIPPED";
        status_color = colorYellow();
    } else if (tc.passed) {
        status = "PASSED";
        status_color = colorGreen();
    } else {
        status = "FAILED";
        status_color = colorRed();
    }
    
    out << "  " << status_color << "[" << status << "]" << colorReset() << " ";
    out << tc.name;
    
    if (options_.show_timing && tc.time_seconds > 0) {
        out << colorDim() << " (" << std::fixed << std::setprecision(3) 
            << tc.time_seconds << "s)" << colorReset();
    }
    out << "\n";
    
    // Show failure details
    if (!tc.passed && !tc.skipped && !tc.failure_message.empty()) {
        out << colorRed();
        // Indent failure message
        std::istringstream iss(tc.failure_message);
        std::string line;
        while (std::getline(iss, line)) {
            out << "    " << line << "\n";
        }
        out << colorReset();
    }
}

void TextReporter::writeSuiteSummary(const TestSuiteResult& suite, std::ostream& out) {
    int passed = suite.tests - suite.failures - suite.errors - suite.skipped;
    
    out << "\n  ";
    
    // Show counts with colors
    if (suite.failures > 0 || suite.errors > 0) {
        out << colorRed() << colorBold();
    } else {
        out << colorGreen() << colorBold();
    }
    
    out << passed << " passed" << colorReset() << ", ";
    
    if (suite.failures > 0) {
        out << colorRed() << suite.failures << " failed" << colorReset() << ", ";
    } else {
        out << "0 failed, ";
    }
    
    if (suite.errors > 0) {
        out << colorRed() << suite.errors << " errors" << colorReset() << ", ";
    }
    
    if (suite.skipped > 0) {
        out << colorYellow() << suite.skipped << " skipped" << colorReset() << ", ";
    }
    
    out << suite.tests << " total";
    
    if (options_.show_timing) {
        out << colorDim() << " (" << std::fixed << std::setprecision(3) 
            << suite.time_seconds << "s)" << colorReset();
    }
    
    out << "\n";
    
    if (suite.execution_failed) {
        out << colorRed() << "  Execution Error: " << suite.execution_error 
            << colorReset() << "\n";
    }
}

void TextReporter::writeTotalSummary(const AggregatedResults& results, std::ostream& out) {
    int passed = results.total_tests - results.total_failures - 
                 results.total_errors - results.total_skipped;
    
    out << "\n";
    writeSeparator(out, '=');
    out << colorBold() << "TOTAL RESULTS" << colorReset() << "\n";
    writeSeparator(out, '=');
    
    out << "\n";
    out << "  Test Suites: " << results.suites.size() << "\n";
    out << "  Total Tests: " << results.total_tests << "\n";
    out << "\n";
    
    // Results summary
    out << "  " << colorGreen() << colorBold() << "Passed:  " << colorReset() 
        << std::setw(6) << passed << "\n";
    
    out << "  " << colorRed() << colorBold() << "Failed:  " << colorReset() 
        << std::setw(6) << results.total_failures << "\n";
    
    if (results.total_errors > 0) {
        out << "  " << colorRed() << colorBold() << "Errors:  " << colorReset() 
            << std::setw(6) << results.total_errors << "\n";
    }
    
    if (results.total_skipped > 0) {
        out << "  " << colorYellow() << colorBold() << "Skipped: " << colorReset() 
            << std::setw(6) << results.total_skipped << "\n";
    }
    
    out << "\n";
    
    if (options_.show_timing) {
        out << "  Total Time: " << std::fixed << std::setprecision(3) 
            << results.total_time_seconds << "s\n";
    }
    
    out << "\n";
    
    // Final status line
    if (results.total_failures == 0 && results.total_errors == 0) {
        out << colorGreen() << colorBold() 
            << "  ✓ All tests passed!" << colorReset() << "\n";
    } else {
        out << colorRed() << colorBold() 
            << "  ✗ Some tests failed" << colorReset() << "\n";
    }
    
    out << "\n";
    
    // List failed tests if any
    if (results.total_failures > 0 || results.total_errors > 0) {
        out << colorRed() << colorBold() << "Failed Tests:" << colorReset() << "\n";
        for (const auto& suite : results.suites) {
            for (const auto& tc : suite.test_cases) {
                if (!tc.passed && !tc.skipped) {
                    out << "  - " << suite.name << " :: " << tc.name << "\n";
                }
            }
        }
        out << "\n";
    }
}

void TextReporter::write(const AggregatedResults& results, std::ostream& out) {
    // Header
    out << "\n";
    writeSeparator(out, '=');
    out << colorBold() << colorCyan() << "HIP CTS Test Runner" << colorReset() << "\n";
    out << "Version: " << results.runner_version << "\n";
    out << "Timestamp: " << results.timestamp << "\n";
    
    if (!results.hip_library.empty()) {
        out << "HIP Library: " << results.hip_library << "\n";
    }
    
    if (!results.extra_args.empty()) {
        out << "Extra Args: ";
        for (size_t i = 0; i < results.extra_args.size(); ++i) {
            if (i > 0) out << " ";
            out << results.extra_args[i];
        }
        out << "\n";
    }
    
    writeSeparator(out, '=');
    
    // Write each suite
    for (const auto& suite : results.suites) {
        writeSuiteHeader(suite, out);
        
        // Always show failed tests
        for (const auto& tc : suite.test_cases) {
            writeTestCase(tc, out);
        }
        
        writeSuiteSummary(suite, out);
    }
    
    // Total summary
    writeTotalSummary(results, out);
}

bool TextReporter::writeToFile(const AggregatedResults& results, 
                                const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        return false;
    }
    
    // Disable colors when writing to file
    Options file_options = options_;
    file_options.use_color = false;
    
    TextReporter file_reporter(file_options);
    file_reporter.write(results, file);
    
    return true;
}

} // namespace runner
} // namespace hip_cts
