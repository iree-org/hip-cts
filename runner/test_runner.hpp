// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <filesystem>

namespace hip_cts {
namespace runner {

// Information about a single test case (for enumeration)
struct TestCaseInfo {
    std::string name;
    std::string class_name;      // Test suite/fixture name
    std::vector<std::string> tags;
};

// Information about a test suite/executable (for enumeration)
struct TestSuiteInfo {
    std::string name;            // Executable name
    std::filesystem::path executable_path;
    std::vector<TestCaseInfo> test_cases;
};

// Result of a single test case
struct TestCaseResult {
    std::string name;
    std::string class_name;      // Test suite/fixture name
    double time_seconds = 0.0;
    bool passed = true;
    bool skipped = false;
    std::string failure_message;
    std::string failure_type;
    std::string stdout_output;
    std::string stderr_output;
};

// Result of a single test suite (one executable)
struct TestSuiteResult {
    std::string name;            // Executable name
    std::string executable_path;
    int tests = 0;
    int failures = 0;
    int errors = 0;
    int skipped = 0;
    double time_seconds = 0.0;
    std::string timestamp;
    std::vector<TestCaseResult> test_cases;
    
    // Execution info
    int exit_code = 0;
    std::string command_line;
    std::string stdout_output;
    std::string stderr_output;
    bool execution_failed = false;
    std::string execution_error;
};

// Aggregated results from all test suites
struct AggregatedResults {
    std::string name = "HIP CTS";
    int total_tests = 0;
    int total_failures = 0;
    int total_errors = 0;
    int total_skipped = 0;
    double total_time_seconds = 0.0;
    std::string timestamp;
    std::vector<TestSuiteResult> suites;
    
    // Runner info
    std::string runner_version = "1.0.0";
    std::string hip_library;
    std::vector<std::string> extra_args;
};

// Callback for progress reporting
using ProgressCallback = std::function<void(const std::string& suite_name, 
                                             int current, int total)>;

// Test runner that discovers and executes test executables
class TestRunner {
public:
    TestRunner();
    ~TestRunner() = default;
    
    // Configuration
    void setTestDirectory(const std::filesystem::path& dir);
    void setHipLibrary(const std::string& library_path);
    void addExtraArg(const std::string& arg);
    void setFilter(const std::string& filter);
    void setVerbose(bool verbose);
    void setProgressCallback(ProgressCallback callback);
    
    // Add a specific test executable
    void addTestExecutable(const std::filesystem::path& executable);
    
    // Discover test executables in the configured directory
    std::vector<std::filesystem::path> discoverTests();
    
    // Enumerate all tests in a single executable (without running them)
    TestSuiteInfo enumerateTests(const std::filesystem::path& executable);
    
    // Enumerate all tests in all discovered/added executables
    std::vector<TestSuiteInfo> enumerateAllTests();
    std::vector<TestSuiteInfo> enumerateAllTestsParallel(int num_threads);
    
    // Run all discovered/added tests
    AggregatedResults runAllTests();
    
    // Run a single test executable and return its results
    TestSuiteResult runTest(const std::filesystem::path& executable);
    
    // Run specific tests from a specific executable
    TestSuiteResult runTest(const std::filesystem::path& executable, 
                           const std::vector<std::string>& test_names);
    
private:
    // Parse JUnit XML output from Catch2
    TestSuiteResult parseJUnitXml(const std::string& xml_content,
                                   const std::filesystem::path& executable);
    
    // Execute a command and capture output
    struct ExecResult {
        int exit_code;
        std::string stdout_output;
        std::string stderr_output;
    };
    ExecResult executeCommand(const std::vector<std::string>& args);
    
    std::filesystem::path test_directory_;
    std::string hip_library_;
    std::vector<std::string> extra_args_;
    std::string filter_;
    bool verbose_ = false;
    ProgressCallback progress_callback_;
    std::vector<std::filesystem::path> test_executables_;
};

} // namespace runner
} // namespace hip_cts
