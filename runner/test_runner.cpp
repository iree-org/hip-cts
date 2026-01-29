// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_runner.hpp"
#include "tinyxml2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hip_cts {
namespace runner {

namespace {

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

// Create a temporary file and return its path
std::filesystem::path createTempFile(const std::string& prefix) {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
#ifdef _WIN32
    char temp_name[MAX_PATH];
    GetTempFileNameA(temp_dir.string().c_str(), prefix.c_str(), 0, temp_name);
    return temp_name;
#else
    std::string template_path = (temp_dir / (prefix + "XXXXXX")).string();
    std::vector<char> path_buf(template_path.begin(), template_path.end());
    path_buf.push_back('\0');
    int fd = mkstemp(path_buf.data());
    if (fd >= 0) {
        close(fd);
        return std::string(path_buf.data());
    }
    // Fallback
    return temp_dir / (prefix + std::to_string(std::rand()));
#endif
}

} // namespace

TestRunner::TestRunner() = default;

void TestRunner::setTestDirectory(const std::filesystem::path& dir) {
    test_directory_ = dir;
}

void TestRunner::setHipLibrary(const std::string& library_path) {
    hip_library_ = library_path;
}

void TestRunner::addExtraArg(const std::string& arg) {
    extra_args_.push_back(arg);
}

void TestRunner::setFilter(const std::string& filter) {
    filter_ = filter;
}

void TestRunner::setVerbose(bool verbose) {
    verbose_ = verbose;
}

void TestRunner::setProgressCallback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

void TestRunner::addTestExecutable(const std::filesystem::path& executable) {
    test_executables_.push_back(executable);
}

std::vector<std::filesystem::path> TestRunner::discoverTests() {
    std::vector<std::filesystem::path> discovered;
    
    if (test_directory_.empty() || !std::filesystem::exists(test_directory_)) {
        return discovered;
    }
    
    // Look for executables in the test directory recursively
    for (const auto& entry : std::filesystem::recursive_directory_iterator(test_directory_)) {
        if (!entry.is_regular_file()) continue;
        
        const auto& path = entry.path();
        
        // Skip files with common non-executable extensions
        std::string ext = path.extension().string();
        if (ext == ".txt" || ext == ".xml" || ext == ".json" || 
            ext == ".cmake" || ext == ".cpp" || ext == ".hpp" ||
            ext == ".h" || ext == ".c" || ext == ".o" || ext == ".a" ||
            ext == ".so" || ext == ".dylib" || ext == ".dll") {
            continue;
        }
        
        // Check if it's an executable
#ifdef _WIN32
        if (ext == ".exe") {
            discovered.push_back(path);
        }
#else
        // On Unix, check execute permission
        auto status = std::filesystem::status(path);
        if ((status.permissions() & std::filesystem::perms::owner_exec) != 
            std::filesystem::perms::none) {
            // Additional check: skip if it's in a CMake build directory internal folder
            std::string path_str = path.string();
            if (path_str.find("CMakeFiles") != std::string::npos) {
                continue;
            }
            // Skip runner itself
            if (path.filename().string().find("hip_cts_runner") != std::string::npos) {
                continue;
            }
            discovered.push_back(path);
        }
#endif
    }
    
    // Sort for deterministic order
    std::sort(discovered.begin(), discovered.end());
    
    return discovered;
}

TestRunner::ExecResult TestRunner::executeCommand(const std::vector<std::string>& args) {
    ExecResult result;
    result.exit_code = -1;
    
    if (args.empty()) {
        return result;
    }
    
#ifdef _WIN32
    // Windows implementation using CreateProcess
    std::string cmd_line;
    for (const auto& arg : args) {
        if (!cmd_line.empty()) cmd_line += " ";
        // Quote arguments with spaces
        if (arg.find(' ') != std::string::npos) {
            cmd_line += "\"" + arg + "\"";
        } else {
            cmd_line += arg;
        }
    }
    
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;
    
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    
    CreatePipe(&stdout_read, &stdout_write, &sa, 0);
    CreatePipe(&stderr_read, &stderr_write, &sa, 0);
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0);
    
    PROCESS_INFORMATION pi;
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;
    si.dwFlags |= STARTF_USESTDHANDLES;
    
    std::vector<char> cmd_buf(cmd_line.begin(), cmd_line.end());
    cmd_buf.push_back('\0');
    
    if (CreateProcessA(NULL, cmd_buf.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(stdout_write);
        CloseHandle(stderr_write);
        
        // Read output
        char buffer[4096];
        DWORD bytes_read;
        while (ReadFile(stdout_read, buffer, sizeof(buffer) - 1, &bytes_read, NULL) && bytes_read > 0) {
            buffer[bytes_read] = '\0';
            result.stdout_output += buffer;
        }
        while (ReadFile(stderr_read, buffer, sizeof(buffer) - 1, &bytes_read, NULL) && bytes_read > 0) {
            buffer[bytes_read] = '\0';
            result.stderr_output += buffer;
        }
        
        WaitForSingleObject(pi.hProcess, INFINITE);
        
        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        result.exit_code = static_cast<int>(exit_code);
        
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(stdout_read);
        CloseHandle(stderr_read);
    }
#else
    // Unix implementation using fork/exec
    int stdout_pipe[2];
    int stderr_pipe[2];
    
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        return result;
    }
    
    pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        return result;
    }
    
    if (pid == 0) {
        // Child process
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        
        // Convert args to char* array
        std::vector<char*> argv;
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);
        
        execv(args[0].c_str(), argv.data());
        _exit(127);
    }
    
    // Parent process
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    
    // Read from pipes
    char buffer[4096];
    ssize_t n;
    
    // Use select to read from both pipes
    fd_set read_fds;
    int max_fd = std::max(stdout_pipe[0], stderr_pipe[0]) + 1;
    bool stdout_open = true, stderr_open = true;
    
    while (stdout_open || stderr_open) {
        FD_ZERO(&read_fds);
        if (stdout_open) FD_SET(stdout_pipe[0], &read_fds);
        if (stderr_open) FD_SET(stderr_pipe[0], &read_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;  // 100ms
        
        int ready = select(max_fd, &read_fds, nullptr, nullptr, &timeout);
        if (ready < 0) break;
        
        if (stdout_open && FD_ISSET(stdout_pipe[0], &read_fds)) {
            n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1);
            if (n > 0) {
                buffer[n] = '\0';
                result.stdout_output += buffer;
            } else {
                stdout_open = false;
            }
        }
        
        if (stderr_open && FD_ISSET(stderr_pipe[0], &read_fds)) {
            n = read(stderr_pipe[0], buffer, sizeof(buffer) - 1);
            if (n > 0) {
                buffer[n] = '\0';
                result.stderr_output += buffer;
            } else {
                stderr_open = false;
            }
        }
        
        // Check if child has exited
        int status;
        pid_t wpid = waitpid(pid, &status, WNOHANG);
        if (wpid == pid) {
            // Child exited, drain remaining output
            while ((n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[n] = '\0';
                result.stdout_output += buffer;
            }
            while ((n = read(stderr_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[n] = '\0';
                result.stderr_output += buffer;
            }
            
            if (WIFEXITED(status)) {
                result.exit_code = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                result.exit_code = 128 + WTERMSIG(status);
            }
            break;
        }
    }
    
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);
    
    // If we didn't get exit code yet, wait for it
    if (result.exit_code < 0) {
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            result.exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            result.exit_code = 128 + WTERMSIG(status);
        }
    }
#endif
    
    return result;
}

TestSuiteResult TestRunner::parseJUnitXml(const std::string& xml_content,
                                           const std::filesystem::path& executable) {
    TestSuiteResult result;
    result.name = executable.filename().string();
    result.executable_path = executable.string();
    result.timestamp = getCurrentTimestamp();
    
    tinyxml2::XMLDocument doc;
    if (doc.Parse(xml_content.c_str()) != tinyxml2::XML_SUCCESS) {
        result.execution_failed = true;
        result.execution_error = "Failed to parse JUnit XML output";
        return result;
    }
    
    // Catch2 outputs either <testsuites> or <testsuite> as root
    tinyxml2::XMLElement* root = doc.RootElement();
    if (!root) {
        result.execution_failed = true;
        result.execution_error = "Empty XML document";
        return result;
    }
    
    std::vector<tinyxml2::XMLElement*> test_suite_elements;
    
    if (std::string(root->Name()) == "testsuites") {
        // Multiple test suites
        for (auto* suite = root->FirstChildElement("testsuite"); 
             suite; 
             suite = suite->NextSiblingElement("testsuite")) {
            test_suite_elements.push_back(suite);
        }
    } else if (std::string(root->Name()) == "testsuite") {
        test_suite_elements.push_back(root);
    }
    
    // Parse all test suites
    for (auto* suite : test_suite_elements) {
        result.tests += suite->IntAttribute("tests", 0);
        result.failures += suite->IntAttribute("failures", 0);
        result.errors += suite->IntAttribute("errors", 0);
        result.skipped += suite->IntAttribute("skipped", 0);
        result.time_seconds += suite->DoubleAttribute("time", 0.0);
        
        // Parse test cases
        for (auto* testcase = suite->FirstChildElement("testcase");
             testcase;
             testcase = testcase->NextSiblingElement("testcase")) {
            TestCaseResult tc;
            tc.name = testcase->Attribute("name") ? testcase->Attribute("name") : "";
            tc.class_name = testcase->Attribute("classname") ? testcase->Attribute("classname") : "";
            tc.time_seconds = testcase->DoubleAttribute("time", 0.0);
            
            // Check for failure
            auto* failure = testcase->FirstChildElement("failure");
            if (failure) {
                tc.passed = false;
                tc.failure_message = failure->GetText() ? failure->GetText() : "";
                tc.failure_type = failure->Attribute("type") ? failure->Attribute("type") : "";
            }
            
            // Check for error
            auto* error = testcase->FirstChildElement("error");
            if (error) {
                tc.passed = false;
                tc.failure_message = error->GetText() ? error->GetText() : "";
                tc.failure_type = error->Attribute("type") ? error->Attribute("type") : "error";
            }
            
            // Check for skipped
            auto* skipped_el = testcase->FirstChildElement("skipped");
            if (skipped_el) {
                tc.skipped = true;
            }
            
            // System out/err
            auto* system_out = testcase->FirstChildElement("system-out");
            if (system_out && system_out->GetText()) {
                tc.stdout_output = system_out->GetText();
            }
            
            auto* system_err = testcase->FirstChildElement("system-err");
            if (system_err && system_err->GetText()) {
                tc.stderr_output = system_err->GetText();
            }
            
            result.test_cases.push_back(std::move(tc));
        }
    }
    
    return result;
}

TestSuiteResult TestRunner::runTest(const std::filesystem::path& executable) {
    TestSuiteResult result;
    result.name = executable.filename().string();
    result.executable_path = executable.string();
    result.timestamp = getCurrentTimestamp();
    
    // Create temp file for XML output
    auto xml_output_path = createTempFile("hip_cts_");
    
    // Build command line
    std::vector<std::string> args;
    args.push_back(executable.string());
    
    // Add HIP library argument if specified
    if (!hip_library_.empty()) {
        args.push_back("--hip-library");
        args.push_back(hip_library_);
    }
    
    // Add extra arguments
    for (const auto& arg : extra_args_) {
        args.push_back(arg);
    }
    
    // Add filter if specified
    if (!filter_.empty()) {
        args.push_back("\"" + filter_ + "\"");
    }
    
    // Tell Catch2 to output JUnit XML
    args.push_back("-r");
    args.push_back("junit");
    args.push_back("-o");
    args.push_back(xml_output_path.string());
    
    // Build command line string for logging
    std::ostringstream cmd_oss;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) cmd_oss << " ";
        cmd_oss << args[i];
    }
    result.command_line = cmd_oss.str();
    
    if (verbose_) {
        std::cerr << "Running: " << result.command_line << std::endl;
    }
    
    // Execute the test
    auto start = std::chrono::steady_clock::now();
    auto exec_result = executeCommand(args);
    auto end = std::chrono::steady_clock::now();
    
    result.exit_code = exec_result.exit_code;
    result.stdout_output = exec_result.stdout_output;
    result.stderr_output = exec_result.stderr_output;
    
    // Read the XML output
    if (std::filesystem::exists(xml_output_path)) {
        std::ifstream xml_file(xml_output_path);
        std::stringstream buffer;
        buffer << xml_file.rdbuf();
        std::string xml_content = buffer.str();
        
        if (!xml_content.empty()) {
            result = parseJUnitXml(xml_content, executable);
            result.exit_code = exec_result.exit_code;
            result.stdout_output = exec_result.stdout_output;
            result.stderr_output = exec_result.stderr_output;
            result.command_line = cmd_oss.str();
        }
        
        // Clean up temp file
        std::filesystem::remove(xml_output_path);
    } else {
        // No XML output - execution may have failed
        result.execution_failed = true;
        result.execution_error = "Test executable did not produce XML output";
    }
    
    // Use wall clock time if we don't have test time
    if (result.time_seconds == 0.0) {
        result.time_seconds = std::chrono::duration<double>(end - start).count();
    }
    
    return result;
}

TestSuiteInfo TestRunner::enumerateTests(const std::filesystem::path& executable) {
    TestSuiteInfo info;
    info.name = executable.filename().string();
    info.executable_path = executable;
    
    // Build command line to list tests
    std::vector<std::string> args;
    args.push_back(executable.string());
    
    // Add HIP library argument if specified
    if (!hip_library_.empty()) {
        args.push_back("--hip-library");
        args.push_back(hip_library_);
    }
    
    // Tell Catch2 to list tests with tags
    args.push_back("--list-tests");
    
    auto exec_result = executeCommand(args);
    
    if (exec_result.exit_code != 0) {
        return info;  // Return empty list on failure
    }
    
    // Parse the output - Catch2 outputs tests in format:
    // All available test cases:
    //   TestName
    //       [tag1][tag2]
    //   NextTestName
    //       [tag1]
    // NN test cases
    std::istringstream stream(exec_result.stdout_output);
    std::string line;
    TestCaseInfo current_test;
    bool has_test = false;
    bool in_test_list = false;
    
    while (std::getline(stream, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // Look for the start of test list
        if (line.find("All available test cases:") != std::string::npos) {
            in_test_list = true;
            continue;
        }
        
        // Skip lines before test list
        if (!in_test_list) continue;
        
        // Check for end of test list (line like "26 test cases")
        if (line.find(" test case") != std::string::npos) {
            break;
        }
        
        // Count leading spaces to determine if this is a test name or tags
        size_t leading_spaces = 0;
        for (char c : line) {
            if (c == ' ') {
                ++leading_spaces;
            } else {
                break;
            }
        }
        
        // Tags have more indentation (6+ spaces) than test names (2 spaces)
        if (leading_spaces >= 6 && has_test) {
            // This is a tag line
            std::regex tag_regex(R"(\[([^\]]+)\])");
            std::sregex_iterator iter(line.begin(), line.end(), tag_regex);
            std::sregex_iterator end_iter;
            for (; iter != end_iter; ++iter) {
                current_test.tags.push_back((*iter)[1].str());
            }
        } else if (leading_spaces >= 2) {
            // This is a test name - save previous test if any
            if (has_test) {
                info.test_cases.push_back(std::move(current_test));
                current_test = TestCaseInfo{};
            }
            
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) {
                current_test.name = line.substr(start, end - start + 1);
                has_test = true;
            }
        }
    }
    
    // Don't forget the last test
    if (has_test) {
        info.test_cases.push_back(std::move(current_test));
    }
    
    return info;
}

std::vector<TestSuiteInfo> TestRunner::enumerateAllTests() {
    return enumerateAllTestsParallel(1);  // Default to sequential
}

std::vector<TestSuiteInfo> TestRunner::enumerateAllTestsParallel(int num_threads) {
    // Get test executables
    std::vector<std::filesystem::path> tests = test_executables_;
    
    // Add discovered tests if we have a test directory
    if (!test_directory_.empty()) {
        auto discovered = discoverTests();
        for (const auto& t : discovered) {
            if (std::find(tests.begin(), tests.end(), t) == tests.end()) {
                tests.push_back(t);
            }
        }
    }
    
    if (tests.empty()) {
        return {};
    }
    
    // Limit threads to number of tests
    num_threads = std::min(num_threads, static_cast<int>(tests.size()));
    num_threads = std::max(1, num_threads);
    
    // Result storage
    std::vector<TestSuiteInfo> all_tests(tests.size());
    
    if (num_threads == 1) {
        // Sequential enumeration
        for (size_t i = 0; i < tests.size(); ++i) {
            all_tests[i] = enumerateTests(tests[i]);
        }
    } else {
        // Parallel enumeration
        std::atomic<size_t> next_idx{0};
        std::mutex results_mutex;
        std::vector<std::pair<size_t, TestSuiteInfo>> results;
        
        auto worker = [&]() {
            // Create a local runner for thread safety
            TestRunner local_runner;
            local_runner.setHipLibrary(hip_library_);
            
            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= tests.size()) break;
                
                auto info = local_runner.enumerateTests(tests[idx]);
                
                std::lock_guard<std::mutex> lock(results_mutex);
                results.emplace_back(idx, std::move(info));
            }
        };
        
        // Launch threads
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }
        
        // Wait for completion
        for (auto& t : threads) {
            t.join();
        }
        
        // Copy results to output in order
        for (auto& [idx, info] : results) {
            all_tests[idx] = std::move(info);
        }
    }
    
    return all_tests;
}

TestSuiteResult TestRunner::runTest(const std::filesystem::path& executable,
                                     const std::vector<std::string>& test_names) {
    TestSuiteResult result;
    result.name = executable.filename().string();
    result.executable_path = executable.string();
    result.timestamp = getCurrentTimestamp();
    
    // Create temp file for XML output
    auto xml_output_path = createTempFile("hip_cts_");
    
    // Build command line
    std::vector<std::string> args;
    args.push_back(executable.string());
    
    // Add HIP library argument if specified
    if (!hip_library_.empty()) {
        args.push_back("--hip-library");
        args.push_back(hip_library_);
    }
    
    // Add extra arguments
    for (const auto& arg : extra_args_) {
        args.push_back(arg);
    }
    
    // Add specific test names as filter
    if (!test_names.empty()) {
        // Catch2 uses comma-separated test names or patterns
        for (const auto& name : test_names) {
            args.push_back("\"" + name + "\"");
        }
    }
    
    // Tell Catch2 to output JUnit XML
    args.push_back("-r");
    args.push_back("junit");
    args.push_back("-o");
    args.push_back(xml_output_path.string());
    
    // Build command line string for logging
    std::ostringstream cmd_oss;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) cmd_oss << " ";
        cmd_oss << args[i];
    }
    result.command_line = cmd_oss.str();
    
    if (verbose_) {
        std::cerr << "Running: " << result.command_line << std::endl;
    }
    
    // Execute the test
    auto start = std::chrono::steady_clock::now();
    auto exec_result = executeCommand(args);
    auto end = std::chrono::steady_clock::now();
    
    result.exit_code = exec_result.exit_code;
    result.stdout_output = exec_result.stdout_output;
    result.stderr_output = exec_result.stderr_output;
    
    // Read the XML output
    if (std::filesystem::exists(xml_output_path)) {
        std::ifstream xml_file(xml_output_path);
        std::stringstream buffer;
        buffer << xml_file.rdbuf();
        std::string xml_content = buffer.str();
        
        if (!xml_content.empty()) {
            result = parseJUnitXml(xml_content, executable);
            result.exit_code = exec_result.exit_code;
            result.stdout_output = exec_result.stdout_output;
            result.stderr_output = exec_result.stderr_output;
            result.command_line = cmd_oss.str();
        }
        
        // Clean up temp file
        std::filesystem::remove(xml_output_path);
    } else {
        result.execution_failed = true;
        result.execution_error = "Test executable did not produce XML output";
    }
    
    // Use wall clock time if we don't have test time
    if (result.time_seconds == 0.0) {
        result.time_seconds = std::chrono::duration<double>(end - start).count();
    }
    
    return result;
}

AggregatedResults TestRunner::runAllTests() {
    AggregatedResults results;
    results.timestamp = getCurrentTimestamp();
    results.hip_library = hip_library_;
    results.extra_args = extra_args_;
    
    // Get test executables
    std::vector<std::filesystem::path> tests = test_executables_;
    
    // Add discovered tests if we have a test directory
    if (!test_directory_.empty()) {
        auto discovered = discoverTests();
        for (const auto& t : discovered) {
            // Avoid duplicates
            if (std::find(tests.begin(), tests.end(), t) == tests.end()) {
                tests.push_back(t);
            }
        }
    }
    
    // Run each test
    int current = 0;
    int total = static_cast<int>(tests.size());
    
    for (const auto& test : tests) {
        current++;
        
        if (progress_callback_) {
            progress_callback_(test.filename().string(), current, total);
        }
        
        auto suite_result = runTest(test);
        
        // Aggregate counts
        results.total_tests += suite_result.tests;
        results.total_failures += suite_result.failures;
        results.total_errors += suite_result.errors;
        results.total_skipped += suite_result.skipped;
        results.total_time_seconds += suite_result.time_seconds;
        
        results.suites.push_back(std::move(suite_result));
    }
    
    return results;
}

} // namespace runner
} // namespace hip_cts
