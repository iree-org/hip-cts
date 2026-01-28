// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_runner.hpp"
#include "xml_aggregator.hpp"
#include "text_reporter.hpp"
#include "html_reporter.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void printUsage(const char* program_name) {
    std::cerr << "HIP CTS Test Runner v1.0.0\n";
    std::cerr << "\n";
    std::cerr << "Usage: " << program_name << " [OPTIONS] [TEST_EXECUTABLE...]\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                Show this help message\n";
    std::cerr << "  -d, --directory DIR       Directory to discover tests in\n";
    std::cerr << "  --hip-library PATH        Path to HIP library to use for tests\n";
    std::cerr << "  -f, --filter PATTERN      Filter tests by name pattern\n";
    std::cerr << "  -r, --repeat N            Repeat all tests N times (default: 1)\n";
    std::cerr << "  -v, --verbose             Enable verbose output\n";
    std::cerr << "  --show-passed             Show individual passed tests in text output\n";
    std::cerr << "\n";
    std::cerr << "Output Options:\n";
    std::cerr << "  -o, --output-format FMT   Output format: text, xml, html, all (default: text)\n";
    std::cerr << "  --xml-output FILE         Write XML output to FILE\n";
    std::cerr << "  --html-output FILE        Write HTML output to FILE\n";
    std::cerr << "  --text-output FILE        Write text output to FILE\n";
    std::cerr << "  --no-color                Disable colored output\n";
    std::cerr << "\n";
    std::cerr << "Pass-through Arguments:\n";
    std::cerr << "  -- ARGS...                Pass remaining arguments to test executables\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << program_name << " -d build/Debug/tests\n";
    std::cerr << "  " << program_name << " -d build/tests --hip-library /path/to/libhip.so\n";
    std::cerr << "  " << program_name << " build/test1 build/test2 -o html --html-output report.html\n";
    std::cerr << "  " << program_name << " -d tests -o all --xml-output results.xml --html-output results.html\n";
}

struct Options {
    std::string test_directory;
    std::string hip_library;
    std::string filter;
    int repeat = 1;
    bool verbose = false;
    bool show_passed = false;
    bool use_color = true;
    std::string output_format = "text";
    std::string xml_output;
    std::string html_output;
    std::string text_output;
    std::vector<std::string> test_executables;
    std::vector<std::string> extra_args;
};

bool parseArgs(int argc, char* argv[], Options& opts) {
    bool parsing_extra_args = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (parsing_extra_args) {
            opts.extra_args.push_back(arg);
            continue;
        }
        
        if (arg == "--") {
            parsing_extra_args = true;
            continue;
        }
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        
        if (arg == "-d" || arg == "--directory") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.test_directory = argv[++i];
        } else if (arg == "--hip-library") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.hip_library = argv[++i];
        } else if (arg == "-f" || arg == "--filter") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.filter = argv[++i];
        } else if (arg == "-r" || arg == "--repeat") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            try {
                opts.repeat = std::stoi(argv[++i]);
                if (opts.repeat < 1) {
                    std::cerr << "Error: --repeat must be at least 1\n";
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: --repeat requires a valid integer\n";
                return false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--show-passed") {
            opts.show_passed = true;
        } else if (arg == "--no-color") {
            opts.use_color = false;
        } else if (arg == "-o" || arg == "--output-format") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.output_format = argv[++i];
            if (opts.output_format != "text" && opts.output_format != "xml" && 
                opts.output_format != "html" && opts.output_format != "all") {
                std::cerr << "Error: Unknown output format: " << opts.output_format << "\n";
                std::cerr << "Valid formats: text, xml, html, all\n";
                return false;
            }
        } else if (arg == "--xml-output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.xml_output = argv[++i];
        } else if (arg == "--html-output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.html_output = argv[++i];
        } else if (arg == "--text-output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            opts.text_output = argv[++i];
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return false;
        } else {
            // Positional argument - test executable
            opts.test_executables.push_back(arg);
        }
    }
    
    // Validate we have something to run
    if (opts.test_directory.empty() && opts.test_executables.empty()) {
        std::cerr << "Error: No tests specified. Use -d <directory> or provide test executables.\n";
        printUsage(argv[0]);
        return false;
    }
    
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    Options opts;
    if (!parseArgs(argc, argv, opts)) {
        return 1;
    }
    
    // Create and configure the test runner
    hip_cts::runner::TestRunner runner;
    
    if (!opts.test_directory.empty()) {
        runner.setTestDirectory(opts.test_directory);
    }
    
    if (!opts.hip_library.empty()) {
        runner.setHipLibrary(opts.hip_library);
    }
    
    if (!opts.filter.empty()) {
        runner.setFilter(opts.filter);
    }
    
    runner.setVerbose(opts.verbose);
    
    for (const auto& arg : opts.extra_args) {
        runner.addExtraArg(arg);
    }
    
    for (const auto& exe : opts.test_executables) {
        runner.addTestExecutable(exe);
    }
    
    // Set up progress callback for console output
    if (opts.output_format == "text" || opts.output_format == "all") {
        runner.setProgressCallback([&opts](const std::string& name, int current, int total) {
            if (opts.use_color) {
                std::cout << "\033[36m";
            }
            std::cout << "[" << current << "/" << total << "]";
            if (opts.use_color) {
                std::cout << "\033[0m";
            }
            std::cout << " Running: " << name << std::endl;
        });
    }
    
    // Run all tests (with optional repeat)
    hip_cts::runner::AggregatedResults results;
    for (int iteration = 0; iteration < opts.repeat; ++iteration) {
        if (opts.repeat > 1) {
            if (opts.use_color) {
                std::cout << "\033[1;35m";
            }
            std::cout << "\n=== Iteration " << (iteration + 1) << " of " << opts.repeat << " ===\n";
            if (opts.use_color) {
                std::cout << "\033[0m";
            }
        }
        
        results = runner.runAllTests();
        
        // Stop early if there are failures
        if (results.total_failures > 0 || results.total_errors > 0) {
            if (opts.repeat > 1) {
                std::cerr << "Stopping after iteration " << (iteration + 1) 
                          << " due to test failures.\n";
            }
            break;
        }
    }
    
    // Generate outputs
    bool output_to_console = true;
    
    // XML output
    if (opts.output_format == "xml" || opts.output_format == "all" || !opts.xml_output.empty()) {
        if (!opts.xml_output.empty()) {
            if (hip_cts::runner::XmlAggregator::writeToFile(results, opts.xml_output)) {
                if (opts.verbose) {
                    std::cerr << "XML output written to: " << opts.xml_output << "\n";
                }
            } else {
                std::cerr << "Error: Failed to write XML to: " << opts.xml_output << "\n";
            }
        } else if (opts.output_format == "xml") {
            hip_cts::runner::XmlAggregator::write(results, std::cout);
            output_to_console = false;
        }
    }
    
    // HTML output
    if (opts.output_format == "html" || opts.output_format == "all" || !opts.html_output.empty()) {
        hip_cts::runner::HtmlReporter::Options html_opts;
        
        if (!opts.html_output.empty()) {
            hip_cts::runner::HtmlReporter reporter(html_opts);
            if (reporter.writeToFile(results, opts.html_output)) {
                if (opts.verbose) {
                    std::cerr << "HTML output written to: " << opts.html_output << "\n";
                }
            } else {
                std::cerr << "Error: Failed to write HTML to: " << opts.html_output << "\n";
            }
        } else if (opts.output_format == "html") {
            hip_cts::runner::HtmlReporter reporter(html_opts);
            reporter.write(results, std::cout);
            output_to_console = false;
        }
    }
    
    // Text output
    if (opts.output_format == "text" || opts.output_format == "all" || !opts.text_output.empty()) {
        hip_cts::runner::TextReporter::Options text_opts;
        text_opts.use_color = opts.use_color;
        text_opts.show_passed = opts.show_passed;
        text_opts.verbose = opts.verbose;
        
        if (!opts.text_output.empty()) {
            hip_cts::runner::TextReporter reporter(text_opts);
            if (reporter.writeToFile(results, opts.text_output)) {
                if (opts.verbose) {
                    std::cerr << "Text output written to: " << opts.text_output << "\n";
                }
            } else {
                std::cerr << "Error: Failed to write text to: " << opts.text_output << "\n";
            }
        } else if (output_to_console) {
            hip_cts::runner::TextReporter reporter(text_opts);
            reporter.write(results, std::cout);
        }
    }
    
    // Return non-zero exit code if there were failures
    return (results.total_failures > 0 || results.total_errors > 0) ? 1 : 0;
}
