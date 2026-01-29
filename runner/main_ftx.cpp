// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test_runner.hpp"
#include "xml_aggregator.hpp"
#include "text_reporter.hpp"
#include "html_reporter.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace ftxui;

namespace {

//=============================================================================
// Application State
//=============================================================================

enum class AppScreen {
    MainMenu = 0,
    TestSelection = 1,
    Settings = 2,
    Running = 3,
    Results = 4,
    Export = 5
};

// Individual test case selection
struct TestCaseSelection {
    std::string name;
    std::vector<std::string> tags;
    bool selected = true;
};

// Suite with expandable test cases
struct SuiteSelection {
    std::string name;
    std::filesystem::path executable_path;
    bool expanded = false;
    bool selected = true;  // Master toggle for the suite
    std::vector<TestCaseSelection> test_cases;
    
    // Count selected tests
    int selectedCount() const {
        int count = 0;
        for (const auto& tc : test_cases) {
            if (tc.selected) ++count;
        }
        return count;
    }
};

struct AppState {
    int current_screen = 0;
    
    // Test discovery - full enumeration with individual tests
    std::vector<SuiteSelection> suites;
    
    // Run state
    std::atomic<bool> running{false};
    std::atomic<bool> cancel_requested{false};
    std::atomic<int> current_suite{0};
    std::atomic<int> total_suites{0};
    std::atomic<int> current_test_in_suite{0};
    std::atomic<int> total_tests_in_suite{0};
    std::string current_suite_name;
    std::string current_test_name;
    std::mutex state_mutex;
    
    // Results
    hip_cts::runner::AggregatedResults results;
    bool has_results = false;
    
    // Settings (modifiable)
    std::string hip_library;
    std::string repeat_str = "1";
    int repeat = 1;
    std::string parallel_str = "1";
    int parallel = 1;  // Number of tests to run in parallel
    std::string test_directory;
    
    // Parallel execution state
    std::atomic<int> tests_completed{0};
    
    // Mutex for thread-safe screen updates (PostEvent is not fully thread-safe)
    std::mutex post_event_mutex;
    std::atomic<int> total_tests_to_run{0};
    
    // Per-suite progress tracking
    struct SuiteProgress {
        std::string name;
        std::atomic<int> completed{0};
        int total{0};
    };
    std::vector<std::unique_ptr<SuiteProgress>> suite_progress;
    std::mutex suite_progress_mutex;
    
    // UI state
    int menu_selected = 0;
    int tree_cursor = 0;
    bool tree_focused = true;  // True when tree has focus, false when buttons have focus
    
    // Results tree state
    int results_cursor = 0;
    std::set<std::string> results_expanded;  // Set of expanded suite names
    bool results_tree_focused = true;
    
    // Individual test case results for flat view
    struct IndividualTestResult {
        std::string suite_name;
        std::string test_name;
        bool passed = true;
        bool skipped = false;
        double time_seconds = 0.0;
        std::string failure_message;
    };
    std::vector<IndividualTestResult> individual_results;
    std::mutex individual_results_mutex;
    
    // Runner
    hip_cts::runner::TestRunner runner;
};

// Export filenames (static to persist across renders)
static std::string g_xml_filename = "results.xml";
static std::string g_text_filename = "results.txt";
static std::string g_html_filename = "results.html";
static std::string g_export_message = "";

//=============================================================================
// Command-line parsing
//=============================================================================

void printUsage(const char* program_name) {
    std::cerr << "HIP CTS Interactive Test Runner v1.0.0\n";
    std::cerr << "\n";
    std::cerr << "Usage: " << program_name << " [OPTIONS]\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help                Show this help message\n";
    std::cerr << "  -d, --directory DIR       Directory to discover tests in\n";
    std::cerr << "  --hip-library PATH        Path to HIP library to use for tests\n";
    std::cerr << "  -r, --repeat N            Repeat all tests N times (default: 1)\n";
    std::cerr << "  -j, --parallel N          Run N tests in parallel (default: 1)\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << program_name << " -d build/Debug/tests\n";
    std::cerr << "  " << program_name << " -d build/tests --hip-library /path/to/libhip.so\n";
    std::cerr << "  " << program_name << " -d build/tests -j 4\n";
}

bool parseArgs(int argc, char* argv[], AppState& state) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        
        if (arg == "-d" || arg == "--directory") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            state.test_directory = argv[++i];
        } else if (arg == "--hip-library") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            state.hip_library = argv[++i];
        } else if (arg == "-r" || arg == "--repeat") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            state.repeat_str = argv[++i];
            try {
                state.repeat = std::stoi(state.repeat_str);
            } catch (...) {
                state.repeat = 1;
            }
        } else if (arg == "-j" || arg == "--parallel") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument\n";
                return false;
            }
            state.parallel_str = argv[++i];
            try {
                state.parallel = std::stoi(state.parallel_str);
                if (state.parallel < 1) state.parallel = 1;
            } catch (...) {
                state.parallel = 1;
            }
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return false;
        }
    }
    
    return true;
}

//=============================================================================
// Test Running
//=============================================================================

// Structure to hold a test job (defined at namespace scope for thread safety)
struct TestJobInfo {
    std::filesystem::path executable_path;
    std::string suite_name;
    std::string test_name;
    size_t suite_progress_idx;
};

// Shared data for parallel workers
struct WorkerSharedData {
    std::vector<TestJobInfo>* jobs;
    std::atomic<size_t> next_job{0};
    std::mutex results_mutex;
    AppState* state;
    ScreenInteractive* screen;
};

void runTestsThread(AppState& state, ScreenInteractive& screen) {
    state.running = true;
    state.cancel_requested = false;
    
    // Initialize per-suite progress tracking
    {
        std::lock_guard<std::mutex> lock(state.suite_progress_mutex);
        state.suite_progress.clear();
        for (auto& suite : state.suites) {
            int selected_count = 0;
            for (const auto& tc : suite.test_cases) {
                if (tc.selected) ++selected_count;
            }
            if (selected_count > 0) {
                auto sp = std::make_unique<AppState::SuiteProgress>();
                sp->name = suite.name;
                sp->completed = 0;
                sp->total = selected_count * state.repeat;  // Account for repeats
                state.suite_progress.push_back(std::move(sp));
            }
        }
    }
    
    // Build flat list of all test jobs
    std::vector<TestJobInfo> all_jobs;
    for (size_t si = 0; si < state.suites.size(); ++si) {
        auto& suite = state.suites[si];
        
        // Find matching suite_progress index
        size_t sp_idx = 0;
        {
            std::lock_guard<std::mutex> lock(state.suite_progress_mutex);
            for (size_t i = 0; i < state.suite_progress.size(); ++i) {
                if (state.suite_progress[i]->name == suite.name) {
                    sp_idx = i;
                    break;
                }
            }
        }
        
        for (const auto& tc : suite.test_cases) {
            if (tc.selected) {
                TestJobInfo job;
                job.executable_path = suite.executable_path;
                job.suite_name = suite.name;
                job.test_name = tc.name;
                job.suite_progress_idx = sp_idx;
                all_jobs.push_back(std::move(job));
            }
        }
    }
    
    // Multiply by repeat count
    if (state.repeat > 1) {
        std::vector<TestJobInfo> repeated_jobs;
        repeated_jobs.reserve(all_jobs.size() * state.repeat);
        for (int r = 0; r < state.repeat; ++r) {
            for (const auto& job : all_jobs) {
                repeated_jobs.push_back(job);
            }
        }
        all_jobs = std::move(repeated_jobs);
    }
    
    state.total_tests_to_run = static_cast<int>(all_jobs.size());
    state.tests_completed = 0;
    state.total_suites = static_cast<int>(state.suites.size());
    
    // Reset results
    state.results = hip_cts::runner::AggregatedResults{};
    state.results.hip_library = state.hip_library;
    {
        std::lock_guard<std::mutex> lock(state.individual_results_mutex);
        state.individual_results.clear();
    }
    state.results_cursor = 0;
    state.results_expanded.clear();
    
    // If no jobs, just finish
    if (all_jobs.empty()) {
        state.has_results = true;
        state.running = false;
        state.current_screen = static_cast<int>(AppScreen::Results);
        {
            std::lock_guard<std::mutex> lock(state.post_event_mutex);
            screen.PostEvent(Event::Custom);
        }
        return;
    }
    
    // Shared data for workers
    WorkerSharedData shared;
    shared.jobs = &all_jobs;
    shared.next_job = 0;
    shared.state = &state;
    shared.screen = &screen;
    
    // Worker function - takes pointer to shared data
    auto workerFn = [](WorkerSharedData* data) {
        AppState& state = *data->state;
        ScreenInteractive& screen = *data->screen;
        std::vector<TestJobInfo>& jobs = *data->jobs;
        
        // Each worker needs its own runner instance for thread safety
        hip_cts::runner::TestRunner runner;
        if (!state.test_directory.empty()) {
            runner.setTestDirectory(state.test_directory);
        }
        if (!state.hip_library.empty()) {
            runner.setHipLibrary(state.hip_library);
        }
        
        while (!state.cancel_requested.load()) {
            // Get next job
            size_t job_idx = data->next_job.fetch_add(1);
            if (job_idx >= jobs.size()) {
                break;
            }
            
            const TestJobInfo& job = jobs[job_idx];
            
            // Update current test info (no screen update here to reduce contention)
            {
                std::lock_guard<std::mutex> lock(state.state_mutex);
                state.current_suite_name = job.suite_name;
                state.current_test_name = job.test_name;
            }
            
            // Run the test
            auto suite_result = runner.runTest(job.executable_path, {job.test_name});
            
            // Update per-suite progress
            {
                std::lock_guard<std::mutex> lock(state.suite_progress_mutex);
                if (job.suite_progress_idx < state.suite_progress.size() &&
                    state.suite_progress[job.suite_progress_idx]) {
                    state.suite_progress[job.suite_progress_idx]->completed++;
                }
            }
            
            // Aggregate results (thread-safe)
            {
                std::lock_guard<std::mutex> lock(data->results_mutex);
                
                state.results.total_tests += suite_result.tests;
                state.results.total_failures += suite_result.failures;
                state.results.total_errors += suite_result.errors;
                state.results.total_skipped += suite_result.skipped;
                state.results.total_time_seconds += suite_result.time_seconds;
                
                // Store individual test case results
                {
                    std::lock_guard<std::mutex> ilock(state.individual_results_mutex);
                    for (const auto& tc : suite_result.test_cases) {
                        AppState::IndividualTestResult itr;
                        itr.suite_name = job.suite_name;
                        itr.test_name = tc.name;
                        itr.passed = tc.passed;
                        itr.skipped = tc.skipped;
                        itr.time_seconds = tc.time_seconds;
                        itr.failure_message = tc.failure_message;
                        state.individual_results.push_back(std::move(itr));
                    }
                }
                
                // Find or create suite in results
                bool found = false;
                for (auto& existing : state.results.suites) {
                    if (existing.executable_path == suite_result.executable_path) {
                        existing.tests += suite_result.tests;
                        existing.failures += suite_result.failures;
                        existing.errors += suite_result.errors;
                        existing.skipped += suite_result.skipped;
                        existing.time_seconds += suite_result.time_seconds;
                        for (auto& tc : suite_result.test_cases) {
                            existing.test_cases.push_back(std::move(tc));
                        }
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    state.results.suites.push_back(std::move(suite_result));
                }
            }
            
            // Update progress
            state.tests_completed++;
            {
                std::lock_guard<std::mutex> lock(state.post_event_mutex);
                screen.PostEvent(Event::Custom);
            }
        }
    };
    
    // Launch worker threads
    int num_workers = std::min(state.parallel, static_cast<int>(all_jobs.size()));
    num_workers = std::max(1, num_workers);
    
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(workerFn, &shared);
    }
    
    // Wait for all workers to complete
    for (auto& w : workers) {
        w.join();
    }
    
    state.has_results = true;
    state.running = false;
    
    // Auto-transition to results
    state.current_screen = static_cast<int>(AppScreen::Results);
    {
        std::lock_guard<std::mutex> lock(state.post_event_mutex);
        screen.PostEvent(Event::Custom);
    }
}

//=============================================================================
// Tree View Component for Test Selection
//=============================================================================

Component TreeViewComponent(AppState& state) {
    // Build a flat list of items for navigation
    // Each item is either a suite header or a test case
    
    return Renderer([&state] {
        std::vector<Element> items;
        int index = 0;
        
        for (size_t si = 0; si < state.suites.size(); ++si) {
            auto& suite = state.suites[si];
            
            // Suite header
            std::string expand_icon = suite.expanded ? "▼ " : "▶ ";
            std::string check_icon;
            int sel_count = suite.selectedCount();
            if (sel_count == 0) {
                check_icon = "☐ ";
            } else if (sel_count == static_cast<int>(suite.test_cases.size())) {
                check_icon = "☑ ";
            } else {
                check_icon = "▣ ";  // Partial selection
            }
            
            std::string suite_text = expand_icon + check_icon + suite.name + 
                                     " (" + std::to_string(sel_count) + "/" + 
                                     std::to_string(suite.test_cases.size()) + ")";
            
            Element suite_elem = text(suite_text);
            bool is_selected = (index == state.tree_cursor);
            if (is_selected) {
                suite_elem = suite_elem | inverted | focus;
            }
            suite_elem = suite_elem | bold;
            items.push_back(suite_elem);
            ++index;
            
            // Test cases if expanded
            if (suite.expanded) {
                for (size_t ti = 0; ti < suite.test_cases.size(); ++ti) {
                    auto& tc = suite.test_cases[ti];
                    
                    std::string tc_check = tc.selected ? "☑ " : "☐ ";
                    std::string tc_text = "    " + tc_check + tc.name;
                    
                    // Add tags
                    if (!tc.tags.empty()) {
                        tc_text += " ";
                        for (const auto& tag : tc.tags) {
                            tc_text += "[" + tag + "]";
                        }
                    }
                    
                    Element tc_elem = text(tc_text);
                    bool is_tc_selected = (index == state.tree_cursor);
                    if (is_tc_selected) {
                        tc_elem = tc_elem | inverted | focus;
                    }
                    if (!tc.selected) {
                        tc_elem = tc_elem | dim;
                    }
                    items.push_back(tc_elem);
                    ++index;
                }
            }
        }
        
        return vbox(items) | vscroll_indicator | yframe | flex;
    });
}

// Helper to get the item at cursor position
struct TreeItem {
    int suite_index = -1;
    int test_index = -1;  // -1 means it's a suite header
};

TreeItem getTreeItemAtCursor(const AppState& state) {
    TreeItem item;
    int index = 0;
    
    for (size_t si = 0; si < state.suites.size(); ++si) {
        const auto& suite = state.suites[si];
        
        if (index == state.tree_cursor) {
            item.suite_index = static_cast<int>(si);
            item.test_index = -1;
            return item;
        }
        ++index;
        
        if (suite.expanded) {
            for (size_t ti = 0; ti < suite.test_cases.size(); ++ti) {
                if (index == state.tree_cursor) {
                    item.suite_index = static_cast<int>(si);
                    item.test_index = static_cast<int>(ti);
                    return item;
                }
                ++index;
            }
        }
    }
    
    return item;
}

int getTreeItemCount(const AppState& state) {
    int count = 0;
    for (const auto& suite : state.suites) {
        ++count;  // Suite header
        if (suite.expanded) {
            count += static_cast<int>(suite.test_cases.size());
        }
    }
    return count;
}

//=============================================================================
// Main Application
//=============================================================================

Component App(AppState& state, ScreenInteractive& screen, std::thread& test_thread) {
    // Menu entries
    static std::vector<std::string> menu_entries = {
        " Run All Tests",
        " Run Selected Tests",
        " Select Tests",
        " Settings",
        " View Results",
        " Export Results",
        " Exit"
    };
    
    //=========================================================================
    // Main Menu Component
    //=========================================================================
    auto main_menu = Menu(&menu_entries, &state.menu_selected);
    
    //=========================================================================
    // Test Selection Component (Tree View)
    //=========================================================================
    auto tree_view = TreeViewComponent(state);
    
    auto select_all_btn = Button("Select All", [&state] {
        for (auto& suite : state.suites) {
            suite.selected = true;
            for (auto& tc : suite.test_cases) {
                tc.selected = true;
            }
        }
    });
    
    auto deselect_all_btn = Button("Deselect All", [&state] {
        for (auto& suite : state.suites) {
            suite.selected = false;
            for (auto& tc : suite.test_cases) {
                tc.selected = false;
            }
        }
    });
    
    auto expand_all_btn = Button("Expand All", [&state] {
        for (auto& suite : state.suites) {
            suite.expanded = true;
        }
    });
    
    auto collapse_all_btn = Button("Collapse All", [&state] {
        for (auto& suite : state.suites) {
            suite.expanded = false;
        }
        state.tree_cursor = 0;
    });
    
    auto back_from_selection_btn = Button("Back", [&state] {
        state.current_screen = static_cast<int>(AppScreen::MainMenu);
    });
    
    auto selection_buttons = Container::Horizontal({
        select_all_btn,
        deselect_all_btn,
        expand_all_btn,
        collapse_all_btn,
        back_from_selection_btn,
    });
    
    auto test_selection_inner = Container::Vertical({
        tree_view,
        selection_buttons,
    });
    
    // Wrap to add visual focus indicators
    auto test_selection = Renderer(test_selection_inner, [&state, tree_view, selection_buttons] {
        // Tree view with focus indicator
        Element tree_elem = tree_view->Render();
        if (state.tree_focused) {
            tree_elem = tree_elem | borderStyled(ROUNDED, Color::Cyan);
        } else {
            tree_elem = tree_elem | borderStyled(ROUNDED, Color::GrayDark);
        }
        
        // Buttons with focus indicator
        Element buttons_elem = selection_buttons->Render() | center;
        if (!state.tree_focused) {
            buttons_elem = buttons_elem | bold;
        } else {
            buttons_elem = buttons_elem | dim;
        }
        
        return vbox({
            tree_elem | flex,
            separator(),
            buttons_elem,
        });
    });
    
    //=========================================================================
    // Shared input option for all input fields (single-line, no Enter newline)
    //=========================================================================
    InputOption input_option;
    input_option.multiline = false;
    
    //=========================================================================
    // Settings Component
    //=========================================================================
    
    auto hip_lib_input = Input(&state.hip_library, "path/to/libhip.so", input_option);
    auto repeat_input = Input(&state.repeat_str, "1", input_option);
    auto parallel_input = Input(&state.parallel_str, "1", input_option);
    auto test_dir_input = Input(&state.test_directory, "path/to/tests", input_option);
    
    auto apply_settings_btn = Button("Apply & Rediscover", [&state, &screen] {
        // Parse repeat
        try {
            state.repeat = std::stoi(state.repeat_str);
            if (state.repeat < 1) state.repeat = 1;
        } catch (...) {
            state.repeat = 1;
            state.repeat_str = "1";
        }
        
        // Parse parallel
        try {
            state.parallel = std::stoi(state.parallel_str);
            if (state.parallel < 1) state.parallel = 1;
        } catch (...) {
            state.parallel = 1;
            state.parallel_str = "1";
        }
        
        // Update runner settings
        state.runner = hip_cts::runner::TestRunner{};
        if (!state.test_directory.empty()) {
            state.runner.setTestDirectory(state.test_directory);
        }
        if (!state.hip_library.empty()) {
            state.runner.setHipLibrary(state.hip_library);
        }
        
        // Re-discover tests (use parallel discovery)
        state.suites.clear();
        auto all_tests = state.runner.enumerateAllTestsParallel(state.parallel > 0 ? state.parallel : 8);
        for (const auto& suite : all_tests) {
            SuiteSelection sel;
            sel.name = suite.name;
            sel.executable_path = suite.executable_path;
            sel.expanded = false;
            sel.selected = true;
            for (const auto& tc : suite.test_cases) {
                TestCaseSelection tcs;
                tcs.name = tc.name;
                tcs.tags = tc.tags;
                tcs.selected = true;
                sel.test_cases.push_back(std::move(tcs));
            }
            state.suites.push_back(std::move(sel));
        }
        
        state.current_screen = static_cast<int>(AppScreen::MainMenu);
    });
    
    auto back_from_settings_btn = Button("Back", [&state] {
        state.current_screen = static_cast<int>(AppScreen::MainMenu);
    });
    
    auto settings_container = Container::Vertical({
        hip_lib_input,
        repeat_input,
        parallel_input,
        test_dir_input,
        Container::Horizontal({apply_settings_btn, back_from_settings_btn}),
    });
    
    auto settings_comp = Renderer(settings_container, [&state, settings_container] {
        return vbox({
            text("Settings") | bold | center,
            separator(),
            text(""),
            hbox({text("HIP Library:     "), settings_container->ChildAt(0)->Render() | flex}),
            text("  Leave empty for default (libamdhip64.so)") | dim,
            text(""),
            hbox({text("Repeat Count:    "), settings_container->ChildAt(1)->Render() | size(WIDTH, EQUAL, 10)}),
            text("  Number of times to run all tests") | dim,
            text(""),
            hbox({text("Parallel Jobs:   "), settings_container->ChildAt(2)->Render() | size(WIDTH, EQUAL, 10)}),
            text("  Number of tests to run concurrently") | dim,
            text(""),
            hbox({text("Test Directory:  "), settings_container->ChildAt(3)->Render() | flex}),
            text("  Directory containing test executables") | dim,
            text(""),
            filler(),
            separator(),
            settings_container->ChildAt(4)->Render() | center,
        }) | border;
    });
    
    //=========================================================================
    // Running Component
    //=========================================================================
    auto running_comp = Renderer([&state] {
        int completed = state.tests_completed.load();
        int total = state.total_tests_to_run.load();
        
        float progress = total > 0 ? static_cast<float>(completed) / total : 0.0f;
        
        std::vector<Element> elements;
        
        elements.push_back(text("Running Tests...") | bold | center);
        
        // Show settings info
        std::string info = "";
        if (state.parallel > 1) {
            info += "parallel=" + std::to_string(state.parallel);
        }
        if (state.repeat > 1) {
            if (!info.empty()) info += ", ";
            info += "repeat=" + std::to_string(state.repeat) + "x";
        }
        if (!info.empty()) {
            elements.push_back(text("(" + info + ")") | dim | center);
        }
        elements.push_back(separator());
        
        // Overall progress
        elements.push_back(text(""));
        elements.push_back(hbox({
            text("Total: ") | bold,
            gauge(progress) | flex,
            text(" " + std::to_string(completed) + "/" + std::to_string(total)),
        }));
        
        elements.push_back(text(""));
        elements.push_back(separator());
        
        // Per-suite progress bars - copy data under lock to avoid holding lock during render
        std::vector<std::tuple<std::string, int, int>> suite_data;
        {
            std::lock_guard<std::mutex> lock(state.suite_progress_mutex);
            for (const auto& sp : state.suite_progress) {
                if (sp) {
                    suite_data.emplace_back(sp->name, sp->completed.load(), sp->total);
                }
            }
        }
        
        if (suite_data.empty()) {
            elements.push_back(text("Initializing...") | dim | center);
        } else {
            // Separate completed and in-progress suites
            int completed_count = 0;
            std::vector<std::tuple<std::string, int, int>> in_progress;
            
            for (const auto& [name, suite_completed, suite_total] : suite_data) {
                if (suite_completed == suite_total && suite_total > 0) {
                    completed_count++;
                } else {
                    in_progress.push_back({name, suite_completed, suite_total});
                }
            }
            
            // Show summary of completed suites if any
            if (completed_count > 0) {
                elements.push_back(hbox({
                    text("✓ " + std::to_string(completed_count) + " suite(s) completed") 
                        | color(Color::Green) | dim,
                }));
                elements.push_back(text(""));
            }
            
            // Show in-progress suites
            if (in_progress.empty() && completed_count > 0) {
                elements.push_back(text("All suites completed!") | center | color(Color::Green) | bold);
            } else {
                for (const auto& [name, suite_completed, suite_total] : in_progress) {
                    float suite_progress = suite_total > 0 ? 
                        static_cast<float>(suite_completed) / suite_total : 0.0f;
                    
                    // Determine color based on progress
                    Color bar_color = Color::Blue;
                    if (suite_completed > 0) {
                        bar_color = Color::Yellow;
                    }
                    
                    // Truncate suite name if too long
                    std::string suite_name = name;
                    if (suite_name.length() > 25) {
                        suite_name = suite_name.substr(0, 22) + "...";
                    }
                    
                    elements.push_back(hbox({
                        text(suite_name) | size(WIDTH, EQUAL, 26),
                        text(" "),
                        gauge(suite_progress) | flex | color(bar_color),
                        text(" " + std::to_string(suite_completed) + "/" + std::to_string(suite_total)) 
                            | size(WIDTH, EQUAL, 10),
                    }));
                }
            }
        }
        
        elements.push_back(filler());
        
        if (!state.running.load() && state.has_results) {
            elements.push_back(separator());
            elements.push_back(text("Tests complete!") | center | color(Color::Green) | bold);
            elements.push_back(text("Press Esc to return to menu") | center | dim);
        } else {
            elements.push_back(separator());
            elements.push_back(text("Press 'c' to cancel...") | center | dim);
        }
        
        return vbox(elements) | border;
    });
    
    //=========================================================================
    // Results Component - Tree View with Scrolling
    //=========================================================================
    auto results_back_btn = Button("Back to Menu", [&state] {
        state.current_screen = static_cast<int>(AppScreen::MainMenu);
    });
    
    auto results_export_btn = Button("Export Results", [&state] {
        state.current_screen = static_cast<int>(AppScreen::Export);
    });
    
    auto results_expand_all_btn = Button("Expand All", [&state] {
        std::lock_guard<std::mutex> lock(state.individual_results_mutex);
        for (const auto& r : state.individual_results) {
            state.results_expanded.insert(r.suite_name);
        }
    });
    
    auto results_collapse_all_btn = Button("Collapse All", [&state] {
        state.results_expanded.clear();
    });
    
    auto results_buttons = Container::Horizontal({
        results_back_btn, results_export_btn, results_expand_all_btn, results_collapse_all_btn
    });
    
    // Create a component that handles keyboard navigation
    auto results_tree_comp = Renderer([&state] {
        // Copy individual results under lock
        std::vector<AppState::IndividualTestResult> results_copy;
        {
            std::lock_guard<std::mutex> lock(state.individual_results_mutex);
            results_copy = state.individual_results;
        }
        
        if (!state.has_results || results_copy.empty()) {
            return text("No results to display") | dim | center;
        }
        
        // Group results by suite
        std::map<std::string, std::vector<const AppState::IndividualTestResult*>> grouped;
        for (const auto& r : results_copy) {
            grouped[r.suite_name].push_back(&r);
        }
        
        // Build the tree view with navigation
        std::vector<Element> tree_elements;
        int visible_idx = 0;
        
        for (const auto& [suite_name, tests] : grouped) {
            bool is_expanded = state.results_expanded.count(suite_name) > 0;
            
            // Count passed/failed for this suite
            int passed = 0, failed = 0, skipped = 0;
            for (const auto* t : tests) {
                if (t->skipped) skipped++;
                else if (t->passed) passed++;
                else failed++;
            }
            
            // Suite row
            Color suite_color = (failed == 0) ? Color::Green : Color::Red;
            std::string status_icon = (failed == 0) ? "✓" : "✗";
            std::string expand_icon = is_expanded ? "▼" : "▶";
            
            bool is_selected = (visible_idx == state.results_cursor);
            auto suite_row = hbox({
                text(expand_icon) | dim,
                text(" "),
                text(status_icon) | color(suite_color),
                text(" "),
                text(suite_name) | bold,
                text(" (" + std::to_string(passed) + "/" + std::to_string(static_cast<int>(tests.size())) + " passed"),
                skipped > 0 ? text(", " + std::to_string(skipped) + " skipped") : text(""),
                text(")"),
            });
            
            if (is_selected && state.results_tree_focused) {
                suite_row = suite_row | inverted | focus;
            }
            tree_elements.push_back(suite_row);
            visible_idx++;
            
            // Show test cases if expanded
            if (is_expanded) {
                for (const auto* tc : tests) {
                    Color tc_color;
                    std::string tc_icon;
                    if (tc->skipped) {
                        tc_color = Color::Yellow;
                        tc_icon = "○";
                    } else if (tc->passed) {
                        tc_color = Color::Green;
                        tc_icon = "✓";
                    } else {
                        tc_color = Color::Red;
                        tc_icon = "✗";
                    }
                    
                    bool tc_selected = (visible_idx == state.results_cursor);
                    
                    std::vector<Element> row_elements = {
                        text("    "),
                        text(tc_icon) | color(tc_color),
                        text(" "),
                        text(tc->test_name),
                    };
                    
                    if (tc->time_seconds > 0) {
                        char time_buf[32];
                        snprintf(time_buf, sizeof(time_buf), " (%.3fs)", tc->time_seconds);
                        row_elements.push_back(text(time_buf) | dim);
                    }
                    
                    auto tc_row = hbox(row_elements);
                    
                    if (tc_selected && state.results_tree_focused) {
                        tc_row = tc_row | inverted | focus;
                    }
                    tree_elements.push_back(tc_row);
                    visible_idx++;
                }
            }
        }
        
        return vbox(tree_elements) | frame | vscroll_indicator | flex;
    });
    
    auto results_comp = CatchEvent(
        Container::Vertical({results_tree_comp, results_buttons}),
        [&state](Event event) {
            if (!state.has_results) return false;
            
            // Tab to switch focus between tree and buttons
            if (event == Event::Tab) {
                state.results_tree_focused = !state.results_tree_focused;
                return true;
            }
            
            if (!state.results_tree_focused) return false;
            
            // Get grouped suite names
            std::vector<std::string> suite_names;
            std::map<std::string, int> suite_test_counts;
            {
                std::lock_guard<std::mutex> lock(state.individual_results_mutex);
                for (const auto& r : state.individual_results) {
                    if (suite_test_counts.find(r.suite_name) == suite_test_counts.end()) {
                        suite_names.push_back(r.suite_name);
                        suite_test_counts[r.suite_name] = 0;
                    }
                    suite_test_counts[r.suite_name]++;
                }
            }
            
            // Calculate total visible items
            int total_visible = 0;
            for (const auto& suite_name : suite_names) {
                total_visible++;  // Suite itself
                if (state.results_expanded.count(suite_name) > 0) {
                    total_visible += suite_test_counts[suite_name];
                }
            }
            
            if (event == Event::ArrowUp || event == Event::Character("k")) {
                if (state.results_cursor > 0) {
                    state.results_cursor--;
                }
                return true;
            }
            
            if (event == Event::ArrowDown || event == Event::Character("j")) {
                if (state.results_cursor < total_visible - 1) {
                    state.results_cursor++;
                }
                return true;
            }
            
            // Enter or Space to expand/collapse
            if (event == Event::Return || event == Event::Character(" ")) {
                // Find which item is selected
                int visible_idx = 0;
                for (const auto& suite_name : suite_names) {
                    if (visible_idx == state.results_cursor) {
                        // Toggle this suite
                        if (state.results_expanded.count(suite_name) > 0) {
                            state.results_expanded.erase(suite_name);
                        } else {
                            state.results_expanded.insert(suite_name);
                        }
                        return true;
                    }
                    visible_idx++;
                    if (state.results_expanded.count(suite_name) > 0) {
                        visible_idx += suite_test_counts[suite_name];
                    }
                }
                return true;
            }
            
            return false;
        }
    );
    
    // Wrap with a renderer to add the header and frame
    results_comp = Renderer(results_comp, [&state, results_comp, results_buttons] {
        if (!state.has_results) {
            return vbox({
                text("No results available") | center,
                text(""),
                text("Run some tests first!") | dim | center,
            }) | border;
        }
        
        auto& results = state.results;
        
        std::vector<Element> elements;
        
        // Summary header
        elements.push_back(text("Test Results Summary") | bold | center);
        elements.push_back(separator());
        
        int passed = results.total_tests - results.total_failures - 
                     results.total_errors - results.total_skipped;
        
        // Stats
        elements.push_back(hbox({
            text("Total: ") | bold, text(std::to_string(results.total_tests)),
            text("  "),
            text("Passed: ") | bold | color(Color::Green), text(std::to_string(passed)),
            text("  "),
            text("Failed: ") | bold | color(Color::Red), text(std::to_string(results.total_failures)),
            text("  "),
            text("Skipped: ") | bold | color(Color::Yellow), text(std::to_string(results.total_skipped)),
        }));
        
        elements.push_back(text("Time: " + std::to_string(results.total_time_seconds) + "s") | dim);
        elements.push_back(text(""));
        
        // Overall result
        bool all_passed = (results.total_failures == 0 && results.total_errors == 0);
        if (all_passed) {
            elements.push_back(text("✓ All tests passed!") | color(Color::Green) | bold | center);
        } else {
            elements.push_back(text("✗ Some tests failed") | color(Color::Red) | bold | center);
        }
        
        elements.push_back(text(""));
        elements.push_back(separator());
        
        // Instructions
        auto tree_border_color = state.results_tree_focused ? Color::Cyan : Color::GrayDark;
        elements.push_back(hbox({
            text("↑↓") | bold, text(" Navigate  "),
            text("Enter/Space") | bold, text(" Expand/Collapse  "),
            text("Tab") | bold, text(" Switch focus"),
        }) | dim);
        
        // Tree view
        elements.push_back(results_comp->Render() | borderStyled(ROUNDED, tree_border_color) | flex);
        
        // Buttons with focus indicator
        auto btn_render = results_buttons->Render();
        if (!state.results_tree_focused) {
            btn_render = btn_render | bold;
        } else {
            btn_render = btn_render | dim;
        }
        elements.push_back(btn_render | center);
        
        return vbox(elements) | border;
    });
    
    //=========================================================================
    // Export Component
    //=========================================================================
    auto xml_input = Input(&g_xml_filename, "results.xml", input_option);
    auto text_input = Input(&g_text_filename, "results.txt", input_option);
    auto html_input = Input(&g_html_filename, "results.html", input_option);
    
    auto export_xml_btn = Button("Save", [&state] {
        if (hip_cts::runner::XmlAggregator::writeToFile(state.results, g_xml_filename)) {
            g_export_message = "Exported to " + g_xml_filename;
        } else {
            g_export_message = "Failed to export XML";
        }
    });
    
    auto export_text_btn = Button("Save", [&state] {
        hip_cts::runner::TextReporter::Options opts;
        opts.use_color = false;
        hip_cts::runner::TextReporter reporter(opts);
        if (reporter.writeToFile(state.results, g_text_filename)) {
            g_export_message = "Exported to " + g_text_filename;
        } else {
            g_export_message = "Failed to export text";
        }
    });
    
    auto export_html_btn = Button("Save", [&state] {
        hip_cts::runner::HtmlReporter::Options opts;
        hip_cts::runner::HtmlReporter reporter(opts);
        if (reporter.writeToFile(state.results, g_html_filename)) {
            g_export_message = "Exported to " + g_html_filename;
        } else {
            g_export_message = "Failed to export HTML";
        }
    });
    
    auto back_from_export_btn = Button("Back", [&state] {
        state.current_screen = static_cast<int>(AppScreen::MainMenu);
    });
    
    auto export_container = Container::Vertical({
        Container::Horizontal({xml_input, export_xml_btn}),
        Container::Horizontal({text_input, export_text_btn}),
        Container::Horizontal({html_input, export_html_btn}),
        back_from_export_btn,
    });
    
    auto export_comp = Renderer(export_container, [export_container] {
        return vbox({
            text("Export Results") | bold | center,
            separator(),
            hbox({text("XML:  "), export_container->ChildAt(0)->Render() | flex}),
            hbox({text("Text: "), export_container->ChildAt(1)->Render() | flex}),
            hbox({text("HTML: "), export_container->ChildAt(2)->Render() | flex}),
            text(""),
            text(g_export_message) | center | color(Color::Green),
            text(""),
            export_container->ChildAt(3)->Render() | center,
        }) | border;
    });
    
    //=========================================================================
    // Tab Container to switch screens
    //=========================================================================
    auto tab = Container::Tab({
        main_menu,             // 0: MainMenu
        test_selection,        // 1: TestSelection
        settings_comp,         // 2: Settings
        running_comp,          // 3: Running
        results_comp,          // 4: Results
        export_comp,           // 5: Export
    }, &state.current_screen);
    
    //=========================================================================
    // Main renderer with header/footer
    //=========================================================================
    auto main_renderer = Renderer(tab, [&state, tab] {
        // Header
        std::string lib_info = state.hip_library.empty() ? "default HIP" : state.hip_library;
        Element header = hbox({
            text(" HIP CTS Interactive Test Runner ") | bold | color(Color::Cyan),
            filler(),
            text("j=" + std::to_string(state.parallel) + " ") | dim,
            text("repeat=" + std::to_string(state.repeat) + " ") | dim,
            text(lib_info) | dim,
        }) | borderLight;
        
        // Footer with navigation hints
        std::string nav_hint;
        switch (state.current_screen) {
            case 0:  // MainMenu
                nav_hint = "↑/↓: Navigate | Enter: Select | q: Quit";
                break;
            case 1:  // TestSelection
                if (state.tree_focused) {
                    nav_hint = "↑/↓: Navigate | Space: Toggle | Enter: Expand/Collapse | Tab: Switch to Buttons";
                } else {
                    nav_hint = "←/→: Select Button | Enter: Activate | Tab: Switch to Tree";
                }
                break;
            case 2:  // Settings
                nav_hint = "Tab: Next field | Enter: Apply";
                break;
            case 3:  // Running
                nav_hint = state.running.load() ? "c: Cancel" : "Esc: Back to menu";
                break;
            case 4:  // Results
                nav_hint = "Tab: Buttons | Enter: Select";
                break;
            case 5:  // Export
                nav_hint = "Tab: Next field | Enter: Save";
                break;
        }
        
        Element footer = text(nav_hint) | dim | center | borderLight;
        
        // Main content with appropriate wrapper
        Element content;
        if (state.current_screen == 0) {
            // Count total selected tests
            int total_selected = 0;
            int total_tests = 0;
            for (const auto& suite : state.suites) {
                total_tests += static_cast<int>(suite.test_cases.size());
                total_selected += suite.selectedCount();
            }
            
            content = vbox({
                text("Main Menu") | bold | center,
                separator(),
                text("Suites: " + std::to_string(state.suites.size()) + 
                     "  |  Tests: " + std::to_string(total_selected) + "/" + 
                     std::to_string(total_tests) + " selected") | dim | center,
                text(""),
                tab->Render() | center,
            }) | border;
        } else if (state.current_screen == 1) {
            // Test selection with tree view
            content = vbox({
                text("Select Tests") | bold | center,
                separator(),
                tab->Render() | flex,
            }) | border;
        } else {
            content = tab->Render();
        }
        
        return vbox({
            header,
            content | flex,
            footer,
        });
    });
    
    //=========================================================================
    // Event handler wrapper
    //=========================================================================
    return CatchEvent(main_renderer, [&state, &screen, &test_thread](Event event) {
        // Handle menu selection on Enter (MainMenu)
        if (state.current_screen == 0 && event == Event::Return) {
            switch (state.menu_selected) {
                case 0:  // Run All Tests
                    for (auto& suite : state.suites) {
                        suite.selected = true;
                        for (auto& tc : suite.test_cases) {
                            tc.selected = true;
                        }
                    }
                    state.current_screen = static_cast<int>(AppScreen::Running);
                    // Join previous thread if finished, then start new one
                    if (!state.running.load()) {
                        if (test_thread.joinable()) {
                            test_thread.join();
                        }
                        test_thread = std::thread(runTestsThread, std::ref(state), std::ref(screen));
                    }
                    return true;
                case 1:  // Run Selected Tests
                    state.current_screen = static_cast<int>(AppScreen::Running);
                    // Join previous thread if finished, then start new one
                    if (!state.running.load()) {
                        if (test_thread.joinable()) {
                            test_thread.join();
                        }
                        test_thread = std::thread(runTestsThread, std::ref(state), std::ref(screen));
                    }
                    return true;
                case 2:  // Select Tests
                    state.current_screen = static_cast<int>(AppScreen::TestSelection);
                    state.tree_focused = true;  // Start with tree focused
                    return true;
                case 3:  // Settings
                    state.current_screen = static_cast<int>(AppScreen::Settings);
                    return true;
                case 4:  // View Results
                    if (state.has_results) {
                        state.current_screen = static_cast<int>(AppScreen::Results);
                    }
                    return true;
                case 5:  // Export
                    if (state.has_results) {
                        state.current_screen = static_cast<int>(AppScreen::Export);
                    }
                    return true;
                case 6:  // Exit
                    screen.Exit();
                    return true;
            }
        }
        
        // Tree navigation for TestSelection screen
        if (state.current_screen == 1) {
            // Tab switches focus between tree and buttons
            if (event == Event::Tab) {
                state.tree_focused = !state.tree_focused;
                return true;
            }
            if (event == Event::TabReverse) {
                state.tree_focused = !state.tree_focused;
                return true;
            }
            
            // Only handle tree navigation when tree is focused
            if (state.tree_focused) {
                int item_count = getTreeItemCount(state);
                
                if (event == Event::ArrowUp || event == Event::Character('k')) {
                    if (state.tree_cursor > 0) {
                        state.tree_cursor--;
                    }
                    return true;
                }
                if (event == Event::ArrowDown || event == Event::Character('j')) {
                    if (state.tree_cursor < item_count - 1) {
                        state.tree_cursor++;
                    }
                    return true;
                }
                if (event == Event::Return) {
                    // Expand/collapse suite
                    auto item = getTreeItemAtCursor(state);
                    if (item.suite_index >= 0 && item.test_index < 0) {
                        state.suites[item.suite_index].expanded = 
                            !state.suites[item.suite_index].expanded;
                    }
                    return true;
                }
                if (event == Event::Character(' ')) {
                    // Toggle selection
                    auto item = getTreeItemAtCursor(state);
                    if (item.suite_index >= 0) {
                        if (item.test_index < 0) {
                            // Toggle all tests in suite
                            auto& suite = state.suites[item.suite_index];
                            bool new_state = suite.selectedCount() < static_cast<int>(suite.test_cases.size());
                            for (auto& tc : suite.test_cases) {
                                tc.selected = new_state;
                            }
                        } else {
                            // Toggle individual test
                            state.suites[item.suite_index].test_cases[item.test_index].selected = 
                                !state.suites[item.suite_index].test_cases[item.test_index].selected;
                        }
                    }
                    return true;
                }
            }
            // When buttons are focused, let events pass through to the button container
        }
        
        // Cancel running tests
        if (state.current_screen == static_cast<int>(AppScreen::Running) && 
            state.running.load() && event.is_character() && event.character() == "c") {
            state.cancel_requested = true;
            return true;
        }
        
        // Handle Escape to go back
        if (event == Event::Escape) {
            if (state.current_screen != 0 && !state.running.load()) {
                // Join test thread if needed
                if (test_thread.joinable()) {
                    test_thread.join();
                }
                state.current_screen = 0;
                return true;
            }
        }
        
        // Handle 'q' to quit from main menu
        if (event.is_character() && event.character() == "q") {
            if (state.current_screen == 0) {
                screen.Exit();
                return true;
            }
        }
        
        return false;
    });
}

} // namespace

int main(int argc, char* argv[]) {
    AppState state;
    
    if (!parseArgs(argc, argv, state)) {
        return 1;
    }
    
    // Configure the test runner
    if (!state.test_directory.empty()) {
        state.runner.setTestDirectory(state.test_directory);
    }
    
    if (!state.hip_library.empty()) {
        state.runner.setHipLibrary(state.hip_library);
    }
    
    // Discover and enumerate tests with individual test cases (use parallel discovery)
    std::cerr << "Discovering tests..." << std::endl;
    auto all_tests = state.runner.enumerateAllTestsParallel(state.parallel > 0 ? state.parallel : 8);
    
    // Build suite selections with individual test cases
    for (const auto& suite : all_tests) {
        SuiteSelection sel;
        sel.name = suite.name;
        sel.executable_path = suite.executable_path;
        sel.expanded = false;
        sel.selected = true;
        
        for (const auto& tc : suite.test_cases) {
            TestCaseSelection tcs;
            tcs.name = tc.name;
            tcs.tags = tc.tags;
            tcs.selected = true;
            sel.test_cases.push_back(std::move(tcs));
        }
        
        state.suites.push_back(std::move(sel));
    }
    
    if (state.suites.empty()) {
        std::cerr << "Error: No tests found. Use -d <directory> to specify test location.\n";
        return 1;
    }
    
    int total_tests = 0;
    for (const auto& suite : state.suites) {
        total_tests += static_cast<int>(suite.test_cases.size());
    }
    std::cerr << "Found " << state.suites.size() << " test suites with " 
              << total_tests << " total tests." << std::endl;
    
    // Run the interactive UI
    auto screen = ScreenInteractive::Fullscreen();
    
    // Thread for running tests
    std::thread test_thread;
    
    // Create and run the app
    auto app = App(state, screen, test_thread);
    
    screen.Loop(app);
    
    // Clean up test thread if running
    if (test_thread.joinable()) {
        test_thread.join();
    }
    
    return 0;
}
