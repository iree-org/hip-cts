// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xml_aggregator.hpp"
#include "tinyxml2.h"

#include <fstream>
#include <sstream>

namespace hip_cts {
namespace runner {

std::string XmlAggregator::generate(const AggregatedResults& results) {
    tinyxml2::XMLDocument doc;
    
    // XML declaration
    doc.InsertFirstChild(doc.NewDeclaration());
    
    // Root element: testsuites
    auto* root = doc.NewElement("testsuites");
    root->SetAttribute("name", results.name.c_str());
    root->SetAttribute("tests", results.total_tests);
    root->SetAttribute("failures", results.total_failures);
    root->SetAttribute("errors", results.total_errors);
    root->SetAttribute("skipped", results.total_skipped);
    root->SetAttribute("time", results.total_time_seconds);
    root->SetAttribute("timestamp", results.timestamp.c_str());
    doc.InsertEndChild(root);
    
    // Add each test suite
    for (const auto& suite : results.suites) {
        auto* suite_el = doc.NewElement("testsuite");
        suite_el->SetAttribute("name", suite.name.c_str());
        suite_el->SetAttribute("tests", suite.tests);
        suite_el->SetAttribute("failures", suite.failures);
        suite_el->SetAttribute("errors", suite.errors);
        suite_el->SetAttribute("skipped", suite.skipped);
        suite_el->SetAttribute("time", suite.time_seconds);
        suite_el->SetAttribute("timestamp", suite.timestamp.c_str());
        suite_el->SetAttribute("file", suite.executable_path.c_str());
        
        // Add properties with execution info
        auto* props = doc.NewElement("properties");
        
        auto* prop_cmd = doc.NewElement("property");
        prop_cmd->SetAttribute("name", "command");
        prop_cmd->SetAttribute("value", suite.command_line.c_str());
        props->InsertEndChild(prop_cmd);
        
        auto* prop_exit = doc.NewElement("property");
        prop_exit->SetAttribute("name", "exit_code");
        prop_exit->SetAttribute("value", suite.exit_code);
        props->InsertEndChild(prop_exit);
        
        suite_el->InsertEndChild(props);
        
        // Add test cases
        for (const auto& tc : suite.test_cases) {
            auto* tc_el = doc.NewElement("testcase");
            tc_el->SetAttribute("name", tc.name.c_str());
            tc_el->SetAttribute("classname", tc.class_name.c_str());
            tc_el->SetAttribute("time", tc.time_seconds);
            
            if (!tc.passed && !tc.skipped) {
                auto* failure_el = doc.NewElement("failure");
                if (!tc.failure_type.empty()) {
                    failure_el->SetAttribute("type", tc.failure_type.c_str());
                }
                if (!tc.failure_message.empty()) {
                    failure_el->SetText(tc.failure_message.c_str());
                }
                tc_el->InsertEndChild(failure_el);
            }
            
            if (tc.skipped) {
                auto* skipped_el = doc.NewElement("skipped");
                tc_el->InsertEndChild(skipped_el);
            }
            
            if (!tc.stdout_output.empty()) {
                auto* stdout_el = doc.NewElement("system-out");
                stdout_el->SetText(tc.stdout_output.c_str());
                tc_el->InsertEndChild(stdout_el);
            }
            
            if (!tc.stderr_output.empty()) {
                auto* stderr_el = doc.NewElement("system-err");
                stderr_el->SetText(tc.stderr_output.c_str());
                tc_el->InsertEndChild(stderr_el);
            }
            
            suite_el->InsertEndChild(tc_el);
        }
        
        // Add suite-level system-out/err
        if (!suite.stdout_output.empty()) {
            auto* stdout_el = doc.NewElement("system-out");
            stdout_el->SetText(suite.stdout_output.c_str());
            suite_el->InsertEndChild(stdout_el);
        }
        
        if (!suite.stderr_output.empty()) {
            auto* stderr_el = doc.NewElement("system-err");
            stderr_el->SetText(suite.stderr_output.c_str());
            suite_el->InsertEndChild(stderr_el);
        }
        
        root->InsertEndChild(suite_el);
    }
    
    // Convert to string
    tinyxml2::XMLPrinter printer;
    doc.Print(&printer);
    return printer.CStr();
}

void XmlAggregator::write(const AggregatedResults& results, std::ostream& out) {
    out << generate(results);
}

bool XmlAggregator::writeToFile(const AggregatedResults& results, 
                                 const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        return false;
    }
    write(results, file);
    return true;
}

} // namespace runner
} // namespace hip_cts
