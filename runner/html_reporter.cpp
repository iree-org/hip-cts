// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "html_reporter.hpp"
#include "xml_aggregator.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

// miniz for compression
#include "miniz.h"

namespace hip_cts {
namespace runner {

HtmlReporter::HtmlReporter(const Options& options) : options_(options) {}

std::vector<uint8_t> HtmlReporter::compressData(const std::string& data) {
    // Use miniz to compress with deflate
    mz_ulong compressed_size = mz_compressBound(data.size());
    std::vector<uint8_t> compressed(compressed_size);
    
    int result = mz_compress2(
        compressed.data(), &compressed_size,
        reinterpret_cast<const unsigned char*>(data.data()), data.size(),
        MZ_BEST_COMPRESSION
    );
    
    if (result != MZ_OK) {
        // Compression failed, return empty
        return {};
    }
    
    compressed.resize(compressed_size);
    return compressed;
}

std::string HtmlReporter::base64Encode(const std::vector<uint8_t>& data) {
    static const char* base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);
    
    for (size_t i = 0; i < data.size(); i += 3) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < data.size()) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < data.size()) n |= static_cast<uint32_t>(data[i + 2]);
        
        result.push_back(base64_chars[(n >> 18) & 0x3F]);
        result.push_back(base64_chars[(n >> 12) & 0x3F]);
        result.push_back((i + 1 < data.size()) ? base64_chars[(n >> 6) & 0x3F] : '=');
        result.push_back((i + 2 < data.size()) ? base64_chars[n & 0x3F] : '=');
    }
    
    return result;
}

std::string HtmlReporter::escapeHtml(const std::string& str) {
    std::ostringstream out;
    for (char c : str) {
        switch (c) {
            case '&': out << "&amp;"; break;
            case '<': out << "&lt;"; break;
            case '>': out << "&gt;"; break;
            case '"': out << "&quot;"; break;
            case '\'': out << "&#39;"; break;
            default: out << c;
        }
    }
    return out.str();
}

std::string HtmlReporter::formatDuration(double seconds) {
    std::ostringstream out;
    if (seconds < 0.001) {
        out << std::fixed << std::setprecision(3) << (seconds * 1000.0) << "ms";
    } else if (seconds < 1.0) {
        out << std::fixed << std::setprecision(2) << (seconds * 1000.0) << "ms";
    } else if (seconds < 60.0) {
        out << std::fixed << std::setprecision(2) << seconds << "s";
    } else {
        int mins = static_cast<int>(seconds) / 60;
        double secs = seconds - mins * 60;
        out << mins << "m " << std::fixed << std::setprecision(1) << secs << "s";
    }
    return out.str();
}

std::string HtmlReporter::generate(const AggregatedResults& results) {
    // First, generate the XML using XmlAggregator
    XmlAggregator xml_agg;
    std::string xml_data = xml_agg.generate(results);
    
    // Compress and base64 encode
    std::vector<uint8_t> compressed = compressData(xml_data);
    std::string encoded_data;
    
    if (compressed.empty()) {
        // Fallback: just base64 encode without compression
        std::vector<uint8_t> raw(xml_data.begin(), xml_data.end());
        encoded_data = base64Encode(raw);
    } else {
        encoded_data = base64Encode(compressed);
    }
    
    bool is_compressed = !compressed.empty();
    
    std::ostringstream html;
    
    // Start HTML document
    html << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>)HTML" << escapeHtml(options_.title) << R"HTML(</title>
    <style>
        :root {
            --bg-primary: #0f0f13;
            --bg-secondary: #1a1a23;
            --bg-tertiary: #252532;
            --text-primary: #e8e6e3;
            --text-secondary: #9d9b97;
            --text-dim: #6b6a68;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --accent-yellow: #fbbf24;
            --accent-blue: #60a5fa;
            --accent-purple: #a78bfa;
            --border-color: #3d3d4d;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', 'Source Code Pro', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .meta { color: var(--text-secondary); font-size: 0.85rem; }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .summary-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }
        
        .summary-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1.2; }
        .summary-card .label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.1em; }
        
        .summary-card.passed .value { color: var(--accent-green); }
        .summary-card.failed .value { color: var(--accent-red); }
        .summary-card.skipped .value { color: var(--accent-yellow); }
        .summary-card.total .value { color: var(--accent-blue); }
        .summary-card.time .value { color: var(--accent-purple); font-size: 1.5rem; }
        
        .progress-bar { background: var(--bg-tertiary); border-radius: 4px; height: 8px; margin: 1rem 0; overflow: hidden; }
        .progress-bar .fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
        .progress-bar .fill.good { background: var(--accent-green); }
        .progress-bar .fill.warning { background: var(--accent-yellow); }
        .progress-bar .fill.bad { background: var(--accent-red); }
        
        .suites { display: flex; flex-direction: column; gap: 1.5rem; }
        
        .suite-group {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }
        
        .group-header {
            padding: 1.25rem 1.5rem;
            background: linear-gradient(135deg, var(--bg-tertiary) 0%, #2a2a3a 100%);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        
        .group-header:hover { background: linear-gradient(135deg, #2d2d3d 0%, #353545 100%); }
        .group-name { font-weight: 700; font-size: 1.25rem; color: var(--accent-blue); }
        .group-count { font-size: 0.85rem; color: var(--text-dim); margin-left: 0.5rem; }
        
        .group-body { padding: 0; max-height: 0; overflow: hidden; transition: max-height 0.4s ease; }
        .suite-group.expanded .group-body { max-height: 50000px; padding: 0.75rem; }
        .suite-group.expanded .toggle-icon { transform: rotate(90deg); }
        
        .suite {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        .suite:last-child { margin-bottom: 0; }
        
        .suite-header {
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        .suite-header:hover { background: #2d2d3d; }
        .suite-name { font-weight: 600; font-size: 0.95rem; }
        
        .suite-stats { display: flex; gap: 1rem; font-size: 0.85rem; }
        .suite-stats .stat { display: flex; align-items: center; gap: 0.3rem; }
        .suite-stats .passed { color: var(--accent-green); }
        .suite-stats .failed { color: var(--accent-red); }
        .suite-stats .skipped { color: var(--accent-yellow); }
        
        .suite-body { padding: 0; max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }
        .suite.expanded .suite-body { max-height: 5000px; }
        
        .test-list { list-style: none; }
        .test-item {
            padding: 0.75rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .test-item:last-child { border-bottom: none; }
        .test-item:hover { background: rgba(255, 255, 255, 0.02); }
        .test-name { flex: 1; word-break: break-word; }
        
        .test-status {
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .test-status.passed { background: rgba(74, 222, 128, 0.15); color: var(--accent-green); }
        .test-status.failed { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); }
        .test-status.skipped { background: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); }
        
        .test-time { color: var(--text-dim); font-size: 0.8rem; white-space: nowrap; }
        
        .failure-details {
            width: 100%;
            background: rgba(248, 113, 113, 0.08);
            border-left: 3px solid var(--accent-red);
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            overflow-x: auto;
        }
        .failure-details pre { white-space: pre-wrap; word-wrap: break-word; color: var(--text-secondary); }
        
        .toggle-icon { font-size: 0.8rem; color: var(--text-dim); transition: transform 0.2s; }
        .suite.expanded .toggle-icon { transform: rotate(90deg); }
        
        .config-section {
            margin-top: 2rem;
            padding: 1.25rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        .config-section h3 { font-size: 1rem; color: var(--text-secondary); margin-bottom: 1rem; }
        .config-item { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
        .config-item .label { color: var(--text-dim); min-width: 120px; }
        .config-item .value { color: var(--text-primary); word-break: break-all; }
        
        footer { margin-top: 3rem; text-align: center; color: var(--text-dim); font-size: 0.8rem; }
        
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
            border: none;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 1rem;
        }
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(96, 165, 250, 0.4);
        }
        .download-btn:active {
            transform: translateY(0);
        }
        .download-btn svg {
            width: 16px;
            height: 16px;
        }
        
        .loading { text-align: center; padding: 4rem; color: var(--text-secondary); }
        .loading .spinner {
            width: 40px; height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        @media (max-width: 768px) {
            body { padding: 1rem; }
            h1 { font-size: 1.75rem; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
            .suite-stats { flex-wrap: wrap; gap: 0.5rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Loading test results...</p>
        </div>
        <div id="content" style="display: none;"></div>
    </div>

    <!-- Embedded compressed XML data -->
    <script id="xml-data" type="application/octet-stream">)HTML";
    
    html << encoded_data;
    
    html << R"HTML(</script>
    <script>
        // Data is )HTML" << (is_compressed ? "compressed" : "uncompressed") << R"HTML(
        const IS_COMPRESSED = )HTML" << (is_compressed ? "true" : "false") << R"HTML(;
        
        // Base64 decode
        function base64ToBytes(base64) {
            const binString = atob(base64);
            const bytes = new Uint8Array(binString.length);
            for (let i = 0; i < binString.length; i++) {
                bytes[i] = binString.charCodeAt(i);
            }
            return bytes;
        }
        
        // Decompress using pako-like inflate (zlib format)
        function inflate(data) {
            // Simple zlib decompression using DecompressionStream if available
            // Otherwise fall back to manual implementation
            return new Promise((resolve, reject) => {
                try {
                    // Skip zlib header (first 2 bytes) and adler32 (last 4 bytes)
                    const deflateData = data.slice(2, -4);
                    
                    const ds = new DecompressionStream('deflate-raw');
                    const writer = ds.writable.getWriter();
                    const reader = ds.readable.getReader();
                    
                    writer.write(deflateData);
                    writer.close();
                    
                    const chunks = [];
                    function read() {
                        reader.read().then(({done, value}) => {
                            if (done) {
                                const result = new Uint8Array(chunks.reduce((a, b) => a + b.length, 0));
                                let offset = 0;
                                for (const chunk of chunks) {
                                    result.set(chunk, offset);
                                    offset += chunk.length;
                                }
                                resolve(new TextDecoder().decode(result));
                            } else {
                                chunks.push(value);
                                read();
                            }
                        }).catch(reject);
                    }
                    read();
                } catch (e) {
                    reject(e);
                }
            });
        }
        
        // Parse XML and extract data
        function parseXml(xmlString) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(xmlString, 'text/xml');
            
            const root = doc.querySelector('testsuites');
            const results = {
                name: root.getAttribute('name') || 'HIP CTS Results',
                totalTests: parseInt(root.getAttribute('tests')) || 0,
                totalFailures: parseInt(root.getAttribute('failures')) || 0,
                totalErrors: parseInt(root.getAttribute('errors')) || 0,
                totalSkipped: parseInt(root.getAttribute('skipped')) || 0,
                totalTime: parseFloat(root.getAttribute('time')) || 0,
                timestamp: root.getAttribute('timestamp') || '',
                suites: []
            };
            
            doc.querySelectorAll('testsuite').forEach(suiteEl => {
                const suite = {
                    name: suiteEl.getAttribute('name') || '',
                    tests: parseInt(suiteEl.getAttribute('tests')) || 0,
                    failures: parseInt(suiteEl.getAttribute('failures')) || 0,
                    errors: parseInt(suiteEl.getAttribute('errors')) || 0,
                    skipped: parseInt(suiteEl.getAttribute('skipped')) || 0,
                    time: parseFloat(suiteEl.getAttribute('time')) || 0,
                    testCases: []
                };
                
                suiteEl.querySelectorAll('testcase').forEach(tcEl => {
                    const tc = {
                        name: tcEl.getAttribute('name') || '',
                        className: tcEl.getAttribute('classname') || '',
                        time: parseFloat(tcEl.getAttribute('time')) || 0,
                        passed: true,
                        skipped: false,
                        failureMessage: ''
                    };
                    
                    const failure = tcEl.querySelector('failure');
                    if (failure) {
                        tc.passed = false;
                        tc.failureMessage = failure.textContent || '';
                    }
                    
                    const skipped = tcEl.querySelector('skipped');
                    if (skipped) {
                        tc.skipped = true;
                        tc.passed = true;
                    }
                    
                    suite.testCases.push(tc);
                });
                
                results.suites.push(suite);
            });
            
            return results;
        }
        
        // Extract category from suite name
        function extractCategory(name) {
            let category = name.replace(/_test$/, '');
            const idx = category.indexOf('_');
            if (idx > 0) category = category.substring(0, idx);
            category = category.charAt(0).toUpperCase() + category.slice(1);
            
            // Map common names
            const mapping = {
                'Api': 'API',
                'Memcpy': 'Memory',
                'Memset': 'Memory', 
                'Malloc': 'Memory',
                'Host': 'Memory',
                'Managed': 'Memory'
            };
            return mapping[category] || category;
        }
        
        // Format duration
        function formatDuration(seconds) {
            if (seconds < 0.001) return (seconds * 1000).toFixed(3) + 'ms';
            if (seconds < 1.0) return (seconds * 1000).toFixed(2) + 'ms';
            if (seconds < 60.0) return seconds.toFixed(2) + 's';
            const mins = Math.floor(seconds / 60);
            const secs = seconds - mins * 60;
            return mins + 'm ' + secs.toFixed(1) + 's';
        }
        
        // Escape HTML
        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
        
        // Render the results
        function render(results) {
            const passed = results.totalTests - results.totalFailures - results.totalErrors - results.totalSkipped;
            const passRate = results.totalTests > 0 ? (100 * passed / results.totalTests) : 0;
            
            // Group suites by category
            const groups = {};
            results.suites.forEach(suite => {
                const cat = extractCategory(suite.name);
                if (!groups[cat]) {
                    groups[cat] = { name: cat, suites: [], totalTests: 0, totalFailures: 0, totalSkipped: 0, totalTime: 0 };
                }
                groups[cat].suites.push(suite);
                groups[cat].totalTests += suite.tests;
                groups[cat].totalFailures += suite.failures + suite.errors;
                groups[cat].totalSkipped += suite.skipped;
                groups[cat].totalTime += suite.time;
            });
            
            // Sort groups
            const sortedGroups = Object.values(groups).sort((a, b) => {
                if (a.totalFailures !== b.totalFailures) return b.totalFailures - a.totalFailures;
                return a.name.localeCompare(b.name);
            });
            
            let html = `
                <header>
                    <h1>)HTML" << escapeHtml(options_.title) << R"HTML(</h1>
                    <p class="meta">Generated: ${escapeHtml(results.timestamp)}</p>
                    <button class="download-btn" onclick="downloadXml()">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Download XML
                    </button>
                </header>
                
                <div class="summary-grid">
                    <div class="summary-card total">
                        <div class="value">${results.totalTests}</div>
                        <div class="label">Total Tests</div>
                    </div>
                    <div class="summary-card passed">
                        <div class="value">${passed}</div>
                        <div class="label">Passed</div>
                    </div>
                    <div class="summary-card failed">
                        <div class="value">${results.totalFailures}</div>
                        <div class="label">Failed</div>
                    </div>
                    <div class="summary-card skipped">
                        <div class="value">${results.totalSkipped}</div>
                        <div class="label">Skipped</div>
                    </div>
                    <div class="summary-card time">
                        <div class="value">${formatDuration(results.totalTime)}</div>
                        <div class="label">Duration</div>
                    </div>
                </div>
                
                <div class="progress-bar">
                    <div class="fill ${passRate >= 90 ? 'good' : (passRate >= 70 ? 'warning' : 'bad')}" 
                         style="width: ${passRate.toFixed(1)}%;"></div>
                </div>
                
                <div class="suites">
            `;
            
            sortedGroups.forEach(group => {
                const groupPassed = group.totalTests - group.totalFailures - group.totalSkipped;
                const hasFailures = group.totalFailures > 0;
                
                html += `
                    <div class="suite-group ${hasFailures ? 'expanded' : ''}">
                        <div class="group-header" onclick="this.parentElement.classList.toggle('expanded')">
                            <div>
                                <span class="group-name">${escapeHtml(group.name)} Tests</span>
                                <span class="group-count">(${group.suites.length} suites)</span>
                            </div>
                            <div class="suite-stats">
                                <span class="stat passed">&#10003; ${groupPassed}</span>
                                <span class="stat failed">&#10007; ${group.totalFailures}</span>
                                ${group.totalSkipped > 0 ? `<span class="stat skipped">&#9675; ${group.totalSkipped}</span>` : ''}
                                <span class="stat">${formatDuration(group.totalTime)}</span>
                                <span class="toggle-icon">&#9654;</span>
                            </div>
                        </div>
                        <div class="group-body">
                `;
                
                group.suites.forEach(suite => {
                    const suitePassed = suite.tests - suite.failures - suite.errors - suite.skipped;
                    const suiteHasFailures = suite.failures > 0 || suite.errors > 0;
                    
                    html += `
                        <div class="suite ${suiteHasFailures ? 'expanded' : ''}">
                            <div class="suite-header" onclick="event.stopPropagation(); this.parentElement.classList.toggle('expanded')">
                                <div>
                                    <span class="suite-name">${escapeHtml(suite.name)}</span>
                                </div>
                                <div class="suite-stats">
                                    <span class="stat passed">&#10003; ${suitePassed}</span>
                                    <span class="stat failed">&#10007; ${suite.failures}</span>
                                    ${suite.skipped > 0 ? `<span class="stat skipped">&#9675; ${suite.skipped}</span>` : ''}
                                    <span class="stat">${formatDuration(suite.time)}</span>
                                    <span class="toggle-icon">&#9654;</span>
                                </div>
                            </div>
                            <div class="suite-body">
                                <ul class="test-list">
                    `;
                    
                    suite.testCases.forEach(tc => {
                        const statusClass = tc.skipped ? 'skipped' : (tc.passed ? 'passed' : 'failed');
                        const statusText = tc.skipped ? 'SKIP' : (tc.passed ? 'PASS' : 'FAIL');
                        
                        html += `
                            <li class="test-item">
                                <span class="test-name">${escapeHtml(tc.name)}</span>
                                <span class="test-time">${formatDuration(tc.time)}</span>
                                <span class="test-status ${statusClass}">${statusText}</span>
                                ${(!tc.passed && !tc.skipped && tc.failureMessage) ? 
                                    `<div class="failure-details"><pre>${escapeHtml(tc.failureMessage)}</pre></div>` : ''}
                            </li>
                        `;
                    });
                    
                    html += `
                                </ul>
                            </div>
                        </div>
                    `;
                });
                
                html += `
                        </div>
                    </div>
                `;
            });
            
            html += `
                </div>
                <footer>
                    <p>HIP CTS Runner &bull; ${results.suites.length} test suite(s)</p>
                </footer>
            `;
            
            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').innerHTML = html;
            document.getElementById('content').style.display = 'block';
        }
        
        // Store XML string for download
        let cachedXmlString = null;
        
        // Download XML file
        async function downloadXml() {
            try {
                if (!cachedXmlString) {
                    const encodedData = document.getElementById('xml-data').textContent.trim();
                    const bytes = base64ToBytes(encodedData);
                    
                    if (IS_COMPRESSED) {
                        cachedXmlString = await inflate(bytes);
                    } else {
                        cachedXmlString = new TextDecoder().decode(bytes);
                    }
                }
                
                const blob = new Blob([cachedXmlString], { type: 'application/xml' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'hip_cts_results.xml';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (e) {
                alert('Error downloading XML: ' + e.message);
                console.error(e);
            }
        }
        
        // Main entry point
        async function main() {
            try {
                const encodedData = document.getElementById('xml-data').textContent.trim();
                const bytes = base64ToBytes(encodedData);
                
                if (IS_COMPRESSED) {
                    cachedXmlString = await inflate(bytes);
                } else {
                    cachedXmlString = new TextDecoder().decode(bytes);
                }
                
                const results = parseXml(cachedXmlString);
                render(results);
            } catch (e) {
                document.getElementById('loading').innerHTML = 
                    '<p style="color: var(--accent-red);">Error loading results: ' + e.message + '</p>';
                console.error(e);
            }
        }
        
        main();
    </script>
</body>
</html>)HTML";
    
    return html.str();
}

void HtmlReporter::write(const AggregatedResults& results, std::ostream& out) {
    out << generate(results);
}

bool HtmlReporter::writeToFile(const AggregatedResults& results, 
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
