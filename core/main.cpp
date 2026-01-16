// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <string>

#include "hip_loader.hpp"

// Global variable to store the HIP library path (set from command line)
static std::string g_hip_library_path;

// Global variable to store the selected HIP device (set from command line)
// Used by HipTestFixture to set the device before each test
int g_hip_device = 0;

int main(int argc, char* argv[]) {
    Catch::Session session;

    // Build a new command line parser on top of Catch2's
    using namespace Catch::Clara;
    
    auto cli = session.cli()
        | Opt(g_hip_library_path, "library")
            ["--hip-library"]
            ("Path to the HIP library (.so file) to test. "
             "Defaults to the backend-configured library if not specified.")
        | Opt(g_hip_device, "device")
            ["--hip-device"]
            ("HIP device index to use for tests. Defaults to 0.");

    session.cli(cli);

    // Parse the command line
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    // Initialize the HIP loader before running tests
    try {
        std::cout << "HIP CTS: Backend: " << hip_cts::config::kBackendName << "\n";
        if (g_hip_library_path.empty()) {
            std::cout << "HIP CTS: Using default HIP library: " 
                      << hip_cts::config::kDefaultHipLibrary << "\n";
        } else {
            std::cout << "HIP CTS: Using HIP library: " << g_hip_library_path << "\n";
            HipLoader::setLibraryPath(g_hip_library_path);
        }
        
        // Access the singleton to trigger initialization (thread-safe)
        auto& loader = HipLoader::instance();
        
        // Print some information about the loaded library
        std::cout << "HIP CTS: Successfully loaded: " << hip().libraryPath() << "\n";
        
        // Try to get device count to verify the library is functional
        int deviceCount = 0;
        hipError_t err = hip().hipGetDeviceCount(&deviceCount);
        if (err == hipSuccess) {
            std::cout << "HIP CTS: Found " << deviceCount << " device(s)\n";
            
            // Print device info for each device
            for (int i = 0; i < deviceCount; ++i) {
                hipDeviceProp_t props;
                if (hip().hipGetDeviceProperties(&props, i) == hipSuccess) {
                    std::cout << "HIP CTS: Device " << i << ": " << props.name << "\n";
                }
            }
            
            // Validate the selected device
            if (g_hip_device < 0 || g_hip_device >= deviceCount) {
                std::cerr << "HIP CTS: Error - Invalid device index " << g_hip_device 
                          << ". Valid range is 0 to " << (deviceCount - 1) << "\n";
                return 1;
            }
            
            std::cout << "HIP CTS: Using device " << g_hip_device << " for tests\n";
        } else {
            std::cerr << "HIP CTS: Warning - hipGetDeviceCount failed with error: " 
                      << hip().hipGetErrorString(err) << "\n";
        }
        
        std::cout << "\n";
        
    } catch (const HipLoaderError& e) {
        std::cerr << "HIP CTS: Failed to initialize HIP loader: " << e.what() << "\n";
        std::cerr << "HIP CTS: Make sure the HIP library is installed and accessible.\n";
        std::cerr << "HIP CTS: You can specify a custom library path with --hip-library\n";
        return 1;
    }

    // Run the tests
    return session.run();
}

