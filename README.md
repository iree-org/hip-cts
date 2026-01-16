# HIP CTS

A conformance test suite for validating HIP (Heterogeneous-compute Interface for Portability) runtime implementations.

## Overview

HIP CTS provides a comprehensive set of tests to verify that a HIP runtime implementation correctly implements the HIP API. The test suite dynamically loads HIP libraries at runtime, allowing it to test different HIP implementations without recompilation.

### Key Features

- **Runtime Library Loading** — Tests any HIP implementation via dynamic loading (`dlopen`/`dlsym`)
- **Configurable Backends** — JSON-based backend configuration for different HIP implementations
- **Comprehensive Coverage** — Tests for device management, memory operations, streams, events, kernels, and more
- **Multiple Output Formats** — HTML, XML, and text test reports
- **GPU Target Support** — Configurable GPU architecture targets for kernel compilation

## Requirements

- CMake 3.20 or later
- C++17 compatible compiler
- Python 3 (for kernel embedding)
- A HIP runtime library to test against
- (Optional) HIP compiler (`hipcc` or compatible) for kernel tests

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `HIP_CTS_BACKEND_CONFIG` | Path to backend configuration JSON file (default: AMD config) |

## Running Tests

### Using CTest

```bash
cd build
ctest --output-on-failure
```

### Running Individual Tests

```bash
# Run a specific test executable
./tests/api_smoke/api_smoke

# Run with a specific HIP device
./tests/api_smoke/api_smoke --device 1

# Run tests matching a pattern
./tests/api_smoke/api_smoke "[memory]"
```

### Using the Test Runner

The included test runner provides aggregated HTML/XML reports:

```bash
./runner/hip_cts_runner --help
```

## Project Structure

```
hip-cts/
├── cmake/              # CMake configuration and helper functions
│   ├── HipKernels.cmake           # Kernel compilation and embedding
│   └── hip_backend_config_amd.json # AMD backend configuration
├── core/               # Core test infrastructure
│   ├── hip_loader.hpp             # Dynamic HIP library loader
│   ├── hip_test_fixture.hpp       # Catch2 test fixtures
│   └── hip_types.hpp              # HIP type definitions
├── runner/             # Test runner with reporting
├── tests/              # Test suites
│   ├── api_smoke/                 # Basic API functionality tests
│   ├── kernel_smoke/              # Kernel launch tests
│   └── memory/                    # Memory operation tests
├── tools/              # Build utilities
│   └── embed_binary.py            # Binary-to-C++ embedding tool
└── third_party/        # Dependencies (Catch2)
```

## Test Categories

| Category | Description |
|----------|-------------|
| `[device]` | Device enumeration, selection, and properties |
| `[memory]` | Memory allocation and deallocation |
| `[memcpy]` | Memory copy operations |
| `[memset]` | Memory initialization |
| `[stream]` | Stream creation and synchronization |
| `[event]` | Event creation and timing |
| `[error]` | Error handling and reporting |
| `[kernel]` | Kernel loading and execution |

## Writing Tests

Tests use [Catch2](https://github.com/catchorg/Catch2) and the `HipTestFixture` for automatic device setup:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

TEST_CASE_METHOD(HipTestFixture, "My HIP test", "[memory]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(ptr != nullptr);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}
```

### Adding Tests with Kernels

Use `add_hip_cts_test` in CMakeLists.txt for tests that require GPU kernels:

```cmake
add_hip_cts_test(my_kernel_test
    SOURCES my_test.cpp
    KERNELS my_kernel.hip
)
```

The kernel will be compiled and embedded, accessible via the generated header:

```cpp
#include "my_kernel_test_my_kernel.hpp"

// Load the kernel from embedded data
hip().hipModuleLoadData(&module, my_kernel_test_my_kernel_data.data);
```

## Backend Configuration

Backend configuration files (JSON) specify how to find and use a particular HIP implementation:

```json
{
  "name": "My HIP Backend",
  "runtime": {
    "default_library": "libamdhip64.so",
    "library_search_paths": ["/opt/rocm/lib"],
    "env_library_path": "ROCM_PATH",
    "env_library_subdir": "lib"
  },
  "compiler": {
    "executable": "hipcc",
    "find_hint_env": "ROCM_PATH",
    "find_hint_subdir": "bin"
  },
  "compilation": {
    "output_extension": ".hsaco",
    "command_template": ["${COMPILER}", "--genco", "-o", "${OUTPUT}", "${TARGET_FLAGS}", "${SOURCE}"]
  },
  "targets": {
    "flag_prefix": "--offload-arch=",
    "defaults": ["gfx900", "gfx906", "gfx1030"]
  }
}
```

## Contributing

Contributions are welcome! When adding new tests:

1. Use the `HipTestFixture` base class for automatic device setup
2. Tag tests appropriately (e.g., `[memory]`, `[stream]`)
3. Clean up all allocated resources
4. Test error conditions where applicable

## License

Apache License 2.0 with LLVM Exceptions. See [LICENSE](LICENSE) for details.
