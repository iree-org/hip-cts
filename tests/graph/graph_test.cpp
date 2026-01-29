// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Helper macro to skip if graph API is not available
//=============================================================================
#define REQUIRE_GRAPH_API() \
    do { \
        if (!hip().hipGraphCreate || !hip().hipGraphDestroy) { \
            SKIP("Graph API not available"); \
        } \
    } while(0)

// Check if we're using the streaming backend by checking device name format
// Streaming backend returns short names like "gfx942", native HIP returns full
// names like "AMD Instinct MI300X"
static bool isStreamingBackend(HipLoader& loader) {
    hipDeviceProp_t props;
    if (loader.hipGetDeviceProperties(&props, 0) != hipSuccess) {
        return false;
    }
    // Streaming backend uses short device name (gcnArchName-like) for device name
    return std::string(props.name).find("AMD") == std::string::npos &&
           std::string(props.name).find("gfx") == 0;
}

#define SKIP_ON_STREAMING_BACKEND() \
    do { \
        if (isStreamingBackend(hip())) { \
            SKIP("Test not supported on streaming backend"); \
        } \
    } while(0)

//=============================================================================
// hipGraphCreate Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphCreate basic", "[graph][create]") {
    REQUIRE_GRAPH_API();
    
    hipGraph_t graph = nullptr;
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(graph != nullptr);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphCreate with null pointer fails", "[graph][create][negative]") {
    REQUIRE_GRAPH_API();
    
    REQUIRE(hip().hipGraphCreate(nullptr, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphCreate with non-zero flags fails", "[graph][create][negative]") {
    REQUIRE_GRAPH_API();
    
    hipGraph_t graph = nullptr;
    hipError_t err = hip().hipGraphCreate(&graph, 1);
    // Some implementations may accept non-zero flags, others return error
    if (err == hipSuccess) {
        // If it succeeded, destroy the graph
        hip().hipGraphDestroy(graph);
    } else {
        REQUIRE(err == hipErrorInvalidValue);
    }
}

//=============================================================================
// hipGraphDestroy Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphDestroy basic", "[graph][destroy]") {
    REQUIRE_GRAPH_API();
    
    hipGraph_t graph = nullptr;
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(graph != nullptr);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphDestroy with null fails", "[graph][destroy][negative]") {
    REQUIRE_GRAPH_API();
    
    REQUIRE(hip().hipGraphDestroy(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// hipGraphInstantiate Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphInstantiate empty graph", "[graph][instantiate]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphExecDestroy) {
        SKIP("hipGraphInstantiate not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(graph != nullptr);
    
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(graphExec != nullptr);
    
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphInstantiate null pGraphExec fails", "[graph][instantiate][negative]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate) {
        SKIP("hipGraphInstantiate not available");
    }
    
    hipGraph_t graph = nullptr;
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    REQUIRE(hip().hipGraphInstantiate(nullptr, graph, nullptr, nullptr, 0) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphInstantiate null graph fails", "[graph][instantiate][negative]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate) {
        SKIP("hipGraphInstantiate not available");
    }
    
    hipGraphExec_t graphExec = nullptr;
    REQUIRE(hip().hipGraphInstantiate(&graphExec, nullptr, nullptr, nullptr, 0) == hipErrorInvalidValue);
}

//=============================================================================
// hipGraphLaunch Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphLaunch empty graph", "[graph][launch]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || !hip().hipGraphExecDestroy) {
        SKIP("Graph launch API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipGraphLaunch(graphExec, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphLaunch null graphExec fails", "[graph][launch][negative]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphLaunch) {
        SKIP("hipGraphLaunch not available");
    }
    
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipGraphLaunch(nullptr, stream) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphLaunch on null stream", "[graph][launch]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || !hip().hipGraphExecDestroy) {
        SKIP("Graph launch API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    
    // Launch on null stream (default stream)
    REQUIRE(hip().hipGraphLaunch(graphExec, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// hipGraphAddEmptyNode Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphAddEmptyNode basic", "[graph][node][empty]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphAddEmptyNode) {
        SKIP("hipGraphAddEmptyNode not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t emptyNode = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0) == hipSuccess);
    REQUIRE(emptyNode != nullptr);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphAddEmptyNode with dependency", "[graph][node][empty]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphAddEmptyNode) {
        SKIP("hipGraphAddEmptyNode not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t node1 = nullptr;
    hipGraphNode_t node2 = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    // Create first node
    REQUIRE(hip().hipGraphAddEmptyNode(&node1, graph, nullptr, 0) == hipSuccess);
    REQUIRE(node1 != nullptr);
    
    // Create second node dependent on first
    REQUIRE(hip().hipGraphAddEmptyNode(&node2, graph, &node1, 1) == hipSuccess);
    REQUIRE(node2 != nullptr);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphAddEmptyNode null pGraphNode fails", "[graph][node][empty][negative]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphAddEmptyNode) {
        SKIP("hipGraphAddEmptyNode not available");
    }
    
    hipGraph_t graph = nullptr;
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    REQUIRE(hip().hipGraphAddEmptyNode(nullptr, graph, nullptr, 0) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// hipGraphAddDependencies Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphAddDependencies basic", "[graph][dependencies]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphAddEmptyNode || !hip().hipGraphAddDependencies) {
        SKIP("Graph node/dependency API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t node1 = nullptr;
    hipGraphNode_t node2 = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    // Create two independent nodes
    REQUIRE(hip().hipGraphAddEmptyNode(&node1, graph, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&node2, graph, nullptr, 0) == hipSuccess);
    
    // Add dependency: node1 -> node2
    hipError_t err = hip().hipGraphAddDependencies(graph, &node1, &node2, 1);
    if (err == hipErrorNotSupported) {
        hip().hipGraphDestroy(graph);
        SKIP("hipGraphAddDependencies not fully supported");
    }
    REQUIRE(err == hipSuccess);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// hipGraphGetNodes Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphGetNodes empty graph", "[graph][nodes]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphGetNodes) {
        SKIP("hipGraphGetNodes not available");
    }
    
    hipGraph_t graph = nullptr;
    size_t numNodes = 999;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    // Query number of nodes
    REQUIRE(hip().hipGraphGetNodes(graph, nullptr, &numNodes) == hipSuccess);
    REQUIRE(numNodes == 0);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGraphGetNodes with nodes", "[graph][nodes]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphGetNodes || !hip().hipGraphAddEmptyNode) {
        SKIP("Graph node API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t node1 = nullptr;
    hipGraphNode_t node2 = nullptr;
    size_t numNodes = 0;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&node1, graph, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&node2, graph, nullptr, 0) == hipSuccess);
    
    // Query number of nodes
    REQUIRE(hip().hipGraphGetNodes(graph, nullptr, &numNodes) == hipSuccess);
    REQUIRE(numNodes == 2);
    
    // Get the nodes
    std::vector<hipGraphNode_t> nodes(numNodes);
    REQUIRE(hip().hipGraphGetNodes(graph, nodes.data(), &numNodes) == hipSuccess);
    REQUIRE(numNodes == 2);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// hipGraphGetRootNodes Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphGetRootNodes", "[graph][nodes][root]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphGetRootNodes || !hip().hipGraphAddEmptyNode || !hip().hipGraphAddDependencies) {
        SKIP("Graph root node API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t node1 = nullptr;
    hipGraphNode_t node2 = nullptr;
    hipGraphNode_t node3 = nullptr;
    size_t numRootNodes = 0;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    // Create: node1 -> node2, node3 (independent)
    REQUIRE(hip().hipGraphAddEmptyNode(&node1, graph, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&node2, graph, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&node3, graph, nullptr, 0) == hipSuccess);
    
    hipError_t err = hip().hipGraphAddDependencies(graph, &node1, &node2, 1);
    if (err == hipErrorNotSupported) {
        hip().hipGraphDestroy(graph);
        SKIP("hipGraphAddDependencies not fully supported");
    }
    REQUIRE(err == hipSuccess);
    
    // Query root nodes - should be node1 and node3
    REQUIRE(hip().hipGraphGetRootNodes(graph, nullptr, &numRootNodes) == hipSuccess);
    REQUIRE(numRootNodes == 2);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// hipGraphNodeGetType Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraphNodeGetType empty node", "[graph][node][type]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphAddEmptyNode || !hip().hipGraphNodeGetType) {
        SKIP("Graph node type API not available");
    }
    
    hipGraph_t graph = nullptr;
    hipGraphNode_t emptyNode = nullptr;
    hipGraphNodeType nodeType;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0) == hipSuccess);
    
    REQUIRE(hip().hipGraphNodeGetType(emptyNode, &nodeType) == hipSuccess);
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
    
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// Graph with Memset Node Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraph with memset node", "[graph][memset]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || 
        !hip().hipGraphExecDestroy || !hip().hipGraphAddMemsetNode) {
        SKIP("Graph memset API not available");
    }
    
    constexpr size_t size = 1024;
    void* devicePtr = nullptr;
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Initialize to non-zero
    REQUIRE(hip().hipMemset(devicePtr, 0xFF, size) == hipSuccess);
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    hipGraphNode_t memsetNode = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    hipMemsetParams memsetParams = {};
    memsetParams.dst = devicePtr;
    memsetParams.elementSize = 1;
    memsetParams.height = 1;
    memsetParams.pitch = size;
    memsetParams.value = 0xAB;
    memsetParams.width = size;
    
    REQUIRE(hip().hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams) == hipSuccess);
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipGraphLaunch(graphExec, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    std::vector<uint8_t> hostData(size);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// Graph with Memcpy Node Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraph with memcpy node", "[graph][memcpy]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || 
        !hip().hipGraphExecDestroy) {
        SKIP("Graph launch API not available");
    }
    if (!hip().hipGraphAddMemcpyNode1D) {
        SKIP("hipGraphAddMemcpyNode1D not available");
    }
    
    constexpr size_t size = 1024;
    void* devicePtr = nullptr;
    std::vector<uint8_t> hostSrc(size, 0x42);
    std::vector<uint8_t> hostDst(size, 0);
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    hipGraphNode_t h2dNode = nullptr;
    hipGraphNode_t d2hNode = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    // Add H2D memcpy node
    REQUIRE(hip().hipGraphAddMemcpyNode1D(&h2dNode, graph, nullptr, 0, 
                                           devicePtr, hostSrc.data(), size,
                                           hipMemcpyHostToDevice) == hipSuccess);
    
    // Add D2H memcpy node with dependency on H2D
    REQUIRE(hip().hipGraphAddMemcpyNode1D(&d2hNode, graph, &h2dNode, 1,
                                           hostDst.data(), devicePtr, size,
                                           hipMemcpyDeviceToHost) == hipSuccess);
    
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipGraphLaunch(graphExec, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostDst[i] == 0x42);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// Graph with Host Node Tests  
//=============================================================================

static int g_hostNodeTestValue = 0;

static void hostNodeSetValue(void* data) {
    int* value = static_cast<int*>(data);
    *value = 42;
}

static void hostNodeAddValue(void* data) {
    int* value = static_cast<int*>(data);
    *value += 10;
}

TEST_CASE_METHOD(HipTestFixture, "hipGraph with host node", "[graph][host]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || 
        !hip().hipGraphExecDestroy || !hip().hipGraphAddHostNode) {
        SKIP("Graph host node API not available");
    }
    
    g_hostNodeTestValue = 0;
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    hipGraphNode_t hostNode1 = nullptr;
    hipGraphNode_t hostNode2 = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    hipHostNodeParams params1 = {};
    params1.fn = hostNodeSetValue;
    params1.userData = &g_hostNodeTestValue;
    
    hipHostNodeParams params2 = {};
    params2.fn = hostNodeAddValue;
    params2.userData = &g_hostNodeTestValue;
    
    REQUIRE(hip().hipGraphAddHostNode(&hostNode1, graph, nullptr, 0, &params1) == hipSuccess);
    REQUIRE(hip().hipGraphAddHostNode(&hostNode2, graph, &hostNode1, 1, &params2) == hipSuccess);
    
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipGraphLaunch(graphExec, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(g_hostNodeTestValue == 52);  // 42 + 10
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}

//=============================================================================
// Multiple Graph Launches
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraph multiple launches", "[graph][launch][memset]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || 
        !hip().hipGraphExecDestroy || !hip().hipGraphAddMemsetNode) {
        SKIP("Graph API not available");
    }
    
    constexpr size_t size = 256;
    void* devicePtr = nullptr;
    
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    hipGraphNode_t memsetNode = nullptr;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    
    hipMemsetParams memsetParams = {};
    memsetParams.dst = devicePtr;
    memsetParams.elementSize = 1;
    memsetParams.height = 1;
    memsetParams.pitch = size;
    memsetParams.value = 0;
    memsetParams.width = size;
    
    REQUIRE(hip().hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams) == hipSuccess);
    REQUIRE(hip().hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Launch multiple times
    constexpr int numLaunches = 5;
    for (int i = 0; i < numLaunches; ++i) {
        REQUIRE(hip().hipGraphLaunch(graphExec, stream) == hipSuccess);
    }
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify (memset to 0)
    std::vector<uint8_t> hostData(size);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0);
    }
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphExecDestroy(graphExec) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// Multiple Graph Instantiations
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGraph multiple instantiations", "[graph][instantiate][launch]") {
    REQUIRE_GRAPH_API();
    if (!hip().hipGraphInstantiate || !hip().hipGraphLaunch || !hip().hipGraphExecDestroy) {
        SKIP("Graph API not available");
    }
    
    hipGraph_t graph = nullptr;
    constexpr int numInstances = 3;
    std::vector<hipGraphExec_t> graphExecs(numInstances, nullptr);
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipGraphCreate(&graph, 0) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Create multiple instantiations
    for (int i = 0; i < numInstances; ++i) {
        REQUIRE(hip().hipGraphInstantiate(&graphExecs[i], graph, nullptr, nullptr, 0) == hipSuccess);
        REQUIRE(graphExecs[i] != nullptr);
    }
    
    // Launch all
    for (int i = 0; i < numInstances; ++i) {
        REQUIRE(hip().hipGraphLaunch(graphExecs[i], stream) == hipSuccess);
    }
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Cleanup
    for (int i = 0; i < numInstances; ++i) {
        REQUIRE(hip().hipGraphExecDestroy(graphExecs[i]) == hipSuccess);
    }
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipGraphDestroy(graph) == hipSuccess);
}
