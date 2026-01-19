// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hip_types.hpp"
#include "hip_backend_config.hpp"
#include <string>
#include <stdexcept>

// Exception thrown when HIP loader encounters an error
class HipLoaderError : public std::runtime_error {
public:
    explicit HipLoaderError(const std::string& message) 
        : std::runtime_error(message) {}
};

// Dynamic loader for HIP runtime library
// This class loads HIP functions via dlopen/dlsym, allowing testing
// of different HIP implementations at runtime.
//
// Thread-safe singleton: Uses Meyers singleton pattern for guaranteed
// thread-safe initialization (C++11 and later).
class HipLoader {
public:
    // Set the library path before first access to instance()
    // Must be called before instance() is first called.
    // If not called, uses the default from backend config.
    static void setLibraryPath(const std::string& library_path);

    // Get the global singleton instance (thread-safe)
    // On first call, loads the HIP library using the path set via setLibraryPath()
    // or the default from backend config.
    static HipLoader& instance();

    // Get the path of the loaded library
    const std::string& libraryPath() const { return library_path_; }

    // Prevent copying and moving
    HipLoader(const HipLoader&) = delete;
    HipLoader& operator=(const HipLoader&) = delete;
    HipLoader(HipLoader&&) = delete;
    HipLoader& operator=(HipLoader&&) = delete;

    ~HipLoader();

    //=========================================================================
    // Device Management
    //=========================================================================
    hipError_t (*hipInit)(unsigned int flags) = nullptr;
    hipError_t (*hipDriverGetVersion)(int* driverVersion) = nullptr;
    hipError_t (*hipRuntimeGetVersion)(int* runtimeVersion) = nullptr;
    hipError_t (*hipGetDeviceCount)(int* count) = nullptr;
    hipError_t (*hipGetDevice)(int* deviceId) = nullptr;
    hipError_t (*hipSetDevice)(int deviceId) = nullptr;
    hipError_t (*hipDeviceSynchronize)() = nullptr;
    hipError_t (*hipDeviceReset)() = nullptr;
    hipError_t (*hipGetDeviceProperties)(hipDeviceProp_t* prop, int deviceId) = nullptr;
    hipError_t (*hipDeviceGetAttribute)(int* pi, hipDeviceAttribute_t attr, int deviceId) = nullptr;
    hipError_t (*hipDeviceGetLimit)(size_t* pValue, enum hipLimit_t limit) = nullptr;
    hipError_t (*hipDeviceSetLimit)(enum hipLimit_t limit, size_t value) = nullptr;
    hipError_t (*hipDeviceGetCacheConfig)(hipFuncCache_t* cacheConfig) = nullptr;
    hipError_t (*hipDeviceSetCacheConfig)(hipFuncCache_t cacheConfig) = nullptr;
    hipError_t (*hipDeviceGetSharedMemConfig)(hipSharedMemConfig* pConfig) = nullptr;
    hipError_t (*hipDeviceSetSharedMemConfig)(hipSharedMemConfig config) = nullptr;
    hipError_t (*hipSetDeviceFlags)(unsigned flags) = nullptr;
    hipError_t (*hipGetDeviceFlags)(unsigned int* flags) = nullptr;
    hipError_t (*hipDeviceCanAccessPeer)(int* canAccessPeer, int deviceId, int peerDeviceId) = nullptr;
    hipError_t (*hipDeviceEnablePeerAccess)(int peerDeviceId, unsigned int flags) = nullptr;
    hipError_t (*hipDeviceDisablePeerAccess)(int peerDeviceId) = nullptr;
    hipError_t (*hipChooseDevice)(int* device, const hipDeviceProp_t* prop) = nullptr;
    hipError_t (*hipDeviceGetPCIBusId)(char* pciBusId, int len, int device) = nullptr;
    hipError_t (*hipDeviceGetByPCIBusId)(int* device, const char* pciBusId) = nullptr;

    //=========================================================================
    // Error Handling
    //=========================================================================
    hipError_t (*hipGetLastError)() = nullptr;
    hipError_t (*hipPeekAtLastError)() = nullptr;
    const char* (*hipGetErrorName)(hipError_t hip_error) = nullptr;
    const char* (*hipGetErrorString)(hipError_t hip_error) = nullptr;

    //=========================================================================
    // Memory Management
    //=========================================================================
    hipError_t (*hipMalloc)(void** ptr, size_t size) = nullptr;
    hipError_t (*hipMallocPitch)(void** ptr, size_t* pitch, size_t width, size_t height) = nullptr;
    hipError_t (*hipMalloc3D)(hipPitchedPtr* pitchedDevPtr, hipExtent extent) = nullptr;
    hipError_t (*hipMallocArray)(hipArray_t* array, const hipChannelFormatDesc* desc, 
                                  size_t width, size_t height, unsigned int flags) = nullptr;
    hipError_t (*hipFree)(void* ptr) = nullptr;
    hipError_t (*hipFreeArray)(hipArray_t array) = nullptr;
    hipError_t (*hipHostMalloc)(void** ptr, size_t size, unsigned int flags) = nullptr;
    hipError_t (*hipHostFree)(void* ptr) = nullptr;
    hipError_t (*hipHostAlloc)(void** ptr, size_t size, unsigned int flags) = nullptr;
    hipError_t (*hipHostGetDevicePointer)(void** devPtr, void* hstPtr, unsigned int flags) = nullptr;
    hipError_t (*hipHostGetFlags)(unsigned int* flagsPtr, void* hostPtr) = nullptr;
    hipError_t (*hipHostRegister)(void* hostPtr, size_t sizeBytes, unsigned int flags) = nullptr;
    hipError_t (*hipHostUnregister)(void* hostPtr) = nullptr;
    hipError_t (*hipMallocManaged)(void** dev_ptr, size_t size, unsigned int flags) = nullptr;
    hipError_t (*hipMemPrefetchAsync)(const void* dev_ptr, size_t count, int device, 
                                       hipStream_t stream) = nullptr;
    hipError_t (*hipMemAdvise)(const void* dev_ptr, size_t count, hipMemoryAdvise advice, 
                                int device) = nullptr;
    hipError_t (*hipMemRangeGetAttribute)(void* data, size_t data_size, 
                                           hipMemRangeAttribute attribute,
                                           const void* dev_ptr, size_t count) = nullptr;
    hipError_t (*hipMemRangeGetAttributes)(void** data, size_t* data_sizes,
                                            hipMemRangeAttribute* attributes, size_t num_attributes,
                                            const void* dev_ptr, size_t count) = nullptr;
    hipError_t (*hipMemGetInfo)(size_t* free, size_t* total) = nullptr;
    hipError_t (*hipMemGetAddressRange)(hipDeviceptr_t* pbase, size_t* psize, 
                                         hipDeviceptr_t dptr) = nullptr;
    hipError_t (*hipPointerGetAttributes)(hipPointerAttribute_t* attributes, const void* ptr) = nullptr;

    //=========================================================================
    // Memory Copy Operations
    //=========================================================================
    hipError_t (*hipMemcpy)(void* dst, const void* src, size_t sizeBytes, 
                            hipMemcpyKind kind) = nullptr;
    hipError_t (*hipMemcpyWithStream)(void* dst, const void* src, size_t sizeBytes,
                                       hipMemcpyKind kind, hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpyHtoD)(hipDeviceptr_t dst, void* src, size_t sizeBytes) = nullptr;
    hipError_t (*hipMemcpyDtoH)(void* dst, hipDeviceptr_t src, size_t sizeBytes) = nullptr;
    hipError_t (*hipMemcpyDtoD)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) = nullptr;
    hipError_t (*hipMemcpyHtoDAsync)(hipDeviceptr_t dst, void* src, size_t sizeBytes, 
                                      hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpyDtoHAsync)(void* dst, hipDeviceptr_t src, size_t sizeBytes, 
                                      hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpyDtoDAsync)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, 
                                      hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpyAsync)(void* dst, const void* src, size_t sizeBytes,
                                  hipMemcpyKind kind, hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpy2D)(void* dst, size_t dpitch, const void* src, size_t spitch,
                               size_t width, size_t height, hipMemcpyKind kind) = nullptr;
    hipError_t (*hipMemcpy2DAsync)(void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind,
                                    hipStream_t stream) = nullptr;
    hipError_t (*hipMemcpy2DToArray)(hipArray_t dst, size_t wOffset, size_t hOffset,
                                      const void* src, size_t spitch, size_t width,
                                      size_t height, hipMemcpyKind kind) = nullptr;
    hipError_t (*hipMemcpy2DFromArray)(void* dst, size_t dpitch, hipArray_const_t src,
                                        size_t wOffset, size_t hOffset, size_t width,
                                        size_t height, hipMemcpyKind kind) = nullptr;
    hipError_t (*hipMemcpy3D)(const hipMemcpy3DParms* p) = nullptr;
    hipError_t (*hipMemcpy3DAsync)(const hipMemcpy3DParms* p, hipStream_t stream) = nullptr;

    //=========================================================================
    // Memory Set Operations  
    //=========================================================================
    hipError_t (*hipMemset)(void* dst, int value, size_t sizeBytes) = nullptr;
    hipError_t (*hipMemsetD8)(hipDeviceptr_t dest, unsigned char value, size_t count) = nullptr;
    hipError_t (*hipMemsetD16)(hipDeviceptr_t dest, unsigned short value, size_t count) = nullptr;
    hipError_t (*hipMemsetD32)(hipDeviceptr_t dest, int value, size_t count) = nullptr;
    hipError_t (*hipMemsetAsync)(void* dst, int value, size_t sizeBytes, 
                                  hipStream_t stream) = nullptr;
    hipError_t (*hipMemsetD8Async)(hipDeviceptr_t dest, unsigned char value, size_t count,
                                    hipStream_t stream) = nullptr;
    hipError_t (*hipMemsetD16Async)(hipDeviceptr_t dest, unsigned short value, size_t count,
                                     hipStream_t stream) = nullptr;
    hipError_t (*hipMemsetD32Async)(hipDeviceptr_t dest, int value, size_t count,
                                     hipStream_t stream) = nullptr;
    hipError_t (*hipMemset2D)(void* dst, size_t pitch, int value, size_t width, 
                               size_t height) = nullptr;
    hipError_t (*hipMemset2DAsync)(void* dst, size_t pitch, int value, size_t width,
                                    size_t height, hipStream_t stream) = nullptr;
    hipError_t (*hipMemset3D)(hipPitchedPtr pitchedDevPtr, int value, 
                               hipExtent extent) = nullptr;
    hipError_t (*hipMemset3DAsync)(hipPitchedPtr pitchedDevPtr, int value,
                                    hipExtent extent, hipStream_t stream) = nullptr;

    //=========================================================================
    // Stream Management
    //=========================================================================
    hipError_t (*hipStreamCreate)(hipStream_t* stream) = nullptr;
    hipError_t (*hipStreamCreateWithFlags)(hipStream_t* stream, unsigned int flags) = nullptr;
    hipError_t (*hipStreamCreateWithPriority)(hipStream_t* stream, unsigned int flags, 
                                               int priority) = nullptr;
    hipError_t (*hipStreamDestroy)(hipStream_t stream) = nullptr;
    hipError_t (*hipStreamQuery)(hipStream_t stream) = nullptr;
    hipError_t (*hipStreamSynchronize)(hipStream_t stream) = nullptr;
    hipError_t (*hipStreamWaitEvent)(hipStream_t stream, hipEvent_t event, 
                                      unsigned int flags) = nullptr;
    hipError_t (*hipStreamGetFlags)(hipStream_t stream, unsigned int* flags) = nullptr;
    hipError_t (*hipStreamGetPriority)(hipStream_t stream, int* priority) = nullptr;
    hipError_t (*hipStreamAddCallback)(hipStream_t stream, hipStreamCallback_t callback,
                                        void* userData, unsigned int flags) = nullptr;
    int (*hipGetStreamDeviceId)(hipStream_t stream) = nullptr;

    //=========================================================================
    // Event Management
    //=========================================================================
    hipError_t (*hipEventCreate)(hipEvent_t* event) = nullptr;
    hipError_t (*hipEventCreateWithFlags)(hipEvent_t* event, unsigned flags) = nullptr;
    hipError_t (*hipEventDestroy)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventRecord)(hipEvent_t event, hipStream_t stream) = nullptr;
    hipError_t (*hipEventQuery)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventSynchronize)(hipEvent_t event) = nullptr;
    hipError_t (*hipEventElapsedTime)(float* ms, hipEvent_t start, hipEvent_t stop) = nullptr;

    //=========================================================================
    // Execution Control
    //=========================================================================
    hipError_t (*hipFuncGetAttributes)(hipFuncAttributes* attr, const void* func) = nullptr;
    hipError_t (*hipFuncSetAttribute)(const void* func, hipFuncAttribute attr, int value) = nullptr;
    hipError_t (*hipFuncSetCacheConfig)(const void* func, hipFuncCache_t config) = nullptr;
    hipError_t (*hipFuncSetSharedMemConfig)(const void* func, hipSharedMemConfig config) = nullptr;
    hipError_t (*hipLaunchKernel)(const void* function_address, dim3 numBlocks, 
                                   dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                   hipStream_t stream) = nullptr;
    hipError_t (*hipLaunchCooperativeKernel)(const void* f, dim3 gridDim, dim3 blockDim,
                                              void** kernelParams, unsigned int sharedMemBytes,
                                              hipStream_t stream) = nullptr;
    hipError_t (*hipLaunchCooperativeKernelMultiDevice)(hipLaunchParams* launchParamsList,
                                                         int numDevices, unsigned int flags) = nullptr;
    hipError_t (*hipModuleLaunchKernel)(hipFunction_t f, unsigned int gridDimX,
                                         unsigned int gridDimY, unsigned int gridDimZ,
                                         unsigned int blockDimX, unsigned int blockDimY,
                                         unsigned int blockDimZ, unsigned int sharedMemBytes,
                                         hipStream_t stream, void** kernelParams,
                                         void** extra) = nullptr;
    hipError_t (*hipLaunchHostFunc)(hipStream_t stream, hipHostFn_t fn, void* userData) = nullptr;

    //=========================================================================
    // Module Management
    //=========================================================================
    hipError_t (*hipModuleLoad)(hipModule_t* module, const char* fname) = nullptr;
    hipError_t (*hipModuleLoadData)(hipModule_t* module, const void* image) = nullptr;
    hipError_t (*hipModuleLoadDataEx)(hipModule_t* module, const void* image,
                                       unsigned int numOptions, hipJitOption* options,
                                       void** optionValues) = nullptr;
    hipError_t (*hipModuleUnload)(hipModule_t module) = nullptr;
    hipError_t (*hipModuleGetFunction)(hipFunction_t* function, hipModule_t module,
                                        const char* kname) = nullptr;
    hipError_t (*hipModuleGetGlobal)(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                      const char* name) = nullptr;
    hipError_t (*hipModuleGetTexRef)(textureReference** texRef, hipModule_t hmod,
                                      const char* name) = nullptr;

    //=========================================================================
    // Occupancy
    //=========================================================================
    hipError_t (*hipOccupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks,
                                                                const void* f,
                                                                int blockSize,
                                                                size_t dynSharedMemPerBlk) = nullptr;
    hipError_t (*hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(int* numBlocks,
                                                                         const void* f,
                                                                         int blockSize,
                                                                         size_t dynSharedMemPerBlk,
                                                                         unsigned int flags) = nullptr;
    hipError_t (*hipOccupancyMaxPotentialBlockSize)(int* gridSize, int* blockSize,
                                                     const void* f, size_t dynSharedMemPerBlk,
                                                     int blockSizeLimit) = nullptr;

private:
    // Private constructor - called by instance()
    HipLoader();

    void initialize();
    void loadAllSymbols();

    template<typename T>
    void loadSymbol(const char* name, T& func);

    template<typename T>
    bool tryLoadSymbol(const char* name, T& func);

    void* handle_ = nullptr;
    std::string library_path_;
    
    // Static storage for library path (set before instance() is called)
    static std::string& pendingLibraryPath();
};

// Convenience function to get the HIP loader instance
inline HipLoader& hip() {
    return HipLoader::instance();
}
