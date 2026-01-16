// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hip_loader.hpp"
#include <dlfcn.h>
#include <sstream>

// Static storage for pending library path
std::string& HipLoader::pendingLibraryPath() {
    static std::string path;
    return path;
}

void HipLoader::setLibraryPath(const std::string& library_path) {
    pendingLibraryPath() = library_path;
}

HipLoader& HipLoader::instance() {
    // Meyers singleton - thread-safe in C++11 and later
    static HipLoader instance;
    return instance;
}

HipLoader::HipLoader() {
    initialize();
}

HipLoader::~HipLoader() {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

void HipLoader::initialize() {
    // Use pending path if set, otherwise use default from config
    library_path_ = pendingLibraryPath().empty() 
        ? hip_cts::config::kDefaultHipLibrary 
        : pendingLibraryPath();
    
    // Try to load the library
    handle_ = dlopen(library_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    
    if (!handle_) {
        std::ostringstream oss;
        oss << "Failed to load HIP library '" << library_path_ << "': " << dlerror();
        throw HipLoaderError(oss.str());
    }

    // Clear any existing error
    dlerror();

    // Load all symbols
    loadAllSymbols();
}

template<typename T>
void HipLoader::loadSymbol(const char* name, T& func) {
    func = reinterpret_cast<T>(dlsym(handle_, name));
    if (!func) {
        std::ostringstream oss;
        oss << "Failed to load symbol '" << name << "': " << dlerror();
        throw HipLoaderError(oss.str());
    }
}

template<typename T>
bool HipLoader::tryLoadSymbol(const char* name, T& func) {
    dlerror(); // Clear any existing error
    func = reinterpret_cast<T>(dlsym(handle_, name));
    return func != nullptr;
}

void HipLoader::loadAllSymbols() {
    // Device Management - Required symbols
    loadSymbol("hipInit", hipInit);
    loadSymbol("hipGetDeviceCount", hipGetDeviceCount);
    loadSymbol("hipGetDevice", hipGetDevice);
    loadSymbol("hipSetDevice", hipSetDevice);
    loadSymbol("hipDeviceSynchronize", hipDeviceSynchronize);
    loadSymbol("hipDeviceReset", hipDeviceReset);
    loadSymbol("hipGetDeviceProperties", hipGetDeviceProperties);
    loadSymbol("hipDeviceGetAttribute", hipDeviceGetAttribute);

    // Device Management - Optional symbols
    tryLoadSymbol("hipDriverGetVersion", hipDriverGetVersion);
    tryLoadSymbol("hipRuntimeGetVersion", hipRuntimeGetVersion);
    tryLoadSymbol("hipDeviceGetLimit", hipDeviceGetLimit);
    tryLoadSymbol("hipDeviceSetLimit", hipDeviceSetLimit);
    tryLoadSymbol("hipDeviceGetCacheConfig", hipDeviceGetCacheConfig);
    tryLoadSymbol("hipDeviceSetCacheConfig", hipDeviceSetCacheConfig);
    tryLoadSymbol("hipDeviceGetSharedMemConfig", hipDeviceGetSharedMemConfig);
    tryLoadSymbol("hipDeviceSetSharedMemConfig", hipDeviceSetSharedMemConfig);
    tryLoadSymbol("hipSetDeviceFlags", hipSetDeviceFlags);
    tryLoadSymbol("hipGetDeviceFlags", hipGetDeviceFlags);
    tryLoadSymbol("hipDeviceCanAccessPeer", hipDeviceCanAccessPeer);
    tryLoadSymbol("hipDeviceEnablePeerAccess", hipDeviceEnablePeerAccess);
    tryLoadSymbol("hipDeviceDisablePeerAccess", hipDeviceDisablePeerAccess);
    tryLoadSymbol("hipChooseDevice", hipChooseDevice);
    tryLoadSymbol("hipDeviceGetPCIBusId", hipDeviceGetPCIBusId);
    tryLoadSymbol("hipDeviceGetByPCIBusId", hipDeviceGetByPCIBusId);

    // Error Handling - Required symbols
    loadSymbol("hipGetLastError", hipGetLastError);
    loadSymbol("hipPeekAtLastError", hipPeekAtLastError);
    loadSymbol("hipGetErrorName", hipGetErrorName);
    loadSymbol("hipGetErrorString", hipGetErrorString);

    // Memory Management - Required symbols
    loadSymbol("hipMalloc", hipMalloc);
    loadSymbol("hipFree", hipFree);
    loadSymbol("hipMemGetInfo", hipMemGetInfo);

    // Memory Management - Optional symbols
    tryLoadSymbol("hipMallocPitch", hipMallocPitch);
    tryLoadSymbol("hipMalloc3D", hipMalloc3D);
    tryLoadSymbol("hipMallocArray", hipMallocArray);
    tryLoadSymbol("hipFreeArray", hipFreeArray);
    tryLoadSymbol("hipHostMalloc", hipHostMalloc);
    tryLoadSymbol("hipHostFree", hipHostFree);
    tryLoadSymbol("hipHostAlloc", hipHostAlloc);
    tryLoadSymbol("hipHostGetDevicePointer", hipHostGetDevicePointer);
    tryLoadSymbol("hipHostGetFlags", hipHostGetFlags);
    tryLoadSymbol("hipHostRegister", hipHostRegister);
    tryLoadSymbol("hipHostUnregister", hipHostUnregister);
    tryLoadSymbol("hipMallocManaged", hipMallocManaged);
    tryLoadSymbol("hipMemPrefetchAsync", hipMemPrefetchAsync);
    tryLoadSymbol("hipMemAdvise", hipMemAdvise);
    tryLoadSymbol("hipMemRangeGetAttribute", hipMemRangeGetAttribute);
    tryLoadSymbol("hipMemRangeGetAttributes", hipMemRangeGetAttributes);
    tryLoadSymbol("hipMemGetAddressRange", hipMemGetAddressRange);
    tryLoadSymbol("hipPointerGetAttributes", hipPointerGetAttributes);

    // Memory Copy Operations - Required symbols
    loadSymbol("hipMemcpy", hipMemcpy);
    loadSymbol("hipMemcpyAsync", hipMemcpyAsync);

    // Memory Copy Operations - Optional symbols
    tryLoadSymbol("hipMemcpyWithStream", hipMemcpyWithStream);
    tryLoadSymbol("hipMemcpyHtoD", hipMemcpyHtoD);
    tryLoadSymbol("hipMemcpyDtoH", hipMemcpyDtoH);
    tryLoadSymbol("hipMemcpyDtoD", hipMemcpyDtoD);
    tryLoadSymbol("hipMemcpyHtoDAsync", hipMemcpyHtoDAsync);
    tryLoadSymbol("hipMemcpyDtoHAsync", hipMemcpyDtoHAsync);
    tryLoadSymbol("hipMemcpyDtoDAsync", hipMemcpyDtoDAsync);
    tryLoadSymbol("hipMemcpy2D", hipMemcpy2D);
    tryLoadSymbol("hipMemcpy2DAsync", hipMemcpy2DAsync);
    tryLoadSymbol("hipMemcpy2DToArray", hipMemcpy2DToArray);
    tryLoadSymbol("hipMemcpy2DFromArray", hipMemcpy2DFromArray);
    tryLoadSymbol("hipMemcpy3D", hipMemcpy3D);
    tryLoadSymbol("hipMemcpy3DAsync", hipMemcpy3DAsync);

    // Memory Set Operations - Required symbols
    loadSymbol("hipMemset", hipMemset);
    loadSymbol("hipMemsetAsync", hipMemsetAsync);

    // Memory Set Operations - Optional symbols
    tryLoadSymbol("hipMemsetD8", hipMemsetD8);
    tryLoadSymbol("hipMemsetD16", hipMemsetD16);
    tryLoadSymbol("hipMemsetD32", hipMemsetD32);
    tryLoadSymbol("hipMemsetD8Async", hipMemsetD8Async);
    tryLoadSymbol("hipMemsetD16Async", hipMemsetD16Async);
    tryLoadSymbol("hipMemsetD32Async", hipMemsetD32Async);
    tryLoadSymbol("hipMemset2D", hipMemset2D);
    tryLoadSymbol("hipMemset2DAsync", hipMemset2DAsync);
    tryLoadSymbol("hipMemset3D", hipMemset3D);
    tryLoadSymbol("hipMemset3DAsync", hipMemset3DAsync);

    // Stream Management - Required symbols
    loadSymbol("hipStreamCreate", hipStreamCreate);
    loadSymbol("hipStreamDestroy", hipStreamDestroy);
    loadSymbol("hipStreamSynchronize", hipStreamSynchronize);

    // Stream Management - Optional symbols
    tryLoadSymbol("hipStreamCreateWithFlags", hipStreamCreateWithFlags);
    tryLoadSymbol("hipStreamCreateWithPriority", hipStreamCreateWithPriority);
    tryLoadSymbol("hipStreamQuery", hipStreamQuery);
    tryLoadSymbol("hipStreamWaitEvent", hipStreamWaitEvent);
    tryLoadSymbol("hipStreamGetFlags", hipStreamGetFlags);
    tryLoadSymbol("hipStreamGetPriority", hipStreamGetPriority);
    tryLoadSymbol("hipStreamAddCallback", hipStreamAddCallback);

    // Event Management - Required symbols
    loadSymbol("hipEventCreate", hipEventCreate);
    loadSymbol("hipEventDestroy", hipEventDestroy);
    loadSymbol("hipEventRecord", hipEventRecord);
    loadSymbol("hipEventSynchronize", hipEventSynchronize);

    // Event Management - Optional symbols
    tryLoadSymbol("hipEventCreateWithFlags", hipEventCreateWithFlags);
    tryLoadSymbol("hipEventQuery", hipEventQuery);
    tryLoadSymbol("hipEventElapsedTime", hipEventElapsedTime);

    // Execution Control - Optional symbols
    tryLoadSymbol("hipFuncGetAttributes", hipFuncGetAttributes);
    tryLoadSymbol("hipFuncSetAttribute", hipFuncSetAttribute);
    tryLoadSymbol("hipFuncSetCacheConfig", hipFuncSetCacheConfig);
    tryLoadSymbol("hipFuncSetSharedMemConfig", hipFuncSetSharedMemConfig);
    tryLoadSymbol("hipLaunchKernel", hipLaunchKernel);
    tryLoadSymbol("hipLaunchCooperativeKernel", hipLaunchCooperativeKernel);
    tryLoadSymbol("hipLaunchCooperativeKernelMultiDevice", hipLaunchCooperativeKernelMultiDevice);
    tryLoadSymbol("hipModuleLaunchKernel", hipModuleLaunchKernel);
    tryLoadSymbol("hipLaunchHostFunc", hipLaunchHostFunc);

    // Module Management - Optional symbols
    tryLoadSymbol("hipModuleLoad", hipModuleLoad);
    tryLoadSymbol("hipModuleLoadData", hipModuleLoadData);
    tryLoadSymbol("hipModuleLoadDataEx", hipModuleLoadDataEx);
    tryLoadSymbol("hipModuleUnload", hipModuleUnload);
    tryLoadSymbol("hipModuleGetFunction", hipModuleGetFunction);
    tryLoadSymbol("hipModuleGetGlobal", hipModuleGetGlobal);
    tryLoadSymbol("hipModuleGetTexRef", hipModuleGetTexRef);

    // Occupancy - Optional symbols
    tryLoadSymbol("hipOccupancyMaxActiveBlocksPerMultiprocessor", 
                  hipOccupancyMaxActiveBlocksPerMultiprocessor);
    tryLoadSymbol("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                  hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    tryLoadSymbol("hipOccupancyMaxPotentialBlockSize", hipOccupancyMaxPotentialBlockSize);
}
