// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>

// Forward declarations of HIP types for dynamic loading
// These mirror the HIP runtime API types

typedef int hipDevice_t;
typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef void* hipDeviceptr_t;

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorMemoryAllocation = 2,
    hipErrorNotInitialized = 3,
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorMapBufferObjectFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    hipErrorInvalidResourceHandle = 400,
    hipErrorIllegalState = 401,
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled = 704,
    hipErrorPeerAccessNotEnabled = 705,
    hipErrorSetOnActiveProcess = 708,
    hipErrorContextIsDestroyed = 709,
    hipErrorAssert = 710,
    hipErrorHostMemoryAlreadyRegistered = 712,
    hipErrorHostMemoryNotRegistered = 713,
    hipErrorLaunchFailure = 719,
    hipErrorCooperativeLaunchTooLarge = 720,
    hipErrorNotSupported = 801,
    hipErrorStreamCaptureUnsupported = 900,
    hipErrorStreamCaptureInvalidated = 901,
    hipErrorStreamCaptureMerge = 902,
    hipErrorStreamCaptureUnmatched = 903,
    hipErrorStreamCaptureUnjoined = 904,
    hipErrorStreamCaptureIsolation = 905,
    hipErrorStreamCaptureImplicit = 906,
    hipErrorCapturedEvent = 907,
    hipErrorStreamCaptureWrongThread = 908,
    hipErrorGraphExecUpdateFailure = 910,
    hipErrorUnknown = 999,
    hipErrorRuntimeMemory = 1052,
    hipErrorRuntimeOther = 1053,
    hipErrorTbd
} hipError_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,
    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin,
    hipDeviceAttributeAccessPolicyMaxWindowSize,
    hipDeviceAttributeAsyncEngineCount,
    hipDeviceAttributeCanMapHostMemory,
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,
    hipDeviceAttributeClockRate,
    hipDeviceAttributeComputeMode,
    hipDeviceAttributeComputePreemptionSupported,
    hipDeviceAttributeConcurrentKernels,
    hipDeviceAttributeConcurrentManagedAccess,
    hipDeviceAttributeMaxBlocksPerMultiProcessor,
    hipDeviceAttributeCooperativeLaunch,
    hipDeviceAttributeCooperativeMultiDeviceLaunch,
    hipDeviceAttributeDeviceOverlap,
    hipDeviceAttributeDirectManagedMemAccessFromHost,
    hipDeviceAttributeGlobalL1CacheSupported,
    hipDeviceAttributeHostNativeAtomicSupported,
    hipDeviceAttributeIntegrated,
    hipDeviceAttributeIsMultiGpuBoard,
    hipDeviceAttributeKernelExecTimeout,
    hipDeviceAttributeL2CacheSize,
    hipDeviceAttributeLocalL1CacheSupported,
    hipDeviceAttributeLuid,
    hipDeviceAttributeLuidDeviceNodeMask,
    hipDeviceAttributeComputeCapabilityMajor,
    hipDeviceAttributeManagedMemory,
    hipDeviceAttributeMaxBlockDimX,
    hipDeviceAttributeMaxBlockDimY,
    hipDeviceAttributeMaxBlockDimZ,
    hipDeviceAttributeMaxGridDimX,
    hipDeviceAttributeMaxGridDimY,
    hipDeviceAttributeMaxGridDimZ,
    hipDeviceAttributeMaxSurface1D,
    hipDeviceAttributeMaxSurface1DLayered,
    hipDeviceAttributeMaxSurface2D,
    hipDeviceAttributeMaxSurface2DLayered,
    hipDeviceAttributeMaxSurface3D,
    hipDeviceAttributeMaxSurfaceCubemap,
    hipDeviceAttributeMaxSurfaceCubemapLayered,
    hipDeviceAttributeMaxTexture1DWidth,
    hipDeviceAttributeMaxTexture1DLayered,
    hipDeviceAttributeMaxTexture1DLinear,
    hipDeviceAttributeMaxTexture1DMipmap,
    hipDeviceAttributeMaxTexture2DWidth,
    hipDeviceAttributeMaxTexture2DHeight,
    hipDeviceAttributeMaxTexture2DGather,
    hipDeviceAttributeMaxTexture2DLayered,
    hipDeviceAttributeMaxTexture2DLinear,
    hipDeviceAttributeMaxTexture2DMipmap,
    hipDeviceAttributeMaxTexture3DWidth,
    hipDeviceAttributeMaxTexture3DHeight,
    hipDeviceAttributeMaxTexture3DDepth,
    hipDeviceAttributeMaxTexture3DAlt,
    hipDeviceAttributeMaxTextureCubemap,
    hipDeviceAttributeMaxTextureCubemapLayered,
    hipDeviceAttributeMaxThreadsDim,
    hipDeviceAttributeMaxThreadsPerBlock,
    hipDeviceAttributeMaxThreadsPerMultiProcessor,
    hipDeviceAttributeMaxPitch,
    hipDeviceAttributeMemoryBusWidth,
    hipDeviceAttributeMemoryClockRate,
    hipDeviceAttributeComputeCapabilityMinor,
    hipDeviceAttributeMultiGpuBoardGroupID,
    hipDeviceAttributeMultiprocessorCount,
    hipDeviceAttributeUnused1,
    hipDeviceAttributePageableMemoryAccess,
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
    hipDeviceAttributePciBusId,
    hipDeviceAttributePciDeviceId,
    hipDeviceAttributePciDomainID,
    hipDeviceAttributePersistingL2CacheMaxSize,
    hipDeviceAttributeMaxRegistersPerBlock,
    hipDeviceAttributeMaxRegistersPerMultiprocessor,
    hipDeviceAttributeReservedSharedMemPerBlock,
    hipDeviceAttributeMaxSharedMemoryPerBlock,
    hipDeviceAttributeSharedMemPerBlockOptin,
    hipDeviceAttributeSharedMemPerMultiprocessor,
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
    hipDeviceAttributeStreamPrioritiesSupported,
    hipDeviceAttributeSurfaceAlignment,
    hipDeviceAttributeTccDriver,
    hipDeviceAttributeTextureAlignment,
    hipDeviceAttributeTexturePitchAlignment,
    hipDeviceAttributeTotalConstantMemory,
    hipDeviceAttributeTotalGlobalMem,
    hipDeviceAttributeUnifiedAddressing,
    hipDeviceAttributeUnused2,
    hipDeviceAttributeWarpSize,
    hipDeviceAttributeMemoryPoolsSupported,
    hipDeviceAttributeVirtualMemoryManagementSupported,
    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,
    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,
    hipDeviceAttributeUnused3,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
    hipDeviceAttributeUnused4,
    hipDeviceAttributeUnused5,
    hipDeviceAttributeHdpMemFlushCntl,
    hipDeviceAttributeHdpRegFlushCntl,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
    hipDeviceAttributeIsLargeBar,
    hipDeviceAttributeAsicRevision,
    hipDeviceAttributeCanUseStreamWaitValue,
    hipDeviceAttributeImageSupport,
    hipDeviceAttributePhysicalMultiProcessorCount,
    hipDeviceAttributeFineGrainSupport,
    hipDeviceAttributeWallClockRate,
    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000
} hipDeviceAttribute_t;

typedef enum hipStreamFlags {
    hipStreamDefault = 0x00,
    hipStreamNonBlocking = 0x01
} hipStreamFlags;

typedef enum hipEventFlags {
    hipEventDefault = 0x0,
    hipEventBlockingSync = 0x1,
    hipEventDisableTiming = 0x2,
    hipEventInterprocess = 0x4
} hipEventFlags;

typedef enum hipHostMallocFlags {
    hipHostMallocDefault = 0x0,
    hipHostMallocPortable = 0x1,
    hipHostMallocMapped = 0x2,
    hipHostMallocWriteCombined = 0x4,
    hipHostMallocNumaUser = 0x20000000,
    hipHostMallocCoherent = 0x40000000,
    hipHostMallocNonCoherent = 0x80000000
} hipHostMallocFlags;

typedef enum hipHostRegisterFlags {
    hipHostRegisterDefault = 0x0,
    hipHostRegisterPortable = 0x1,
    hipHostRegisterMapped = 0x2,
    hipHostRegisterIoMemory = 0x4,
    hipHostRegisterReadOnly = 0x8
} hipHostRegisterFlags;

typedef enum hipMemoryType {
    hipMemoryTypeUnified = 0,
    hipMemoryTypeHost = 1,
    hipMemoryTypeDevice = 2,
    hipMemoryTypeManaged = 3
} hipMemoryType;

// UUID structure for device identification
struct hipUUID_t {
    unsigned char bytes[16];
};

// Architecture feature flags
struct hipDeviceArch_t {
    unsigned hasGlobalInt32Atomics : 1;
    unsigned hasGlobalFloatAtomicExch : 1;
    unsigned hasSharedInt32Atomics : 1;
    unsigned hasSharedFloatAtomicExch : 1;
    unsigned hasFloatAtomicAdd : 1;
    unsigned hasGlobalInt64Atomics : 1;
    unsigned hasSharedInt64Atomics : 1;
    unsigned hasDoubles : 1;
    unsigned hasWarpVote : 1;
    unsigned hasWarpBallot : 1;
    unsigned hasWarpShuffle : 1;
    unsigned hasFunnelShift : 1;
    unsigned hasThreadFenceSystem : 1;
    unsigned hasSyncThreadsExt : 1;
    unsigned hasSurfaceFuncs : 1;
    unsigned has3dGrid : 1;
    unsigned hasDynamicParallelism : 1;
};

struct hipDeviceProp_t {
    char name[256];
    hipUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseHipArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingHipArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved[63];
    int hipReserved[32];
    char gcnArchName[256];
    size_t maxSharedMemoryPerMultiProcessor;
    int clockInstructionRate;
    hipDeviceArch_t arch;
    unsigned int* hdpMemFlushCntl;
    unsigned int* hdpRegFlushCntl;
    int cooperativeMultiDeviceUnmatchedFunc;
    int cooperativeMultiDeviceUnmatchedGridDim;
    int cooperativeMultiDeviceUnmatchedBlockDim;
    int cooperativeMultiDeviceUnmatchedSharedMem;
    int isLargeBar;
    int asicRevision;
};

struct hipPointerAttribute_t {
    hipMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags;
};

// Dim3 structure
struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) 
        : x(_x), y(_y), z(_z) {}
};

// Stream callback type
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);

// Host function callback type
typedef void (*hipHostFn_t)(void* userData);

// Limit enum
typedef enum hipLimit_t {
    hipLimitStackSize = 0x00,
    hipLimitPrintfFifoSize = 0x01,
    hipLimitMallocHeapSize = 0x02,
    hipLimitRange
} hipLimit_t;

// Function cache configuration
typedef enum hipFuncCache_t {
    hipFuncCachePreferNone = 0,
    hipFuncCachePreferShared = 1,
    hipFuncCachePreferL1 = 2,
    hipFuncCachePreferEqual = 3
} hipFuncCache_t;

// Shared memory configuration
typedef enum hipSharedMemConfig {
    hipSharedMemBankSizeDefault = 0,
    hipSharedMemBankSizeFourByte = 1,
    hipSharedMemBankSizeEightByte = 2
} hipSharedMemConfig;

// Memory advise enum
typedef enum hipMemoryAdvise {
    hipMemAdviseSetReadMostly = 1,
    hipMemAdviseUnsetReadMostly = 2,
    hipMemAdviseSetPreferredLocation = 3,
    hipMemAdviseUnsetPreferredLocation = 4,
    hipMemAdviseSetAccessedBy = 5,
    hipMemAdviseUnsetAccessedBy = 6,
    hipMemAdviseSetCoarseGrain = 100,
    hipMemAdviseUnsetCoarseGrain = 101
} hipMemoryAdvise;

// Memory range attribute
typedef enum hipMemRangeAttribute {
    hipMemRangeAttributeReadMostly = 1,
    hipMemRangeAttributePreferredLocation = 2,
    hipMemRangeAttributeAccessedBy = 3,
    hipMemRangeAttributeLastPrefetchLocation = 4,
    hipMemRangeAttributeCoherencyMode = 100
} hipMemRangeAttribute;

// Managed memory attach flags for hipMallocManaged
#define hipMemAttachGlobal 0x01
#define hipMemAttachHost 0x02
#define hipMemAttachSingle 0x04

// Array types
typedef struct hipArray* hipArray_t;
typedef const struct hipArray* hipArray_const_t;

// Channel format description
struct hipChannelFormatDesc {
    int x;
    int y;
    int z;
    int w;
    enum hipChannelFormatKind {
        hipChannelFormatKindSigned = 0,
        hipChannelFormatKindUnsigned = 1,
        hipChannelFormatKindFloat = 2,
        hipChannelFormatKindNone = 3
    } f;
};

// Pitched pointer
struct hipPitchedPtr {
    void* ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};

// Extent
struct hipExtent {
    size_t width;
    size_t height;
    size_t depth;
};

// 3D memory copy parameters
struct hipMemcpy3DParms {
    hipArray_t srcArray;
    struct hipPos {
        size_t x, y, z;
    } srcPos;
    hipPitchedPtr srcPtr;
    hipArray_t dstArray;
    struct hipPos dstPos;
    hipPitchedPtr dstPtr;
    hipExtent extent;
    hipMemcpyKind kind;
};

// Function attributes
struct hipFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    size_t sharedSizeBytes;
};

// Function attribute enum
typedef enum hipFuncAttribute {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFuncAttribute;

// Launch parameters for multi-device launch
struct hipLaunchParams {
    void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    hipStream_t stream;
};

// JIT options
typedef enum hipJitOption {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock = 1,
    hipJitOptionWallTime = 2,
    hipJitOptionInfoLogBuffer = 3,
    hipJitOptionInfoLogBufferSizeBytes = 4,
    hipJitOptionErrorLogBuffer = 5,
    hipJitOptionErrorLogBufferSizeBytes = 6,
    hipJitOptionOptimizationLevel = 7,
    hipJitOptionTargetFromContext = 8,
    hipJitOptionTarget = 9,
    hipJitOptionFallbackStrategy = 10,
    hipJitOptionGenerateDebugInfo = 11,
    hipJitOptionLogVerbose = 12,
    hipJitOptionGenerateLineInfo = 13,
    hipJitOptionCacheMode = 14,
    hipJitOptionSm3xOpt = 15,
    hipJitOptionFastCompile = 16,
    hipJitOptionNumOptions
} hipJitOption;

// Texture reference (opaque)
struct textureReference;


