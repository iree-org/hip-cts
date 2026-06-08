# No standalone hip-cts reproducer for the raw-HSACO "truncated implicit kernarg suffix" bug

**Bug:** HRX (the streaming HIP binding) segfaulted on the first `conv2d` in ResNet
training on a non-default-ordinal-clean build.
**Fix:** ROCm/hrx-system branch `users/zjgarvey/fix/raw-hsaco-implicit-kernarg-suffix`.
**Practical reproducer:** ROCm/hrx-pytorch-smoke-test `repro/conv2d_min.py` (needs torch + MIOpen).

This note exists because a *dependency-free* hip-cts reproducer is not practical;
the reasoning is below.

## Root cause (verified via gdb on `conv2d_min.py`)

The amdgpu raw-HSACO loader,
`runtime/src/iree/hal/drivers/amdgpu/executable.c` →
`iree_hal_amdgpu_executable_raw_hsaco_custom_kernarg_layout()`, rejected any kernel
whose `kernarg_segment_size` is smaller than
`implicit_args_offset + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE` (the *full* implicit/
hidden-args block) with:

```
INVALID_ARGUMENT; AMDGPU kernel `miopenSp3AsmConv_v30_3_1_gfx9_fp32_f3x2_stride1.kd`
                  has truncated implicit kernarg suffix     (executable.c:2096)
```

MIOpen ignores that load failure and then calls `hipModuleUnload()` with the
uninitialized module handle, which the binding dereferences in
`iree_hal_streaming_module_release` → **SIGSEGV**. The fix relaxes the validation
(the custom-direct dispatch path already over-reserves and zero-fills, so a partial
suffix is safe).

The failing kernel is `miopenSp3AsmConv_*` — a **hand-written assembly** convolution
kernel.

## Why there's no simple, dependency-free CTS reproducer

The validation only rejects kernels with a **partial** implicit-args suffix: a hidden
arg is declared, but `kernarg_segment_size` does not span the full implicit-args block.

- **Normal compiled HIP kernels cannot produce this.** Per the AMDGPU code-object ABI,
  clang/hipcc reserve the **full** implicit-args block in `kernarg_segment_size`
  regardless of how many implicit args the kernel uses. So any kernel a CTS can compile
  from `.hip` (the `kernel_smoke` pattern) passes this validation — it never reaches the
  rejected branch. The partial suffix is characteristic of **hand-written assembly**
  kernels that declare only the implicit args they use, which is exactly what MIOpen's
  `*AsmConv*` kernels do.

- Producing a triggering code object **without MIOpen** requires one of:
  1. **Hand-authoring a gfx-specific AMDGPU assembly kernel** plus its code-object
     msgpack metadata, with a declared hidden arg and a deliberately-undersized
     `.kernarg_segment_size`. This is architecture-specific and brittle, and hip-cts has
     no `.s`-assembly kernel build path today (kernels go `.hip` → hipcc, which won't
     emit a truncated suffix).
  2. **Binary-patching a compiled kernel's metadata note** to shrink
     `kernarg_segment_size` — intricate and fragile.
  3. **Capturing and embedding a real MIOpen `*AsmConv*` HSACO** — needs MIOpen (the
     heavy external dependency we are trying to avoid) and ships an opaque vendor binary.

None is a simple, self-contained CTS test, so per the task this note stands in for the
reproducer. The torch+MIOpen `conv2d_min.py` remains the practical repro.

## Verifying the fix

Build the binding from `users/zjgarvey/fix/raw-hsaco-implicit-kernarg-suffix` and either
run `conv2d_min.py` (PASS) or the full native-vs-HRX A/B smoke (58/58). To run hip-cts
against an HRX binding, set `LD_LIBRARY_PATH` to the rocm-sdk lib dirs
(`_rocm_sdk_core/lib:_rocm_sdk_libraries/lib`).

## Aside — a *different* bug found while probing (not this one)

Building hip-cts `kernel_smoke` (normal compiled kernels, loaded via
`hipModuleLoadData`) and running it against **stock** hrx-system main fails earlier, in
the offload-bundle parser:

```
fat_binary.c:389: INVALID_ARGUMENT; offload bundle entry[0] out of range
```

This is the length-unknown `hipModuleLoadData` path that Andrew's #45 ("Fix HIP module
load data bundle parsing") addresses; the nightly feature branch's `fat_binary.c`
changes already handle it (`kernel_smoke` passes there). It is unrelated to the
kernarg-suffix bug above, but `kernel_smoke` would make a good regression test for *that*
issue once a bundle fix lands.
