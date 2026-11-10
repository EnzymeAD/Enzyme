// The state the two halves of the CUDA profiler share: the device half
// (FPProfilerDevice.cu, linked by the pass into every device module it
// profiles) updates it, the host half (FPProfilerCUDA.cu, a library the
// application links) allocates it and writes the profile from it.
#ifndef POSEIDON_FPPROFILER_CUDA_H
#define POSEIDON_FPPROFILER_CUDA_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__)
#define POSEIDON_HD __host__ __device__
#else
#define POSEIDON_HD
#endif

// Per-function slot range: bounds the static optimizable-instruction count of
// a single site (one slot per instruction, setPoseidonMetadata), not the
// dynamic execution count. Overflow aborts in poseidonLogValueCUDA; raise
// POSEIDON_PROFILE_MAX_SLOTS if a site ever needs more.
#define POSEIDON_DEFAULT_MAX_SLOTS 4096
#define POSEIDON_DEFAULT_MAX_FUNCS 512
// Device heap for the reverse passes' caches; POSEIDON_PROFILE_HEAP_MB.
#define POSEIDON_DEFAULT_HEAP_MB 4096
#define POSEIDON_MAX_OPERANDS 8

// Length of the per-function name slot the device registry copies a profiled
// function's mangled name into. A truncated name is not a cosmetic loss: the
// host would write the profile under the truncated name while the pass looks
// for it under the full one. 2048 covers every name observed with an order of
// magnitude of slack; overflow aborts in poseidonRegisterIdx rather than
// clipping.
#define POSEIDON_FUNCNAME_MAXLEN 2048

// Length of the per-function buffer the compile-time static data (the header
// lines Poseidon computed about the site and cannot recompute at profile-use)
// is published through. Overflow aborts in poseidonProfileStaticCUDA rather than
// writing a truncated header line.
#define POSEIDON_PROFILE_STATIC_MAXLEN 4096

// The device variable every profiled device module defines and the host half
// of the same translation unit registers.
#define POSEIDON_CUDA_PROFILE_TABLE "__poseidon_profile_table"

POSEIDON_HD inline uint64_t doubleToOrderedUint64(double val) {
  uint64_t bits;
  memcpy(&bits, &val, sizeof(bits));
  if (bits & 0x8000000000000000ULL)
    return ~bits;
  else
    return bits | 0x8000000000000000ULL;
}

POSEIDON_HD inline double orderedUint64ToDouble(uint64_t bits) {
  if (bits & 0x8000000000000000ULL)
    bits = bits & ~0x8000000000000000ULL;
  else
    bits = ~bits;
  double val;
  memcpy(&val, &bits, sizeof(val));
  return val;
}

// Every minimum in the profiler is stored complemented and folded with
// atomicMax, so that 0 is the identity for every field and the whole table can
// be initialized with a plain cudaMemset instead of a kernel launch from a
// static initializer, which in a multi-TU executable can run before the other
// translation units' __cudaRegisterFatBinary constructors. It also makes two
// records of one function merge field by field with max and +.
//
// 0 is a strict identity: ~ordered(v) == 0 only for a positive NaN, which every
// writer rejects first. Readback is orderedUint64ToDoubleMin; an unset field
// (stored 0, so ~0 == UINT64_MAX) decodes to the same sentinel value the
// .fpprofile readers expect, so the emitted text is unchanged.
POSEIDON_HD inline uint64_t doubleToOrderedUint64Min(double val) {
  return ~doubleToOrderedUint64(val);
}

POSEIDON_HD inline double orderedUint64ToDoubleMin(uint64_t stored) {
  return orderedUint64ToDouble(~stored);
}

struct GPUProfileSlot {
  uint64_t minRes;
  uint64_t maxRes;
  uint64_t minOperands[POSEIDON_MAX_OPERANDS];
  uint64_t maxOperands[POSEIDON_MAX_OPERANDS];
  uint64_t minMagOperands[POSEIDON_MAX_OPERANDS]; // smallest nonzero |operand|
  double sumValue;
  double sumGrad;
  // Sum of |gradient|. The signed sum is what the accuracy model weights with,
  // and it cancels for any value whose adjoint alternates in sign across
  // executions -- a sum-factorization contraction read by a derivative operator
  // does exactly that. The magnitude sum cannot cancel, so it is the honest
  // answer to "how much does the output move when this value moves".
  double sumAbsGrad;
  double sumSens;
  uint64_t exec;
  uint32_t numOperands;
};

// The device allocations every profiled device module records into. Without
// relocatable device code each module has its own copy of the device globals,
// so the handles are one constant per module, written by the host half into
// every module that registered one.
struct PoseidonCUDAProfileTable {
  GPUProfileSlot *slots;
  size_t slotCount;
  // Multi-function support: each wrapped kernel gets its own slot block and
  // output file. A probe's funcName is a distinct device-global string
  // constant, so its pointer is a race-free key (atomicCAS on the 64-bit
  // pointer). Slot block = slots[funcIdx*slotCount + idx]. The capacity is a
  // runtime quantity (POSEIDON_PROFILE_MAX_FUNCS): one fused framework
  // integrator can produce hundreds of profiled clones in one module. Overflow
  // aborts (poseidonRegisterIdx); it must never fold one function's atomics
  // into another function's slot block.
  int maxFuncs;
  const char **funcReg; // [maxFuncs]
  char *funcNames;      // [maxFuncs][POSEIDON_FUNCNAME_MAXLEN]
  // Per-function launch geometry: gridDim is captured in the value logger,
  // blockDim and the launch/CTA count in the entry probe
  // poseidonProfileBlockDimsCUDA. All three are per function: distinct clones
  // of one operator run under different block shapes, and a module-global max
  // would corrupt the matmul M/N reconstruction and the boundary-cast frequency
  // scale that read them. 0 = unknown (neutral).
  uint32_t *gridDimByFunc;                // [maxFuncs][3]
  uint32_t *blockDimByFunc;               // [maxFuncs][3]
  unsigned long long *launchCountByFunc;  // [maxFuncs]
  // Per-function compile-time static data, published by the site's own kernel
  // through the same pointer-keyed registry the names use and copied verbatim
  // into the .fpprofile header. [maxFuncs][POSEIDON_PROFILE_STATIC_MAXLEN]
  char *staticByFunc;
  // Condition-number probe (see poseidonProbePerturbCUDA): one site id per
  // run; eps == 0 disables the hook.
  double probeEps;
  int probeSite;
};

#endif // POSEIDON_FPPROFILER_CUDA_H
