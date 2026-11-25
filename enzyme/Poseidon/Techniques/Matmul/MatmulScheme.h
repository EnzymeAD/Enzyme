//===- MatmulScheme.h - The arithmetic of a matrix-product candidate ------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A matrix-product candidate is a scheme, the arithmetic it computes, realized
// either in the kernel or as a host library call. This header holds what
// follows from the scheme alone, for both realizations: its parameters, the
// values a library call's launch descriptor stores for it, and the fragments
// of its label.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMUL_SCHEME_H
#define POSEIDON_MATMUL_SCHEME_H

#include "Core/FPKind.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <string>

namespace poseidon {

enum class Scheme {
  // Each operand rounded once to the operand format, the products accumulated
  // in the accumulator format.
  Direct,
  // Each operand split into `slices` slices of the operand format whose cross
  // products are accumulated apart and recombined: tensor-core error
  // correction (TCEC).
  Split,
  // Operands scaled to integers and multiplied as INT8 residues modulo
  // `moduli` moduli, recombined by the CRT (Ozaki scheme II).
  Modular,
  // The FP64 product.
  Native,
};

enum class Realization {
  // Tensor-core code emitted in place of the reduction loop.
  InKernel,
  // A host runtime call that replaces the launch, recorded in a launch
  // descriptor.
  LibraryCall,
};

struct SchemeVariant {
  Scheme scheme = Scheme::Direct;
  FPKind operand = FPKind::Invalid;
  FPKind accumulator = FPKind::Invalid;
  unsigned slices = 0; // Split
  unsigned moduli = 0; // Modular

  static constexpr SchemeVariant direct(FPKind operand, FPKind accumulator) {
    return {Scheme::Direct, operand, accumulator, 0, 0};
  }
  static constexpr SchemeVariant split(FPKind operand, FPKind accumulator,
                                       unsigned slices) {
    return {Scheme::Split, operand, accumulator, slices, 0};
  }
  static constexpr SchemeVariant modular(unsigned moduli) {
    return {Scheme::Modular, FPKind::S8, FPKind::S32, 0, moduli};
  }
  static constexpr SchemeVariant native() {
    return {Scheme::Native, FPKind::F64, FPKind::F64, 0, 0};
  }
};

inline unsigned sliceCount(const SchemeVariant &v) {
  if (v.scheme != Scheme::Split)
    llvm::report_fatal_error("sliceCount: not a split scheme");
  return v.slices;
}

inline unsigned modulusCount(const SchemeVariant &v) {
  if (v.scheme != Scheme::Modular)
    llvm::report_fatal_error("modulusCount: not a modular scheme");
  return v.moduli;
}

// A format a split scheme slices an operand into. Each slice is the residual
// left by the slices before it, multiplied by 2^scaleExponent and rounded to
// nearest in the format, so it keeps `significandBits` bits of that residual;
// the reconstruction divides the scale back out exactly. The scale therefore
// sets no precision: it only keeps a residual, at most 2^-significandBits of
// what it is cut from, inside the format's exponent range, which BF16 shares
// with F32.
struct SliceFormat {
  FPKind kind;
  unsigned significandBits;
  int scaleExponent;
};
inline constexpr SliceFormat kSliceFormats[] = {
    {FPKind::F16, 11, 11},
    {FPKind::BF16, 8, 7},
    {FPKind::TF32, 11, 11},
};

inline const SliceFormat &sliceFormat(FPKind k) {
  for (const SliceFormat &f : kSliceFormats)
    if (f.kind == k)
      return f;
  llvm::report_fatal_error("sliceFormat: not a slice format");
}

// Captured significand bits of the N-slice split: N times the bits of one
// slice. The fill narrows F64 to F32 and the accumulator is F32, so the family
// is capped at 24 bits.
inline unsigned splitCapturedBits(const SchemeVariant &v) {
  return std::min(sliceFormat(v.operand).significandBits * sliceCount(v), 24u);
}

// Whether this compiler has __poseidon_tcec_dgemm (runtime/tcec/tcec_rt.cu),
// which runs on cuMpSGEMM and exists only in a build with it.
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
inline constexpr bool kHaveTcecDispatch = true;
#else
inline constexpr bool kHaveTcecDispatch = false;
#endif

// __poseidon_tcec_dgemm's compute mode: cuMpSGEMM's FP16TCEC is a two-slice
// FP16 split with an FP32 accumulator.
inline PoseidonTcecMode splitLibraryComputeMode(const SchemeVariant &v) {
  if (v.scheme == Scheme::Split && v.operand == FPKind::F16 &&
      v.accumulator == FPKind::F32 && v.slices == 2)
    return kTcecFp16Tcec;
  llvm::report_fatal_error(
      "no __poseidon_tcec_dgemm compute mode realizes this split");
}

// __poseidon_direct_dgemm's operand format (runtime/direct/direct_rt.cu), all
// with an FP32 accumulator; FP32 operands are cuBLAS's SGEMM rather than a
// tensor-core product.
inline PoseidonDirectMode directLibraryOperandFormat(const SchemeVariant &v) {
  if (v.scheme == Scheme::Direct && v.accumulator == FPKind::F32) {
    switch (v.operand) {
    case FPKind::F16:
      return kDirectFp16;
    case FPKind::BF16:
      return kDirectBf16;
    case FPKind::TF32:
      return kDirectTf32;
    case FPKind::F32:
      return kDirectFp32;
    default:
      break;
    }
  }
  llvm::report_fatal_error(
      "no __poseidon_direct_dgemm operand format realizes this product");
}

// The host runtime a library call goes to (runtime/GemmDispatch.h):
//   OzakiII -> __poseidon_ozaki_dgemm: INT8 residue/CRT, accuracy set by
//              num_moduli;
//   Tcec    -> __poseidon_tcec_dgemm: FP16 tensor cores + error correction,
//              F32-class;
//   Direct  -> __poseidon_direct_dgemm: one rounding of each operand to
//              FP16/BF16/TF32/FP32, FP32 accumulator;
//   Native  -> __poseidon_native_dgemm: the library's DGEMM.
enum class DispatchScheme { OzakiII, Tcec, Direct, Native };

// The scheme a launch descriptor names for a library call of `v`.
inline DispatchScheme descriptorScheme(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Modular:
    return DispatchScheme::OzakiII;
  case Scheme::Native:
    return DispatchScheme::Native;
  case Scheme::Split:
    return DispatchScheme::Tcec;
  case Scheme::Direct:
    return DispatchScheme::Direct;
  }
  llvm_unreachable("unknown Scheme");
}

// The launch descriptor's num_moduli field for a library call of `v`, the last
// argument of its entry point: the modulus count, the TCEC compute mode, or
// the direct operand format. The native entry has none, and the field is 0.
inline unsigned descriptorModulusField(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Modular:
    return modulusCount(v);
  case Scheme::Native:
    return 0;
  case Scheme::Split:
    return splitLibraryComputeMode(v);
  case Scheme::Direct:
    return directLibraryOperandFormat(v);
  }
  llvm_unreachable("unknown Scheme");
}

// Label fragments of the scheme, shared by both realizations.
inline std::string schemeFormats(const SchemeVariant &v, char separator) {
  return std::string(fpKindName(v.operand)) + separator +
         fpKindName(v.accumulator);
}

inline const char *schemeTag(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Direct:
    return "direct";
  case Scheme::Split:
    return "tcec";
  case Scheme::Modular:
    return "ozaki-ii";
  case Scheme::Native:
    return "native-dgemm";
  }
  llvm_unreachable("unknown Scheme");
}

inline std::string schemeParameter(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Split:
    return "n=" + std::to_string(sliceCount(v));
  case Scheme::Modular:
    return "nm=" + std::to_string(modulusCount(v));
  case Scheme::Direct:
  case Scheme::Native:
    return "";
  }
  llvm_unreachable("unknown Scheme");
}

} // namespace poseidon
#endif // POSEIDON_MATMUL_SCHEME_H
