//=- MatmulScheme.h - The arithmetic of a matrix-product candidate --------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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

#include "FPKind.h"

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

// Captured significand bits of the N-slice Veltkamp split
// (emitNwayVeltkampSplit): slice 0 keeps p bits and each further slice adds p,
// since the residual is rounded to p bits and rescaled by the same factor; the
// SCALE only keeps the residual inside the slice format's exponent range,
// which the accuracy model charges through the slice format. The fill narrows
// F64 to F32 and the accumulator is F32, so the family is capped at 24 bits.
// p = 8 for BF16, 11 for F16/TF32 (verified by a bit-level simulation of the
// emitted split).
inline unsigned splitCapturedBits(const SchemeVariant &v) {
  unsigned sliceSigBits = (v.operand == FPKind::BF16) ? 8u : 11u;
  return std::min(sliceSigBits * sliceCount(v), 24u);
}

// __poseidon_tcec_dgemm's compute mode (runtime/tcec/tcec_rt.cu); its
// self-contained backend implements mode 1, FP16TCEC, a two-slice FP16 split
// with an FP32 accumulator.
inline unsigned splitLibraryComputeMode(const SchemeVariant &v) {
  if (v.scheme == Scheme::Split && v.operand == FPKind::F16 &&
      v.accumulator == FPKind::F32 && v.slices == 2)
    return 1;
  llvm::report_fatal_error(
      "no __poseidon_tcec_dgemm compute mode realizes this split");
}

// __poseidon_direct_dgemm's operand format (runtime/direct/direct_rt.cu), all
// with an FP32 accumulator; FP32 operands are cuBLAS's SGEMM rather than a
// tensor-core product.
inline unsigned directLibraryOperandFormat(const SchemeVariant &v) {
  if (v.scheme == Scheme::Direct && v.accumulator == FPKind::F32) {
    switch (v.operand) {
    case FPKind::F16:
      return 1;
    case FPKind::BF16:
      return 2;
    case FPKind::TF32:
      return 3;
    case FPKind::F32:
      return 4;
    default:
      break;
    }
  }
  llvm::report_fatal_error(
      "no __poseidon_direct_dgemm operand format realizes this product");
}

// The host runtime a library call goes to. All of them take the identical
// calling convention, so it selects only the callee name:
//   OzakiII -> __poseidon_ozaki_dgemm(_ex): INT8 residue/CRT, accuracy set by
//              num_moduli; num_moduli = 0 is cuBLAS's DGEMM;
//   Tcec    -> __poseidon_tcec_dgemm(_ex): FP16 tensor cores + error
//              correction, F32-class;
//   Direct  -> __poseidon_direct_dgemm(_ex): one rounding of each operand to
//              FP16/BF16/TF32/FP32, FP32 accumulator.
enum class DispatchScheme { OzakiII, Tcec, Direct };

// The scheme a launch descriptor names for a library call of `v`.
inline DispatchScheme descriptorScheme(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Modular:
  case Scheme::Native:
    return DispatchScheme::OzakiII;
  case Scheme::Split:
    return DispatchScheme::Tcec;
  case Scheme::Direct:
    return DispatchScheme::Direct;
  }
  llvm_unreachable("unknown Scheme");
}

// The launch descriptor's num_moduli field for a library call of `v`, which
// each runtime reads as its own parameter: the modulus count (0 selects the
// native product on the same entry), the TCEC compute mode, or the direct
// operand format.
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

// Label fragments of the scheme; each realization composes them with its own
// (matmulOptionLabel, ozakiIOptionLabel, the refusal messages, the cost-model
// classes).
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
