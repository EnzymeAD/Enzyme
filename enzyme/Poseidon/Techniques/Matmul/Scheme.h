//===- Scheme.h - The arithmetic of a matrix-product candidate ------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A matrix-product candidate is a scheme, the arithmetic it computes, realized
// either in the kernel or as a host library call. This header holds what
// follows from the scheme alone: its parameters, the values a library call's
// launch descriptor stores for it, and the fragments of its label.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_MATMUL_SCHEME_H
#define POSEIDON_TECHNIQUES_MATMUL_SCHEME_H

#include "Core/NumberFormat.h"
#include "runtime/gemm/SchemeConstants.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace poseidon {

/// The arithmetic a candidate computes the product with.
enum class Scheme {
  /// Each operand rounded once to the operand format, the products accumulated
  /// in the accumulator format.
  Direct,
  /// Each operand split into `slices` slices of the operand format whose cross
  /// products are accumulated apart and recombined: tensor-core error
  /// correction (TCEC).
  Tcec,
  /// Operands scaled to integers and multiplied as INT8 residues modulo
  /// `moduli` moduli, recombined by the CRT (Ozaki scheme II).
  OzakiII,
  /// The FP64 product.
  Native,
};

/// Where a scheme runs.
enum class Realization {
  /// Tensor-core code emitted in place of the reduction loop.
  InKernel,
  /// A host runtime call that replaces the launch, recorded in a launch
  /// descriptor.
  LibraryCall,
};

/// A scheme with its formats and its parameter.
struct SchemeVariant {
  /// The scheme.
  Scheme scheme = Scheme::Direct;
  /// The format operands, or their slices, are rounded to; INT8 for Ozaki-II.
  NumberFormat operand = NumberFormat::Invalid;
  /// The format products are accumulated in; INT32 for Ozaki-II.
  NumberFormat accumulator = NumberFormat::Invalid;
  unsigned slices = 0; // Tcec
  unsigned moduli = 0; // OzakiII

  /// The direct scheme on `operand` operands with an `accumulator` accumulator.
  static constexpr SchemeVariant direct(NumberFormat operand,
                                        NumberFormat accumulator) {
    return {Scheme::Direct, operand, accumulator, 0, 0};
  }
  /// The split scheme into `slices` slices of `operand`, accumulated in
  /// `accumulator`.
  static constexpr SchemeVariant
  tcec(NumberFormat operand, NumberFormat accumulator, unsigned slices) {
    return {Scheme::Tcec, operand, accumulator, slices, 0};
  }
  /// Ozaki-II with `moduli` moduli.
  static constexpr SchemeVariant ozakiII(unsigned moduli) {
    return {Scheme::OzakiII, NumberFormat::INT8, NumberFormat::INT32, 0,
            moduli};
  }
  /// The FP64 product.
  static constexpr SchemeVariant native() {
    return {Scheme::Native, NumberFormat::FP64, NumberFormat::FP64, 0, 0};
  }
};

/// The slice count of `v`, which must be a split variant.
inline unsigned sliceCount(const SchemeVariant &v) {
  assert(v.scheme == Scheme::Tcec && "only a split scheme has slices");
  return v.slices;
}

/// The modulus count of `v`, which must be an Ozaki-II variant.
inline unsigned modulusCount(const SchemeVariant &v) {
  assert(v.scheme == Scheme::OzakiII && "only a modular scheme has moduli");
  return v.moduli;
}

/// A format a split scheme slices an operand into. Each slice is the residual
/// left by the slices before it, multiplied by 2^scaleExponent and rounded to
/// nearest in the format, so it keeps `significandBits` bits of that residual;
/// the reconstruction divides the scale back out exactly. The scale therefore
/// sets no precision: it only keeps a residual, at most 2^-significandBits of
/// what it is cut from, inside the format's exponent range, which BF16 shares
/// with F32.
struct SliceFormat {
  /// The slice format.
  NumberFormat kind;
  /// The bits of its residual one slice keeps.
  unsigned significandBits;
  /// The power of two each residual is multiplied by before it is rounded.
  int scaleExponent;
};
/// The formats a split scheme slices into.
inline constexpr SliceFormat kSliceFormats[] = {
    {NumberFormat::FP16, 11, 11},
    {NumberFormat::BF16, 8, 7},
    {NumberFormat::TF32, 11, 11},
};

/// The slice format of `k`, which must be one of kSliceFormats.
inline const SliceFormat &sliceFormat(NumberFormat k) {
  for (const SliceFormat &f : kSliceFormats)
    if (f.kind == k)
      return f;
  llvm_unreachable("a split scheme slices only into a slice format");
}

/// Captured significand bits of the N-slice split: N times the bits of one
/// slice. The fill narrows F64 to F32 and the accumulator is F32, so the family
/// is capped at 24 bits.
inline unsigned splitCapturedBits(const SchemeVariant &v) {
  return std::min(sliceFormat(v.operand).significandBits * sliceCount(v), 24u);
}

// __poseidon_tcec_dgemm (runtime/gemm/Tcec.cu) runs on cuMpSGEMM and exists
// only in a build with it.
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
/// This build has __poseidon_tcec_dgemm.
inline constexpr bool kHaveTcecRuntime = true;
#else
/// This build has no __poseidon_tcec_dgemm.
inline constexpr bool kHaveTcecRuntime = false;
#endif

/// __poseidon_tcec_dgemm's compute mode: cuMpSGEMM's FP16TCEC is a two-slice
/// FP16 split with an FP32 accumulator.
inline PoseidonTcecMode splitLibraryComputeMode(const SchemeVariant &v) {
  if (v.scheme == Scheme::Tcec && v.operand == NumberFormat::FP16 &&
      v.accumulator == NumberFormat::FP32 && v.slices == 2)
    return kTcecFp16Tcec;
  llvm::report_fatal_error(
      "no __poseidon_tcec_dgemm compute mode realizes this split");
}

/// __poseidon_direct_dgemm's operand format (runtime/gemm/Direct.cu), all
/// with an FP32 accumulator; FP32 operands are cuBLAS's SGEMM rather than a
/// tensor-core product.
inline PoseidonDirectMode directLibraryOperandFormat(const SchemeVariant &v) {
  if (v.scheme == Scheme::Direct && v.accumulator == NumberFormat::FP32) {
    switch (v.operand) {
    case NumberFormat::FP16:
      return kDirectFp16;
    case NumberFormat::BF16:
      return kDirectBf16;
    case NumberFormat::TF32:
      return kDirectTf32;
    case NumberFormat::FP32:
      return kDirectFp32;
    default:
      break;
    }
  }
  llvm::report_fatal_error(
      "no __poseidon_direct_dgemm operand format realizes this product");
}

/// The launch descriptor's num_moduli field for a library call of `v`, the last
/// argument of its entry point: the modulus count, the TCEC compute mode, or
/// the direct operand format. The native entry has none, and the field is 0.
inline unsigned descriptorModulusField(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::OzakiII:
    return modulusCount(v);
  case Scheme::Native:
    return 0;
  case Scheme::Tcec:
    return splitLibraryComputeMode(v);
  case Scheme::Direct:
    return directLibraryOperandFormat(v);
  }
  llvm_unreachable("unknown Scheme");
}

/// Label fragments of the scheme, shared by both realizations.
inline std::string schemeFormats(const SchemeVariant &v, char separator) {
  return std::string(dtypeName(v.operand)) + separator +
         dtypeName(v.accumulator);
}

/// The scheme's word in candidate labels.
inline const char *schemeTag(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Direct:
    return "direct";
  case Scheme::Tcec:
    return "tcec";
  case Scheme::OzakiII:
    return "ozaki-ii";
  case Scheme::Native:
    return "native-dgemm";
  }
  llvm_unreachable("unknown Scheme");
}

/// The scheme's parameter in candidate labels, `n=<slices>` or `nm=<moduli>`;
/// empty for a scheme without one.
inline std::string schemeParameter(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Tcec:
    return "n=" + std::to_string(sliceCount(v));
  case Scheme::OzakiII:
    return "nm=" + std::to_string(modulusCount(v));
  case Scheme::Direct:
  case Scheme::Native:
    return "";
  }
  llvm_unreachable("unknown Scheme");
}

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_MATMUL_SCHEME_H
