// RUN: %poseidon_test_programs/poseidon-test-matmul-scheme | FileCheck %s

// What a matrix-product scheme alone decides: each library call's descriptor
// scheme and num_moduli fields, a split's modelled bits, and the labels.

#include "Techniques/Matmul/Scheme.h"
#include "runtime/gemm/SchemeConstants.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace poseidon;

static const char *name(Scheme s) {
  switch (s) {
  case Scheme::OzakiII:
    return "OzakiII";
  case Scheme::Tcec:
    return "Tcec";
  case Scheme::Direct:
    return "Direct";
  case Scheme::Native:
    return "Native";
  }
  return "?";
}

static void library(const SchemeVariant &v) {
  outs() << "library " << schemeTag(v) << " [" << schemeParameter(v) << "] "
         << schemeFormats(v, '/') << ": descriptor " << name(v.scheme) << " "
         << descriptorModulusField(v) << "\n";
}

static void split(const SchemeVariant &v) {
  outs() << "split " << schemeParameter(v) << " " << schemeFormats(v, '_')
         << ": " << splitCapturedBits(v) << " bits\n";
}

int main() {
  for (unsigned nm : kOzakiIIModuliCounts)
    library(SchemeVariant::ozakiII(nm));
  library(SchemeVariant::native());
  library(SchemeVariant::tcec(NumberFormat::FP16, NumberFormat::FP32, 2));
  for (NumberFormat k : {NumberFormat::FP16, NumberFormat::BF16, NumberFormat::TF32, NumberFormat::FP32})
    library(SchemeVariant::direct(k, NumberFormat::FP32));

  for (unsigned n : {2u, 3u, 5u})
    split(SchemeVariant::tcec(NumberFormat::FP16, NumberFormat::FP32, n));
  split(SchemeVariant::tcec(NumberFormat::BF16, NumberFormat::FP32, 2));
  for (unsigned n : {2u, 3u, 5u})
    split(SchemeVariant::tcec(NumberFormat::TF32, NumberFormat::FP32, n));
  return 0;
}

// CHECK:      library ozaki-ii [nm=8] s8/s32: descriptor OzakiII 8
// CHECK-NEXT: library ozaki-ii [nm=9] s8/s32: descriptor OzakiII 9
// CHECK-NEXT: library ozaki-ii [nm=10] s8/s32: descriptor OzakiII 10
// CHECK-NEXT: library ozaki-ii [nm=11] s8/s32: descriptor OzakiII 11
// CHECK-NEXT: library ozaki-ii [nm=12] s8/s32: descriptor OzakiII 12
// CHECK-NEXT: library ozaki-ii [nm=13] s8/s32: descriptor OzakiII 13
// CHECK-NEXT: library ozaki-ii [nm=14] s8/s32: descriptor OzakiII 14
// CHECK-NEXT: library native-dgemm [] f64/f64: descriptor Native 0
// CHECK-NEXT: library tcec [n=2] f16/f32: descriptor Tcec 1
// CHECK-NEXT: library direct [] f16/f32: descriptor Direct 1
// CHECK-NEXT: library direct [] bf16/f32: descriptor Direct 2
// CHECK-NEXT: library direct [] tf32/f32: descriptor Direct 3
// CHECK-NEXT: library direct [] f32/f32: descriptor Direct 4

// The library split is the in-kernel n=2 FP16 split: the same 22 bits.
// CHECK-NEXT: split n=2 f16_f32: 22 bits
// CHECK-NEXT: split n=3 f16_f32: 24 bits
// CHECK-NEXT: split n=5 f16_f32: 24 bits
// CHECK-NEXT: split n=2 bf16_f32: 16 bits
// CHECK-NEXT: split n=2 tf32_f32: 22 bits
// CHECK-NEXT: split n=3 tf32_f32: 24 bits
// CHECK-NEXT: split n=5 tf32_f32: 24 bits
