// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ $(llvm-config --cxxflags) -I%S/../../../Poseidon %S/../../../Poseidon/Core/FPKind.cpp %s $(llvm-config --ldflags --libs support --system-libs) -o %t/matmul_scheme
// RUN: %t/matmul_scheme | FileCheck %s

// What a matrix-product candidate's scheme alone decides: the scheme and
// num_moduli field the launch descriptor stores for each library call
// candidate generation proposes, the significand bits the accuracy model is
// given for a split in either realization, and the label fragments.

#include "Techniques/Matmul/MatmulScheme.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace poseidon;

static const char *name(DispatchScheme s) {
  switch (s) {
  case DispatchScheme::OzakiII:
    return "OzakiII";
  case DispatchScheme::Tcec:
    return "Tcec";
  case DispatchScheme::Direct:
    return "Direct";
  }
  return "?";
}

static void library(const SchemeVariant &v) {
  outs() << "library " << schemeTag(v) << " [" << schemeParameter(v) << "] "
         << schemeFormats(v, '/') << ": descriptor "
         << name(descriptorScheme(v)) << " " << descriptorModulusField(v)
         << "\n";
}

static void split(const SchemeVariant &v) {
  outs() << "split " << schemeParameter(v) << " " << schemeFormats(v, '_')
         << ": " << splitCapturedBits(v) << " bits\n";
}

int main() {
  for (unsigned nm : kOzIIModuliCounts)
    library(SchemeVariant::modular(nm));
  library(SchemeVariant::native());
  library(SchemeVariant::split(FPKind::F16, FPKind::F32, 2));
  for (FPKind k : {FPKind::F16, FPKind::BF16, FPKind::TF32, FPKind::F32})
    library(SchemeVariant::direct(k, FPKind::F32));

  for (unsigned n : {2u, 3u, 5u})
    split(SchemeVariant::split(FPKind::F16, FPKind::F32, n));
  split(SchemeVariant::split(FPKind::BF16, FPKind::F32, 2));
  for (unsigned n : {2u, 3u, 5u})
    split(SchemeVariant::split(FPKind::TF32, FPKind::F32, n));
  return 0;
}

// CHECK:      library ozaki-ii [nm=8] s8/s32: descriptor OzakiII 8
// CHECK-NEXT: library ozaki-ii [nm=9] s8/s32: descriptor OzakiII 9
// CHECK-NEXT: library ozaki-ii [nm=10] s8/s32: descriptor OzakiII 10
// CHECK-NEXT: library ozaki-ii [nm=11] s8/s32: descriptor OzakiII 11
// CHECK-NEXT: library ozaki-ii [nm=12] s8/s32: descriptor OzakiII 12
// CHECK-NEXT: library ozaki-ii [nm=13] s8/s32: descriptor OzakiII 13
// CHECK-NEXT: library ozaki-ii [nm=14] s8/s32: descriptor OzakiII 14
// CHECK-NEXT: library native-dgemm [] f64/f64: descriptor OzakiII 0
// CHECK-NEXT: library tcec [n=2] f16/f32: descriptor Tcec 1
// CHECK-NEXT: library direct [] f16/f32: descriptor Direct 1
// CHECK-NEXT: library direct [] bf16/f32: descriptor Direct 2
// CHECK-NEXT: library direct [] tf32/f32: descriptor Direct 3
// CHECK-NEXT: library direct [] f32/f32: descriptor Direct 4

// The library split is the in-kernel n=2 FP16 split, so both are modelled
// with the same 22 bits.
// CHECK-NEXT: split n=2 f16_f32: 22 bits
// CHECK-NEXT: split n=3 f16_f32: 24 bits
// CHECK-NEXT: split n=5 f16_f32: 24 bits
// CHECK-NEXT: split n=2 bf16_f32: 16 bits
// CHECK-NEXT: split n=2 tf32_f32: 22 bits
// CHECK-NEXT: split n=3 tf32_f32: 24 bits
// CHECK-NEXT: split n=5 tf32_f32: 24 bits
