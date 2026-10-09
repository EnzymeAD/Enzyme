; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Two floats packed into one i64 (as SROA does for a pair of floats), then unpacked.
define float @tester(float %a, float %b) {
entry:
  %ai = bitcast float %a to i32
  %bi = bitcast float %b to i32
  %bext = zext i32 %bi to i64
  %bshl = shl nuw i64 %bext, 32
  %aext = zext i32 %ai to i64
  %pack = or disjoint i64 %bshl, %aext
  %lo = trunc i64 %pack to i32
  %lof = bitcast i32 %lo to float
  %hish = lshr i64 %pack, 32
  %hi = trunc i64 %hish to i32
  %hif = bitcast i32 %hi to float
  %m = fmul float %lof, %hif
  ret float %m
}

define float @test_derivative(float %a, float %b) {
entry:
  %0 = tail call float (...) @__enzyme_fwddiff(float (float, float)* nonnull @tester, float %a, float 1.0, float %b, float 0.0)
  ret float %0
}

declare float @__enzyme_fwddiff(...)

; CHECK: define internal float @fwddiffetester(float %a, float %"a'", float %b, float %"b'")
; CHECK:   %[[dbext:.+]] = zext i32 %{{.*}} to i64
; CHECK:   %[[dbshl:.+]] = shl i64 %[[dbext]], 32
; CHECK:   %[[daext:.+]] = zext i32 %{{.*}} to i64
; CHECK:   %[[dpack:.+]] = or i64 %[[dbshl]], %[[daext]]
; CHECK:   %{{.*}} = lshr i64 %[[dpack]], 32
