; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

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

define { float, float } @test_derivative(float %a, float %b) {
entry:
  %0 = tail call { float, float } (...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %a, float %b)
  ret { float, float } %0
}

declare { float, float } @__enzyme_autodiff(...)

; CHECK: define internal { float, float } @diffetester(float %a, float %b, float %differeturn)
; CHECK:   %[[hi:.+]] = and i64 %[[d:.+]], -4294967296
; CHECK-NEXT:   %[[lo:.+]] = and i64 %[[d]], 4294967295
; CHECK-NEXT:   %[[lot:.+]] = trunc i64 %[[lo]] to i32
; CHECK-NEXT:   %[[his:.+]] = lshr i64 %[[hi]], 32
; CHECK-NEXT:   %[[hit:.+]] = trunc i64 %[[his]] to i32
; CHECK-NEXT:   %[[db:.+]] = bitcast i32 %[[hit]] to float
; CHECK-NEXT:   %[[da:.+]] = bitcast i32 %[[lot]] to float
; CHECK-NEXT:   %[[r0:.+]] = insertvalue { float, float } undef, float %[[da]], 0
; CHECK-NEXT:   %[[r1:.+]] = insertvalue { float, float } %[[r0]], float %[[db]], 1
; CHECK-NEXT:   ret { float, float } %[[r1]]
