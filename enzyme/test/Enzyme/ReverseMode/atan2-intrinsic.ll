; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -S -o %t.ll && %lli %t.ll; fi
; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -mtriple=nvptx64-nvidia-cuda -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -S | FileCheck %s; fi

; CUDA compilation can produce llvm.atan2 instead of a libdevice call.
; Check both argument adjoints, the return seed, and both precisions.

declare double @llvm.atan2.f64(double, double)
declare float @llvm.atan2.f32(float, float)
declare { double, double } @__enzyme_autodiff(ptr, ...)

define double @atan2_64(double %y, double %x) {
entry:
  %v = call double @llvm.atan2.f64(double %y, double %x)
  ret double %v
}

define { double, double } @gradient_64(double %y, double %x) {
  %d = call { double, double } (ptr, ...) @__enzyme_autodiff(ptr @atan2_64, double %y, double %x)
  ret { double, double } %d
}

; CHECK-LABEL: define internal { double, double } @diffeatan2_64(
; CHECK-NEXT: entry:
; CHECK-DAG: %[[XX64:.+]] = fmul fast double %x, %x
; CHECK-DAG: %[[YY64:.+]] = fmul fast double %y, %y
; CHECK-DAG: %[[DEN64:.+]] = fadd fast double %[[XX64]], %[[YY64]]
; CHECK-DAG: %[[SY64:.+]] = fmul fast double %differeturn, %x
; CHECK-DAG: %[[DY64:.+]] = fdiv fast double %[[SY64]], %[[DEN64]]
; CHECK-DAG: %[[SX64:.+]] = fmul fast double %differeturn, %y
; CHECK-DAG: %[[QX64:.+]] = fdiv fast double %[[SX64]], %[[DEN64]]
; CHECK-DAG: %[[DX64:.+]] = fneg fast double %[[QX64]]
; CHECK-NEXT: %[[R64:.+]] = insertvalue { double, double } undef, double %[[DY64]], 0
; CHECK-NEXT: %[[S64:.+]] = insertvalue { double, double } %[[R64]], double %[[DX64]], 1
; CHECK-NEXT: ret { double, double } %[[S64]]
; CHECK-NEXT: }

define float @atan2_32(float %y, float %x) {
entry:
  %v = call float @llvm.atan2.f32(float %y, float %x)
  ret float %v
}

define { float, float } @gradient_32(float %y, float %x) {
  %d = call { float, float } (ptr, ...) @__enzyme_autodiff(ptr @atan2_32, float %y, float %x)
  ret { float, float } %d
}

; CHECK-LABEL: define internal { float, float } @diffeatan2_32(
; CHECK-NEXT: entry:
; CHECK-DAG: %[[XX32:.+]] = fmul fast float %x, %x
; CHECK-DAG: %[[YY32:.+]] = fmul fast float %y, %y
; CHECK-DAG: %[[DEN32:.+]] = fadd fast float %[[XX32]], %[[YY32]]
; CHECK-DAG: %[[SY32:.+]] = fmul fast float %differeturn, %x
; CHECK-DAG: %[[DY32:.+]] = fdiv fast float %[[SY32]], %[[DEN32]]
; CHECK-DAG: %[[SX32:.+]] = fmul fast float %differeturn, %y
; CHECK-DAG: %[[QX32:.+]] = fdiv fast float %[[SX32]], %[[DEN32]]
; CHECK-DAG: %[[DX32:.+]] = fneg fast float %[[QX32]]
; CHECK-NEXT: %[[R32:.+]] = insertvalue { float, float } undef, float %[[DY32]], 0
; CHECK-NEXT: %[[S32:.+]] = insertvalue { float, float } %[[R32]], float %[[DX32]], 1
; CHECK-NEXT: ret { float, float } %[[S32]]
; CHECK-NEXT: }

; At (y, x) = (1, 2), seed 1 gives (dy, dx) = (0.4, -0.2).
define i32 @main() {
  %d64 = call { double, double } @gradient_64(double 1.0, double 2.0)
  %y64 = extractvalue { double, double } %d64, 0
  %x64 = extractvalue { double, double } %d64, 1
  %ylo64 = fcmp ogt double %y64, 3.990000e-01
  %yhi64 = fcmp olt double %y64, 4.010000e-01
  %xlo64 = fcmp ogt double %x64, -2.010000e-01
  %xhi64 = fcmp olt double %x64, -1.990000e-01
  %a = and i1 %ylo64, %yhi64
  %b = and i1 %xlo64, %xhi64
  %ok64 = and i1 %a, %b
  %d32 = call { float, float } @gradient_32(float 1.0, float 2.0)
  %y32 = extractvalue { float, float } %d32, 0
  %x32 = extractvalue { float, float } %d32, 1
  %yd = fpext float %y32 to double
  %xd = fpext float %x32 to double
  %ylo32 = fcmp ogt double %yd, 3.990000e-01
  %yhi32 = fcmp olt double %yd, 4.010000e-01
  %xlo32 = fcmp ogt double %xd, -2.010000e-01
  %xhi32 = fcmp olt double %xd, -1.990000e-01
  %c = and i1 %ylo32, %yhi32
  %d = and i1 %xlo32, %xhi32
  %ok32 = and i1 %c, %d
  %ok = and i1 %ok64, %ok32
  %status = select i1 %ok, i32 0, i32 1
  ret i32 %status
}
