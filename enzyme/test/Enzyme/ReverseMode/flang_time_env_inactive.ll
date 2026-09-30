; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-global-activity -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s; fi

; The time, command line and environment queries of LLVM flang's runtime
; are inactive, also with -enzyme-global-activity (with which a call to an
; unknown function may read active memory):
;   tester(x) = x * CPU_TIME() * SYSTEM_CLOCK(COUNT_RATE=)

declare double @_FortranACpuTime()
declare i64 @_FortranASystemClockCountRate(i32)
declare void @_FortranAGetEnvVariable(ptr, ptr, ptr, ptr, i1, ptr, i32)

define double @tester(double %x) {
entry:
  %t = call double @_FortranACpuTime()
  %r = call i64 @_FortranASystemClockCountRate(i32 8)
  %rf = sitofp i64 %r to double
  %m = fmul double %x, %t
  %res = fmul double %m, %rf
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @tester, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(ptr, ...)

; CHECK: define internal { double } @diffetester(double %x, double %[[differet:.+]])
; CHECK: %t = call double @_FortranACpuTime()
; CHECK: %r = call i64 @_FortranASystemClockCountRate(i32 8)
; CHECK: %[[rf:.+]] = sitofp i64 %r to double
; CHECK: %[[d1:.+]] = fmul fast double %[[differet]], %[[rf]]
; CHECK: %[[d2:.+]] = fmul fast double %[[d1]], %t
; CHECK: insertvalue { double } {{(undef|poison)}}, double %[[d2]], 0
; CHECK-NOT: fakeaugmented
; CHECK-NOT: unreachable
