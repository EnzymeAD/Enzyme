; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; The function to differentiate is a select on a runtime condition and the
; result of the request is used. Every former use of the request must read
; the merged result of both branches.

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

define double @square(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @triple(double %x) {
  %r = fmul double %x, 3.000000e+00
  ret double %r
}

define double @differentiate(i1 %c, double %x) noinline {
  %fn = select i1 %c, ptr @square, ptr @triple
  %d = call double (...) @__enzyme_autodiff(ptr %fn, double %x)
  ret double %d
}

define double @differentiate_multiuse(i1 %c, double %x) noinline {
  %fn = select i1 %c, ptr @square, ptr @triple
  %d = call double (...) @__enzyme_autodiff(ptr %fn, double %x)
  %m = fmul double %d, %x
  %r = fadd double %d, %m
  ret double %r
}

define i32 @main() {
  %a = call double @differentiate(i1 true, double 3.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.str, double %a)
  %b = call double @differentiate(i1 false, double 3.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.str, double %b)
  %c = call double @differentiate_multiuse(i1 true, double 3.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.str, double %c)
  %d = call double @differentiate_multiuse(i1 false, double 3.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.str, double %d)
  ret i32 0
}

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

; CHECK: define double @differentiate(i1 %c, double %x)
; CHECK: br i1 %c, label %sel1, label %sel2
; CHECK: [[POST:[0-9a-z.]+]]:
; CHECK-NEXT: %[[MERGE:[0-9a-z.]+]] = phi double [ %[[T:[0-9a-z.]+]], %sel1 ], [ %[[F:[0-9a-z.]+]], %sel2 ]
; CHECK-NEXT: ret double %[[MERGE]]
; CHECK: sel1:
; CHECK-NEXT: %[[TS:[0-9a-z.]+]] = call {{.*}}{ double } @diffesquare(double %x, double 1.000000e+00)
; CHECK-NEXT: %[[T]] = extractvalue { double } %[[TS]], 0
; CHECK-NEXT: br label %[[POST]]
; CHECK: sel2:
; CHECK-NEXT: %[[FS:[0-9a-z.]+]] = call {{.*}}{ double } @diffetriple(double %x, double 1.000000e+00)
; CHECK-NEXT: %[[F]] = extractvalue { double } %[[FS]], 0
; CHECK-NEXT: br label %[[POST]]

; CHECK: define double @differentiate_multiuse(i1 %c, double %x)
; CHECK: br i1 %c, label %sel1, label %sel2
; CHECK: [[POST:[0-9a-z.]+]]:
; CHECK-NEXT: %[[MERGE:[0-9a-z.]+]] = phi double [ %[[T:[0-9a-z.]+]], %sel1 ], [ %[[F:[0-9a-z.]+]], %sel2 ]
; CHECK-NEXT: %m = fmul double %[[MERGE]], %x
; CHECK-NEXT: %r = fadd double %[[MERGE]], %m
; CHECK-NEXT: ret double %r
; CHECK: sel1:
; CHECK-NEXT: %[[TS:[0-9a-z.]+]] = call {{.*}}{ double } @diffesquare(double %x, double 1.000000e+00)
; CHECK-NEXT: %[[T]] = extractvalue { double } %[[TS]], 0
; CHECK-NEXT: br label %[[POST]]
; CHECK: sel2:
; CHECK-NEXT: %[[FS:[0-9a-z.]+]] = call {{.*}}{ double } @diffetriple(double %x, double 1.000000e+00)
; CHECK-NEXT: %[[F]] = extractvalue { double } %[[FS]], 0
; CHECK-NEXT: br label %[[POST]]

; EVAL: 6.000000
; EVAL-NEXT: 3.000000
; EVAL-NEXT: 24.000000
; EVAL-NEXT: 12.000000
