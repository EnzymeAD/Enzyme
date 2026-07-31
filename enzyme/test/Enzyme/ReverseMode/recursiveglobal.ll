; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Constant globals whose initializers refer back to themselves. Type analysis
; must not recurse infinitely analyzing such an initializer.

; A relative pointer jump table, as emitted when lowering a switch over several
; constant tables. Nothing can be said about what it points at.
@tab = constant [1 x i64] [i64 sub (i64 0, i64 ptrtoint ([1 x i64]* @tab to i64))]

; A self referential struct which also holds a double. Breaking the cycle must
; not discard the rest of the initializer: the pointer at offset 0 and the
; double at offset 8 are both still deduced.
@mixed = constant { i8*, double } { i8* bitcast ({ i8*, double }* @mixed to i8*), double 3.000000e+00 }

define double @f(double %x) {
  %v = load double, double* bitcast ([1 x i64]* @tab to double*), align 8
  ret double %v
}

define double @g(double %x) {
  %p = getelementptr inbounds { i8*, double }, { i8*, double }* @mixed, i64 0, i32 1
  %v = load double, double* %p, align 8
  %m = fmul double %v, %x
  ret double %m
}

define double @d(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f, double %x)
  %s = call double (...) @__enzyme_autodiff(double (double)* @g, double %x)
  %t = fadd double %r, %s
  ret double %t
}

declare double @__enzyme_autodiff(...)

; The textual pointer types differ across LLVM versions, so the globals are
; matched loosely and the deduced types are checked via their metadata below.
; CHECK: @tab = constant {{.*}}!enzyme_type ![[TAB:[0-9]+]]
; CHECK: @mixed = constant {{.*}}!enzyme_type ![[MIXED:[0-9]+]]

; @tab is only read, so the derivative is zero.
; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK: ret { double } zeroinitializer

; CHECK: define internal { double } @diffeg(double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK: %[[MUL:.+]] = fmul fast double %differeturn, 3.000000e+00
; CHECK: insertvalue { double } undef, double %[[MUL]], 0

; @tab is a pointer to an unknown pointee, as its initializer is entirely
; self referential.
; CHECK-DAG: ![[TAB]] = !{!"Unknown", i32 -1, ![[TABP:[0-9]+]]}
; CHECK-DAG: ![[TABP]] = !{!"Pointer"}

; @mixed keeps the types of both of its fields.
; CHECK-DAG: ![[MIXED]] = !{!"Unknown", i32 -1, ![[MIXEDI:[0-9]+]]}
; CHECK-DAG: ![[MIXEDI]] = !{!"Pointer", i32 0, !{{[0-9]+}}, i32 8, ![[MIXEDF:[0-9]+]]}
; CHECK-DAG: ![[MIXEDF]] = !{!"Float@double"}
