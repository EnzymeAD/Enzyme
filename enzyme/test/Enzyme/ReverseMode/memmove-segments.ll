; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,instcombine,%simplifycfg)" -S | FileCheck %s

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; A memmove of constant size over an array of {double, i64} is split by type
; into segments: double, integer, double, integer. When the ranges overlap, one
; segment's source is another segment's destination. With v = {s0, s1, s2},
; s = {x, n}, n = 1 and f = x0 n0 + 2 x1 n1 + 3 x2 n2 after the move:
;   down: memmove(v, v + 1, 32) makes v = {s1, s2, s2}: df/dx = {0, 1, 5}
;   up:   memmove(v + 1, v, 32) makes v = {s0, s0, s1}: df/dx = {3, 3, 0}
; The derivatives of the double segments must be moved back to front for down,
; and the copies of the integer segments of the shadow must go back to front
; for up: its n = {10, 20, 30} become {10, 10, 20}.

; EVAL: down: x 0.000000 1.000000 5.000000 n 20 30 30
; EVAL: up: x 3.000000 3.000000 0.000000 n 10 10 20

%struct.S = type { double, i64 }

@.down = private unnamed_addr constant [6 x i8] c"down:\00", align 1
@.up = private unnamed_addr constant [4 x i8] c"up:\00", align 1
@.fmt = private unnamed_addr constant [29 x i8] c"%s x %f %f %f n %ld %ld %ld\0A\00", align 1

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

declare void @__enzyme_autodiff(i8*, %struct.S*, %struct.S*)

declare i32 @printf(i8*, ...)

define double @sum(%struct.S* %v) {
entry:
  %x0p = getelementptr inbounds %struct.S, %struct.S* %v, i64 0, i32 0
  %x1p = getelementptr inbounds %struct.S, %struct.S* %v, i64 1, i32 0
  %x2p = getelementptr inbounds %struct.S, %struct.S* %v, i64 2, i32 0
  %n0p = getelementptr inbounds %struct.S, %struct.S* %v, i64 0, i32 1
  %n1p = getelementptr inbounds %struct.S, %struct.S* %v, i64 1, i32 1
  %n2p = getelementptr inbounds %struct.S, %struct.S* %v, i64 2, i32 1
  %x0 = load double, double* %x0p, align 8
  %x1 = load double, double* %x1p, align 8
  %x2 = load double, double* %x2p, align 8
  %n0 = load i64, i64* %n0p, align 8
  %n1 = load i64, i64* %n1p, align 8
  %n2 = load i64, i64* %n2p, align 8
  %n0f = sitofp i64 %n0 to double
  %n1f = sitofp i64 %n1 to double
  %n2f = sitofp i64 %n2 to double
  %t0 = fmul double %x0, %n0f
  %t1 = fmul double %x1, %n1f
  %t2 = fmul double %x2, %n2f
  %t1s = fmul double %t1, 2.000000e+00
  %t2s = fmul double %t2, 3.000000e+00
  %a = fadd double %t0, %t1s
  %f = fadd double %a, %t2s
  ret double %f
}

define double @down(%struct.S* %v) {
entry:
  %v1 = getelementptr inbounds %struct.S, %struct.S* %v, i64 1
  %dst = bitcast %struct.S* %v to i8*
  %src = bitcast %struct.S* %v1 to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 32, i1 false)
  %f = call double @sum(%struct.S* %v)
  ret double %f
}

define double @up(%struct.S* %v) {
entry:
  %v1 = getelementptr inbounds %struct.S, %struct.S* %v, i64 1
  %dst = bitcast %struct.S* %v1 to i8*
  %src = bitcast %struct.S* %v to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 32, i1 false)
  %f = call double @sum(%struct.S* %v)
  ret double %f
}

define void @print(i8* %name, %struct.S* %dv) {
entry:
  %x0p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 0, i32 0
  %x1p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 1, i32 0
  %x2p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 2, i32 0
  %n0p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 0, i32 1
  %n1p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 1, i32 1
  %n2p = getelementptr inbounds %struct.S, %struct.S* %dv, i64 2, i32 1
  %x0 = load double, double* %x0p, align 8
  %x1 = load double, double* %x1p, align 8
  %x2 = load double, double* %x2p, align 8
  %n0 = load i64, i64* %n0p, align 8
  %n1 = load i64, i64* %n1p, align 8
  %n2 = load i64, i64* %n2p, align 8
  %fmt = getelementptr inbounds [29 x i8], [29 x i8]* @.fmt, i64 0, i64 0
  %r = call i32 (i8*, ...) @printf(i8* %fmt, i8* %name, double %x0, double %x1, double %x2, i64 %n0, i64 %n1, i64 %n2)
  ret void
}

define void @init(%struct.S* %v, %struct.S* %dv) {
entry:
  %va = bitcast %struct.S* %v to [3 x %struct.S]*
  %dva = bitcast %struct.S* %dv to [3 x %struct.S]*
  store [3 x %struct.S] [%struct.S { double 1.000000e+00, i64 1 }, %struct.S { double 2.000000e+00, i64 1 }, %struct.S { double 3.000000e+00, i64 1 }], [3 x %struct.S]* %va, align 8
  store [3 x %struct.S] [%struct.S { double 0.000000e+00, i64 10 }, %struct.S { double 0.000000e+00, i64 20 }, %struct.S { double 0.000000e+00, i64 30 }], [3 x %struct.S]* %dva, align 8
  ret void
}

define i32 @main() {
entry:
  %v = alloca [3 x %struct.S], align 8
  %dv = alloca [3 x %struct.S], align 8
  %v0 = getelementptr inbounds [3 x %struct.S], [3 x %struct.S]* %v, i64 0, i64 0
  %dv0 = getelementptr inbounds [3 x %struct.S], [3 x %struct.S]* %dv, i64 0, i64 0
  call void @init(%struct.S* %v0, %struct.S* %dv0)
  call void @__enzyme_autodiff(i8* bitcast (double (%struct.S*)* @down to i8*), %struct.S* %v0, %struct.S* %dv0)
  call void @print(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.down, i64 0, i64 0), %struct.S* %dv0)
  call void @init(%struct.S* %v0, %struct.S* %dv0)
  call void @__enzyme_autodiff(i8* bitcast (double (%struct.S*)* @up to i8*), %struct.S* %v0, %struct.S* %dv0)
  call void @print(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.up, i64 0, i64 0), %struct.S* %dv0)
  ret i32 0
}

; The copies of the integer segments of the shadow go front to back for down,
; and the derivatives of the double segments back to front.

; CHECK: define internal void @diffedown(%struct.S* nocapture %v, %struct.S* nocapture %"v'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v1 = getelementptr inbounds %struct.S, %struct.S* %v, i64 1
; CHECK-NEXT:   %dst = bitcast %struct.S* %v to i8*
; CHECK-NEXT:   %src = bitcast %struct.S* %v1 to i8*
; CHECK-NEXT:   %[[n0:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 0, i32 1
; CHECK-NEXT:   %[[n1:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 1
; CHECK-NEXT:   %[[l1:.+]] = load i64, i64* %[[n1]], align 8
; CHECK-NEXT:   store i64 %[[l1]], i64* %[[n0]], align 8
; CHECK-NEXT:   %[[n1b:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 1
; CHECK-NEXT:   %[[n2:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 2, i32 1
; CHECK-NEXT:   %[[l2:.+]] = load i64, i64* %[[n2]], align 8
; CHECK-NEXT:   store i64 %[[l2]], i64* %[[n1b]], align 8
; CHECK-NEXT:   call void @llvm.memmove.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(32) %dst, i8* noundef nonnull align 8 dereferenceable(32) %src, i64 32, i1 false)
; CHECK-NEXT:   call void @diffesum(%struct.S* %v, %struct.S* %"v'", double %differeturn)
; CHECK-NEXT:   %[[x1:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 0
; CHECK-NEXT:   %[[x2:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 2, i32 0
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:
; CHECK:   %dst.i.i = getelementptr inbounds double, double* %[[x1]], i64 %{{.+}}
; CHECK:   %src.i.i = getelementptr inbounds double, double* %[[x2]], i64 %{{.+}}

; CHECK: __enzyme_memmoveadd_doubleda8sa8.exit:
; CHECK-NEXT:   %[[x0:.+]] = getelementptr %struct.S, %struct.S* %"v'", i64 0, i32 0
; CHECK-NEXT:   %[[x1b:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 0
; CHECK-NEXT:   br label %for.body.i9

; CHECK: for.body.i9:
; CHECK:   %dst.i.i4 = getelementptr inbounds double, double* %[[x0]], i64 %{{.+}}
; CHECK:   %src.i.i6 = getelementptr inbounds double, double* %[[x1b]], i64 %{{.+}}

; For up, the copies go back to front, and the derivatives front to back.

; CHECK: define internal void @diffeup(%struct.S* nocapture %v, %struct.S* nocapture %"v'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v1 = getelementptr inbounds %struct.S, %struct.S* %v, i64 1
; CHECK-NEXT:   %dst = bitcast %struct.S* %v1 to i8*
; CHECK-NEXT:   %src = bitcast %struct.S* %v to i8*
; CHECK-NEXT:   %[[n2:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 2, i32 1
; CHECK-NEXT:   %[[n1:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 1
; CHECK-NEXT:   %[[l1:.+]] = load i64, i64* %[[n1]], align 8
; CHECK-NEXT:   store i64 %[[l1]], i64* %[[n2]], align 8
; CHECK-NEXT:   %[[n1b:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 1
; CHECK-NEXT:   %[[n0:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 0, i32 1
; CHECK-NEXT:   %[[l0:.+]] = load i64, i64* %[[n0]], align 8
; CHECK-NEXT:   store i64 %[[l0]], i64* %[[n1b]], align 8
; CHECK-NEXT:   call void @llvm.memmove.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(32) %dst, i8* noundef nonnull align 8 dereferenceable(32) %src, i64 32, i1 false)
; CHECK-NEXT:   call void @diffesum(%struct.S* %v, %struct.S* %"v'", double %differeturn)
; CHECK-NEXT:   %[[x1:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 0
; CHECK-NEXT:   %[[x0:.+]] = getelementptr %struct.S, %struct.S* %"v'", i64 0, i32 0
; CHECK-NEXT:   br label %for.body.i

; CHECK: __enzyme_memmoveadd_doubleda8sa8.exit:
; CHECK-NEXT:   %[[x2:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 2, i32 0
; CHECK-NEXT:   %[[x1b:.+]] = getelementptr inbounds %struct.S, %struct.S* %"v'", i64 1, i32 0
