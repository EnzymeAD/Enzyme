; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; The ranges of a memmove may overlap, e.g. when it shifts the elements of an
; array. With v = {v0, v1, v2} and f = v[0] + 2 v[1] + 3 v[2] after the move:
;   down: memmove(v, v + 1, 16) makes v = {v1, v2, v2}: f = v1 + 5 v2
;   up:   memmove(v + 1, v, 16) makes v = {v0, v0, v1}: f = 3 v0 + 3 v1

; EVAL: down: 0.000000 1.000000 5.000000
; EVAL: up: 3.000000 3.000000 0.000000

@.down = private unnamed_addr constant [16 x i8] c"down: %f %f %f\0A\00", align 1
@.up = private unnamed_addr constant [14 x i8] c"up: %f %f %f\0A\00", align 1

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

declare void @__enzyme_autodiff(i8*, double*, double*)

declare i32 @printf(i8*, ...)

define double @sum(double* %v) {
entry:
  %p1 = getelementptr inbounds double, double* %v, i64 1
  %p2 = getelementptr inbounds double, double* %v, i64 2
  %a = load double, double* %v, align 8
  %b = load double, double* %p1, align 8
  %c = load double, double* %p2, align 8
  %b2 = fmul double %b, 2.000000e+00
  %c3 = fmul double %c, 3.000000e+00
  %ab = fadd double %a, %b2
  %f = fadd double %ab, %c3
  ret double %f
}

define double @down(double* %v) {
entry:
  %p1 = getelementptr inbounds double, double* %v, i64 1
  %dst = bitcast double* %v to i8*
  %src = bitcast double* %p1 to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 16, i1 false)
  %f = call double @sum(double* %v)
  ret double %f
}

define double @up(double* %v) {
entry:
  %p1 = getelementptr inbounds double, double* %v, i64 1
  %dst = bitcast double* %p1 to i8*
  %src = bitcast double* %v to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 16, i1 false)
  %f = call double @sum(double* %v)
  ret double %f
}

define void @print(i8* %fmt, double* %dv) {
entry:
  %p1 = getelementptr inbounds double, double* %dv, i64 1
  %p2 = getelementptr inbounds double, double* %dv, i64 2
  %a = load double, double* %dv, align 8
  %b = load double, double* %p1, align 8
  %c = load double, double* %p2, align 8
  %r = call i32 (i8*, ...) @printf(i8* %fmt, double %a, double %b, double %c)
  ret void
}

define i32 @main() {
entry:
  %v = alloca [3 x double], align 8
  %dv = alloca [3 x double], align 8
  %v0 = getelementptr inbounds [3 x double], [3 x double]* %v, i64 0, i64 0
  %dv0 = getelementptr inbounds [3 x double], [3 x double]* %dv, i64 0, i64 0
  store [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], [3 x double]* %v, align 8
  store [3 x double] zeroinitializer, [3 x double]* %dv, align 8
  call void @__enzyme_autodiff(i8* bitcast (double (double*)* @down to i8*), double* %v0, double* %dv0)
  call void @print(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.down, i64 0, i64 0), double* %dv0)
  store [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], [3 x double]* %v, align 8
  store [3 x double] zeroinitializer, [3 x double]* %dv, align 8
  call void @__enzyme_autodiff(i8* bitcast (double (double*)* @up to i8*), double* %v0, double* %dv0)
  call void @print(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.up, i64 0, i64 0), double* %dv0)
  ret i32 0
}

; The adjoint of a memmove moves each derivative from dst[i] to src[i]. When
; dst < src, it has to go back to front, or the derivative of dst[0] is added to
; src[0] = dst[1] and then moved again, to src[1].

; CHECK: define internal void @diffedown(double* nocapture %v, double* nocapture %"v'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"p1'ipg" = getelementptr inbounds double, double* %"v'", i64 1
; CHECK-NEXT:   %p1 = getelementptr inbounds double, double* %v, i64 1
; CHECK-NEXT:   %dst = bitcast double* %v to i8*
; CHECK-NEXT:   %src = bitcast double* %p1 to i8*
; CHECK-NEXT:   call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 16, i1 false)
; CHECK-NEXT:   call void @diffesum(double* %v, double* %"v'", double %differeturn)
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %[[back:.+]] = sub i64 1, %idx.i
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %"v'", i64 %[[back]]
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 8
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %"p1'ipg", i64 %[[back]]
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i, align 8
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %[[add]], double* %src.i.i, align 8
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %[[done:.+]] = icmp eq i64 2, %idx.next.i
; CHECK-NEXT:   br i1 %[[done]], label %__enzyme_memmoveadd_doubleda8sa8.exit, label %for.body.i

; CHECK: define internal void @__enzyme_memmoveadd_doubleda8sa8(double* nocapture %dst, double* nocapture %src, i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %backwards = icmp ult double* %dst, %src
; CHECK-NEXT:   %[[empty:.+]] = icmp eq i64 %num, 0
; CHECK-NEXT:   br i1 %[[empty]], label %for.end, label %for.body

; CHECK: for.body:
; CHECK-NEXT:   %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.body ]
; CHECK-NEXT:   %[[last:.+]] = sub i64 %num, 1
; CHECK-NEXT:   %[[rev:.+]] = sub i64 %[[last]], %idx
; CHECK-NEXT:   %i = select i1 %backwards, i64 %[[rev]], i64 %idx
; CHECK-NEXT:   %dst.i = getelementptr inbounds double, double* %dst, i64 %i
; CHECK-NEXT:   %dst.i.l = load double, double* %dst.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i, align 8
; CHECK-NEXT:   %src.i = getelementptr inbounds double, double* %src, i64 %i
; CHECK-NEXT:   %src.i.l = load double, double* %src.i, align 8
; CHECK-NEXT:   %[[sum:.+]] = fadd fast double %src.i.l, %dst.i.l
; CHECK-NEXT:   store double %[[sum]], double* %src.i, align 8
; CHECK-NEXT:   %idx.next = add nuw i64 %idx, 1
; CHECK-NEXT:   %[[end:.+]] = icmp eq i64 %num, %idx.next
; CHECK-NEXT:   br i1 %[[end]], label %for.end, label %for.body

; CHECK: for.end:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeup(double* nocapture %v, double* nocapture %"v'", double %differeturn)
; CHECK: for.body.i:
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %"p1'ipg", i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 8
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %"v'", i64 %idx.i
