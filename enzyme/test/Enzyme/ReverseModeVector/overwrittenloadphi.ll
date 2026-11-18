; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; Each iteration loads the pointer b->v. Every other iteration moves the data
; to the spare buffer, swaps b->v and b->spare, and loads b->v again. The store
; goes through the phi of the two loads, so the reverse pass needs the shadow of
; that phi in every iteration. b->v is overwritten after both loads, so their
; shadows must be cached: reloaded in the reverse pass, they would give the
; final b'->v. With a vector width of 2, the shadow of each load is one load
; per lane, and each lane must be cached like the load it comes from.
;
; out = x^n v[0], so d out / dx = n x^(n-1) = 4 * 1.1^3 = 5.324 for n = 4, and
; 10.648 for the second lane, whose seed is 2.

; EVAL: dx: 5.324000 10.648000

%Box = type { ptr, ptr }

@.fmt = private unnamed_addr constant [11 x i8] c"dx: %f %f\0A\00", align 1

declare i32 @printf(ptr, ...)

declare void @__enzyme_autodiff(...)

define void @f(ptr %out, ptr %b, ptr %x, i64 %n) {
entry:
  %sp = getelementptr inbounds %Box, ptr %b, i64 0, i32 1
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %latch ]
  %v0 = load ptr, ptr %b, align 8
  %t = load double, ptr %v0, align 8
  %odd = and i64 %i, 1
  %even = icmp eq i64 %odd, 0
  br i1 %even, label %move, label %latch

move:
  %s = load ptr, ptr %sp, align 8
  store double %t, ptr %s, align 8
  store ptr %s, ptr %b, align 8
  store ptr %v0, ptr %sp, align 8
  %v1 = load ptr, ptr %b, align 8
  br label %latch

latch:
  %v = phi ptr [ %v0, %loop ], [ %v1, %move ]
  %xv = load double, ptr %x, align 8
  %m = fmul double %xv, %t
  store double %m, ptr %v, align 8
  %inc = add i64 %i, 1
  %done = icmp sgt i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  %vf = load ptr, ptr %b, align 8
  %r = load double, ptr %vf, align 8
  store double %r, ptr %out, align 8
  ret void
}

define i32 @main() {
entry:
  %out = alloca double, align 8
  %dout1 = alloca double, align 8
  %dout2 = alloca double, align 8
  %buf = alloca [2 x double], align 8
  %dbuf1 = alloca [2 x double], align 8
  %dbuf2 = alloca [2 x double], align 8
  %b = alloca %Box, align 8
  %db1 = alloca %Box, align 8
  %db2 = alloca %Box, align 8
  %x = alloca double, align 8
  %dx1 = alloca double, align 8
  %dx2 = alloca double, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 1.000000e+00, ptr %dout1, align 8
  store double 2.000000e+00, ptr %dout2, align 8
  store [2 x double] [double 1.000000e+00, double 0.000000e+00], ptr %buf, align 8
  store [2 x double] zeroinitializer, ptr %dbuf1, align 8
  store [2 x double] zeroinitializer, ptr %dbuf2, align 8
  %buf1 = getelementptr inbounds [2 x double], ptr %buf, i64 0, i64 1
  %b.spare = getelementptr inbounds %Box, ptr %b, i64 0, i32 1
  store ptr %buf, ptr %b, align 8
  store ptr %buf1, ptr %b.spare, align 8
  %dbuf11 = getelementptr inbounds [2 x double], ptr %dbuf1, i64 0, i64 1
  %db1.spare = getelementptr inbounds %Box, ptr %db1, i64 0, i32 1
  store ptr %dbuf1, ptr %db1, align 8
  store ptr %dbuf11, ptr %db1.spare, align 8
  %dbuf21 = getelementptr inbounds [2 x double], ptr %dbuf2, i64 0, i64 1
  %db2.spare = getelementptr inbounds %Box, ptr %db2, i64 0, i32 1
  store ptr %dbuf2, ptr %db2, align 8
  store ptr %dbuf21, ptr %db2.spare, align 8
  store double 1.100000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx1, align 8
  store double 0.000000e+00, ptr %dx2, align 8
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_width", i64 2, ptr %out, ptr %dout1, ptr %dout2, ptr %b, ptr %db1, ptr %db2, ptr %x, ptr %dx1, ptr %dx2, i64 4)
  %r1 = load double, ptr %dx1, align 8
  %r2 = load double, ptr %dx2, align 8
  %p = call i32 (ptr, ...) @printf(ptr @.fmt, double %r1, double %r2)
  ret i32 0
}

; The forward pass caches each lane of the shadows of %v0 and %v1, and the
; reverse pass rebuilds the phi of shadows from those caches, lane by lane,
; instead of loading b'->v again.

; CHECK: define internal void @diffe2f(ptr {{.*}}%out, [2 x ptr] %"out'", ptr {{.*}}%b, [2 x ptr] %"b'", ptr {{.*}}%x, [2 x ptr] %"x'", i64 %n)

; CHECK: loop:
; CHECK:        %"v0'ipl" = load ptr, ptr %{{.+}}, align 8
; CHECK:        %"v0'ipl[[L0:[0-9]+]]" = load ptr, ptr %{{.+}}, align 8
; CHECK:        %[[p0:.+]] = getelementptr inbounds ptr, ptr %[[c0:[^,]+]], i64 %iv
; CHECK-NEXT:   store ptr %"v0'ipl", ptr %[[p0]], align 8
; CHECK-NEXT:   %[[p1:.+]] = getelementptr inbounds ptr, ptr %[[c1:[^,]+]], i64 %iv
; CHECK-NEXT:   store ptr %"v0'ipl[[L0]]", ptr %[[p1]], align 8

; CHECK: move:
; CHECK:        %"v1'ipl" = load ptr, ptr %{{.+}}, align 8
; CHECK:        %"v1'ipl[[L1:[0-9]+]]" = load ptr, ptr %{{.+}}, align 8
; CHECK-NEXT:   %[[p2:.+]] = getelementptr inbounds ptr, ptr %[[c2:[^,]+]], i64 %iv
; CHECK-NEXT:   store ptr %"v1'ipl[[L1]]", ptr %[[p2]], align 8
; CHECK-NEXT:   %[[p3:.+]] = getelementptr inbounds ptr, ptr %[[c3:[^,]+]], i64 %iv
; CHECK-NEXT:   store ptr %"v1'ipl", ptr %[[p3]], align 8

; CHECK: invertlatch_phirc:
; CHECK-NEXT:   %[[r2:.+]] = getelementptr inbounds ptr, ptr %[[c2]], i64 %"iv'ac.0"
; CHECK-NEXT:   %{{.+}} = load ptr, ptr %[[r2]], align 8
; CHECK-NEXT:   br label %invertlatch_phimerge

; CHECK: invertlatch_phirc{{[0-9]+}}:
; CHECK-NEXT:   %[[r1:.+]] = getelementptr inbounds ptr, ptr %[[c1]], i64 %"iv'ac.0"
; CHECK-NEXT:   %{{.+}} = load ptr, ptr %[[r1]], align 8
; CHECK-NEXT:   br label %invertlatch_phimerge

; CHECK: invertlatch_phimerge_phirc:
; CHECK-NEXT:   %[[r3:.+]] = getelementptr inbounds ptr, ptr %[[c3]], i64 %"iv'ac.0"
; CHECK-NEXT:   %{{.+}} = load ptr, ptr %[[r3]], align 8
; CHECK-NEXT:   br label %invertlatch_phimerge_phimerge

; CHECK: invertlatch_phimerge_phirc{{[0-9]+}}:
; CHECK-NEXT:   %[[r0:.+]] = getelementptr inbounds ptr, ptr %[[c0]], i64 %"iv'ac.0"
; CHECK-NEXT:   %{{.+}} = load ptr, ptr %[[r0]], align 8
; CHECK-NEXT:   br label %invertlatch_phimerge_phimerge
