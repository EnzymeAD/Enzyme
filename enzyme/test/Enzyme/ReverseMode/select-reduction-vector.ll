; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A loop that keeps the lane-wise maximum of vectors in a phi. The reverse
; pass needs, per lane, the last iteration that selected a new value, so the
; index that records it is a vector as well.

define double @f(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi <2 x double> [ zeroinitializer, %entry ], [ %sel, %loop ]
  %p = getelementptr inbounds <2 x double>, ptr %x, i64 %i
  %v = load <2 x double>, ptr %p, align 16
  %c = fcmp ogt <2 x double> %v, %acc
  %sel = select <2 x i1> %c, <2 x double> %v, <2 x double> %acc
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  %e0 = extractelement <2 x double> %sel, i64 0
  %e1 = extractelement <2 x double> %sel, i64 1
  %r = fadd double %e0, %e1
  ret double %r
}

declare void @__enzyme_autodiff(...)

define void @test(ptr %x, ptr %dx, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i64 %n)
  ret void
}

; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn)
; CHECK: loop:
; CHECK-NEXT:   [[IDX:%[0-9]+]] = phi <2 x i64> [ zeroinitializer, %entry ], [ [[NIDX:%[0-9]+]], %loop ]
; CHECK:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK:   [[SPLAT:%[a-z0-9.]+]] = shufflevector <2 x i64> {{%[a-z0-9.]+}}, <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK:   %c = fcmp ogt <2 x double> %v, %acc
; CHECK-NEXT:   [[NIDX]] = select <2 x i1> %c, <2 x i64> [[SPLAT]], <2 x i64> [[IDX]]
; CHECK-NEXT:   %sel = select <2 x i1> %c, <2 x double> %v, <2 x double> %acc

; CHECK: invertloop:
; CHECK:   %iv.next_unwrap = add nuw nsw i64 %"iv'ac.0", 1
; CHECK:   [[USPLAT:%[a-z0-9.]+]] = shufflevector <2 x i64> {{%[a-z0-9.]+}}, <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   [[DSEL:%[0-9]+]] = load <2 x double>, ptr %"sel'de", align 16
; CHECK-NEXT:   [[ISSEL:%[0-9]+]] = icmp eq <2 x i64> [[NIDX]], [[USPLAT]]
; CHECK-NEXT:   [[DV:%[0-9]+]] = select fast <2 x i1> [[ISSEL]], <2 x double> [[DSEL]], <2 x double> zeroinitializer
; CHECK-NEXT:   %"p'ipg_unwrap" = getelementptr inbounds <2 x double>, ptr %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   [[OLD:%[0-9]+]] = load <2 x double>, ptr %"p'ipg_unwrap", align 16
; CHECK-NEXT:   [[NEW:%[0-9]+]] = fadd fast <2 x double> [[OLD]], [[DV]]
; CHECK-NEXT:   store <2 x double> [[NEW]], ptr %"p'ipg_unwrap", align 16
