; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s; fi

; The pointer of %v is a phi whose incoming values are the alloca %x and a
; pointer into the readonly argument %q. After the loop, %x is overwritten by %r, so
; the reverse pass must use the cached value of %v (not reload it).
;
; arePointersGuaranteedNoAlias(getelementptr/phi, %x) used to claim the two
; cannot alias: getBaseObject stops at the multi-incoming phi, %x is a noalias
; alloca, and notCapturedBefore walks through phis as pointer arithmetic, so it
; never noticed that %x flows directly into the phi. The load was then
; considered reloadable and the reverse pass read the overwritten value.
; (Reduced from the ADBench LSTM, EnzymeAD/Enzyme#3713.)

define void @f(ptr noalias readonly %w, ptr noalias readonly %q, ptr noalias %s, ptr noalias %out) {
entry:
  %x = alloca double, align 8
  %w0 = load double, ptr %w, align 8
  store double %w0, ptr %x, align 8
  %w1p = getelementptr inbounds double, ptr %w, i64 1
  %w1 = load double, ptr %w1p, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = phi ptr [ %x, %entry ], [ %qp, %loop ]
  %v = load double, ptr %p, align 8
  %m = fmul double %v, %w1
  %sp = getelementptr inbounds double, ptr %s, i64 %i
  store double %m, ptr %sp, align 8
  %qp = getelementptr inbounds double, ptr %q, i64 %i
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp eq i64 %i.next, 2
  br i1 %c, label %exit, label %loop

exit:
  %w2p = getelementptr inbounds double, ptr %w, i64 2
  %w2 = load double, ptr %w2p, align 8
  %r = fmul double %m, %w2
  store double %r, ptr %x, align 8
  %o = load double, ptr %x, align 8
  store double %o, ptr %out, align 8
  ret void
}

declare void @__enzyme_autodiff(...)

define void @df(ptr %w, ptr %dw, ptr %q, ptr %s, ptr %ds, ptr %out, ptr %dout) {
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %w, ptr %dw, metadata !"enzyme_const", ptr %q, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dupnoneed", ptr %out, ptr %dout)
  ret void
}

; The value of %v must be cached in the forward loop and read back from the
; cache in the reverse; the reverse must not reload it through the pointer phi
; (previously `%v_unwrap = load double, ptr select(.., %q, %x)` after %x had
; been overwritten by %r).

; CHECK: define internal void @diffef(
; CHECK: loop:
; CHECK:   %v = load double, ptr %p, align 8
; CHECK:   %[[CACHE:.+]] = getelementptr inbounds double, ptr %[[MC:.+]], i64 %iv
; CHECK:   store double %v, ptr %[[CACHE]], align 8, !invariant.group
; CHECK: exit:
; CHECK:   store double %r, ptr %x, align 8
; CHECK: invertloop:
; CHECK-NOT: %v_unwrap
; CHECK:   %[[RC:.+]] = getelementptr inbounds double, ptr %[[MC]], i64 %"iv'ac.0"
; CHECK:   load double, ptr %[[RC]], align 8, !invariant.group
; CHECK-NOT: %v_unwrap
; CHECK: incinvertloop:
