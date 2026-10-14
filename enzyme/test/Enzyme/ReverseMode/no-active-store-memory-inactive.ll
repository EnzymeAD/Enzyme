; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-recursive-no-active-store -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; An index buffer is only ever zeroed or written with integers (and read as
; integers, by TBAA), and reaches
; the callee only through a local header passed to a parameter through which
; nothing active is stored. The buffer and the header can never hold a
; derivative, so they are inactive: no shadow buffer is allocated and the
; callee gets no shadow header.

; Reads A at an index kept in a separate buffer reached through a header, and
; writes an index back through it, like det_of_minor's row and column lists.
define double @lookup(ptr %A, ptr %hdr, i64 %k) {
  %idx = load ptr, ptr %hdr, align 8
  %ip = getelementptr inbounds i64, ptr %idx, i64 %k
  %i = load i64, ptr %ip, align 8, !tbaa !0
  %p = getelementptr inbounds double, ptr %A, i64 %i
  %v = load double, ptr %p, align 8
  store i64 0, ptr %ip, align 8
  %sq = fmul double %v, %v
  ret double %sq
}

; The index buffer has a runtime size and is only zeroed by a memset, so type
; analysis cannot tell that all of it holds integers.
define double @f(ptr %A, i64 %n, i64 %k) {
  %bytes = shl i64 %n, 3
  %buf = call noalias ptr @malloc(i64 %bytes)
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 %bytes, i1 false)
  %hdr = alloca ptr, align 8
  store ptr %buf, ptr %hdr, align 8
  %r = call double @lookup(ptr %A, ptr %hdr, i64 %k)
  call void @free(ptr %buf)
  ret double %r
}

define void @df(ptr %A, ptr %dA, i64 %n, i64 %k) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, ptr %A, ptr %dA, i64 %n, i64 %k)
  ret void
}

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare double @__enzyme_autodiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffef(ptr {{.*}}%A, ptr {{.*}}%"A'", i64 %n, i64 %k, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %bytes = shl i64 %n, 3
; CHECK-NEXT:   %buf = call noalias ptr @malloc(i64 %bytes)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 %bytes, i1 false)
; CHECK-NEXT:   %hdr = alloca ptr, align 8
; CHECK-NEXT:   store ptr %buf, ptr %hdr, align 8
; CHECK-NEXT:   call void @diffelookup(ptr %A, ptr %"A'", ptr %hdr, i64 %k, double %differeturn)
; CHECK-NEXT:   call void @free(ptr %buf)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffelookup(ptr {{.*}}%A, ptr {{.*}}%"A'", ptr {{.*}}%hdr, i64 %k, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %idx = load ptr, ptr %hdr, align 8
; CHECK-NEXT:   %ip = getelementptr inbounds {{(nuw )?}}i64, ptr %idx, i64 %k
; CHECK-NEXT:   %i = load i64, ptr %ip, align 8{{.*}}
; CHECK-NEXT:   %"p'ipg" = getelementptr inbounds {{(nuw )?}}double, ptr %"A'", i64 %i
; CHECK-NEXT:   %p = getelementptr inbounds {{(nuw )?}}double, ptr %A, i64 %i
; CHECK-NEXT:   %v = load double, ptr %p, align 8
; CHECK-NEXT:   store i64 0, ptr %ip, align 8
; CHECK-NEXT:   %[[m0:.+]] = fmul fast double %differeturn, %v
; CHECK-NEXT:   %[[m1:.+]] = fmul fast double %differeturn, %v
; CHECK-NEXT:   %[[d:.+]] = fadd fast double %[[m0]], %[[m1]]
; CHECK-NEXT:   %[[old:.+]] = load double, ptr %"p'ipg", align 8
; CHECK-NEXT:   %[[new:.+]] = fadd fast double %[[old]], %[[d]]
; CHECK-NEXT:   store double %[[new]], ptr %"p'ipg", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
