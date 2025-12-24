; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=_Z8simulatef -S | FileCheck %s

declare {} addrspace(10)* @ijl_alloc_array_1d({} addrspace(10)*, i64)

declare void @ijl_bounds_error_ints({} addrspace(12)*, i64*, i64)

define double @_Z8simulatef(i1 %cmp, i64 %val) {
entry:
  br i1 %cmp, label %b1, label %b2

b1:
  %alloc1 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_1d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 123962498842016 to {}*) to {} addrspace(10)*), i64 2)
  br label %end

b2:
  %alloc2 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_1d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 123962498842016 to {}*) to {} addrspace(10)*), i64 2)
  br label %end

end:
  %phi = phi {} addrspace(10)* [ %alloc1, %b1 ], [ %alloc2, %b2 ]
  %ac = addrspacecast {} addrspace(10)* %phi to double addrspace(13)* addrspace(11)*
  %arrayptr189517 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %ac, align 8
  %gep = getelementptr inbounds double, double addrspace(13)* %arrayptr189517, i64 0
  %si = sitofp i64 %val to double
  store double %si, double addrspace(13)* %gep, align 8
  br i1 %cmp, label %oob, label %exit

oob:
  %errorbox186 = alloca i64
  %ei = addrspacecast {} addrspace(10)* %phi to {} addrspace(12)*
  call void @ijl_bounds_error_ints({} addrspace(12)* %ei, i64* noundef nonnull align 8 %errorbox186, i64 noundef 1)
  unreachable

exit:
  %res = load double, double addrspace(13)* %gep
  ret double %res
}

; CHECK: entry
; CHECK-NEXT:   br i1 %cmp, label %b1, label %b2: icv:1 ici:1
; CHECK-NEXT: b1
; CHECK-NEXT:   %alloc1 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_1d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 123962498842016 to {}*) to {} addrspace(10)*), i64 2): icv:1 ici:1
; CHECK-NEXT:   br label %end: icv:1 ici:1
; CHECK-NEXT: b2
; CHECK-NEXT:   %alloc2 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_1d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 123962498842016 to {}*) to {} addrspace(10)*), i64 2): icv:1 ici:1
; CHECK-NEXT:   br label %end: icv:1 ici:1
; CHECK-NEXT: end
; CHECK-NEXT:   %phi = phi {} addrspace(10)* [ %alloc1, %b1 ], [ %alloc2, %b2 ]: icv:1 ici:1
; CHECK-NEXT:   %ac = addrspacecast {} addrspace(10)* %phi to double addrspace(13)* addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %arrayptr189517 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %ac, align 8: icv:1 ici:1
; CHECK-NEXT:   %gep = getelementptr inbounds double, double addrspace(13)* %arrayptr189517, i64 0: icv:1 ici:1
; CHECK-NEXT:   %si = sitofp i64 %val to double: icv:1 ici:1
; CHECK-NEXT:   store double %si, double addrspace(13)* %gep, align 8: icv:1 ici:1
; CHECK-NEXT:   br i1 %cmp, label %oob, label %exit: icv:1 ici:1
; CHECK-NEXT: oob
; CHECK-NEXT:   %errorbox186 = alloca i64, align 8: icv:1 ici:1
; CHECK-NEXT:   %ei = addrspacecast {} addrspace(10)* %phi to {} addrspace(12)*: icv:1 ici:1
; CHECK-NEXT:   call void @ijl_bounds_error_ints({} addrspace(12)* %ei, i64* noundef nonnull align 8 %errorbox186, i64 noundef 1): icv:1 ici:1
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: exit
; CHECK-NEXT:   %res = load double, double addrspace(13)* %gep, align 8: icv:1 ici:1
; CHECK-NEXT:   ret double %res: icv:1 ici:1
