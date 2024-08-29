; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify)" -S | FileCheck %s; fi

define double @_f2(ptr %0, i64 %1) personality ptr null {
  call void @__rust_dealloc(ptr %0, i64 %1)
  ret double 0.000000e+00
}

declare void @__rust_dealloc(ptr, i64)

declare double @__enzyme_fwddiff(...)

define double @enzyme_opt_helper_0() {
  %1 = call double (...) @__enzyme_fwddiff(ptr @_f2, metadata !"enzyme_dup", ptr null, ptr null, metadata !"enzyme_const", i64 0)
  ret double 0.000000e+00
}

; CHECK: define internal double @fwddiffe_f2(ptr %0, ptr %"'", i64 %1) 
; CHECK-NEXT:   call void @__rust_dealloc(ptr %0, i64 %1)
; CHECK-NEXT:   %3 = icmp ne ptr %0, %"'"
; CHECK-NEXT:   br i1 %3, label %free0.i, label %__enzyme_checked_free_1___rust_dealloc.exit

; CHECK: free0.i:
; CHECK-NEXT:   call void @__rust_dealloc(ptr %"'", i64 %1) 
; CHECK-NEXT:   br label %__enzyme_checked_free_1___rust_dealloc.exit

; CHECK: __enzyme_checked_free_1___rust_dealloc.exit:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }
