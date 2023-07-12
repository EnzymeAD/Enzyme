; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

declare token @llvm.julia.gc_preserve_begin(...) 

declare void @llvm.julia.gc_preserve_end(token)

define void @f({} addrspace(10)* %z) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

; Function Attrs: nounwind
declare i8* @__enzyme_virtualreverse(...)

define i8* @test() {
entry:
  %0 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*)* @f)
  ret i8* %0
}

; CHECK: define internal i8* @augmented_f({} addrspace(10)* %z, {} addrspace(10)* %"z'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i8*
; CHECK-NEXT:   store i8* null, i8** %0
; CHECK-NEXT:   %1 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z, {} addrspace(10)* %"z'")
; CHECK-NEXT:   %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
; CHECK-NEXT:   store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
; CHECK-NEXT:   call void @llvm.julia.gc_preserve_end(token %1) 
; CHECK-NEXT:   %2 = load i8*, i8** %0
; CHECK-NEXT:   ret i8* %2
; CHECK-NEXT: }

; CHECK: define internal void @diffef({} addrspace(10)* %z, {} addrspace(10)* %"z'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z, {} addrspace(10)* %"z'")
; CHECK-NEXT:   %"z_flt'ipc" = bitcast {} addrspace(10)* %"z'" to double addrspace(10)*
; CHECK-NEXT:   call void @llvm.julia.gc_preserve_end(token %0)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %1 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z, {} addrspace(10)* %"z'")
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(10)* %"z_flt'ipc", align 8
; CHECK-NEXT:   call void @llvm.julia.gc_preserve_end(token %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
