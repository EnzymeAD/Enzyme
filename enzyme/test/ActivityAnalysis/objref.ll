; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -activity-analysis-inactive-args -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -activity-analysis-inactive-args -S | FileCheck %s

declare nonnull i8** @julia.pointer_from_objref({} addrspace(11)*) 

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)

declare i8* @malloc(i64)

define i8* @f({} addrspace(11)* %a9) {
entry:
  %a8 = call noalias i8* @malloc(i64 32)
  %a10 = call nonnull i8** @julia.pointer_from_objref({} addrspace(11)* %a9) 
  %a12 = load i8*, i8** %a10, align 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %a8, i8* %a12, i64 32, i1 false)
  ret i8* %a8
}

; CHECK: {} addrspace(11)* %a9: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %a8 = call noalias i8* @malloc(i64 32): icv:1 ici:1
; CHECK-NEXT:   %a10 = call nonnull i8** @julia.pointer_from_objref({} addrspace(11)* %a9): icv:1 ici:1
; CHECK-NEXT:   %a12 = load i8*, i8** %a10, align 8: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memmove.p0i8.p0i8.i64(i8* %a8, i8* %a12, i64 32, i1 false): icv:1 ici:1
; CHECK-NEXT:   ret i8* %a8: icv:1 ici:1
