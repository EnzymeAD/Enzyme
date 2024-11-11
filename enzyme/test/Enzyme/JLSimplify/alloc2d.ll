; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare noalias nonnull {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)*, i64, i64) local_unnamed_addr #0

declare fastcc nonnull {} addrspace(10)* @a20()

define i1 @chillin()  {
bb:
  %i = call fastcc nonnull {} addrspace(10)* @a20()
  %i2 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)* addrspacecast ({}* inttoptr (i64 139753406594768 to {}*) to {} addrspace(10)*), i64 5, i64 6)
  %i3 = icmp eq {} addrspace(10)* %i2, %i
  ret i1 %i3
}

attributes #0 = { inaccessiblememonly mustprogress nofree nounwind willreturn }


; CHECK: define i1 @chillin() {
; CHECK-NEXT: bb:
; CHECK-NEXT:   %i = call fastcc nonnull {} addrspace(10)* @a20()
; CHECK-NEXT:   %i2 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)* addrspacecast ({}* inttoptr (i64 139753406594768 to {}*) to {} addrspace(10)*), i64 5, i64 6)
; CHECK-NEXT:   %i3 = icmp eq {} addrspace(10)* %i2, %i
; CHECK-NEXT:   ret i1 false
; CHECK-NEXT: }
