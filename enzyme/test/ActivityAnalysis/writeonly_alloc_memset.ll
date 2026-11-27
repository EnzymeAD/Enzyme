; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=alloc_zero -activity-analysis-duplicated-ret=1 -S -o /dev/null | FileCheck %s --check-prefix=ALLOC; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=zero_indirect -S -o /dev/null | FileCheck %s --check-prefix=ZERO; fi

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) #0
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

define ptr addrspace(10) @alloc_zero(i64 %sz) {
entry:
  %mem = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 %sz, ptr addrspace(10) null)
  %raw = addrspacecast ptr addrspace(10) %mem to ptr
  call void @llvm.memset.p0.i64(ptr %raw, i8 0, i64 %sz, i1 false)
  ret ptr addrspace(10) %mem
}

define void @zero_indirect(ptr %arr, i64 %sz) {
entry:
  %buf = load ptr, ptr %arr, align 8, !enzyme_type !{!"Unknown", i64 -1, !{!"Pointer", i64 -1, !{!"Float@double"}}}
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 %sz, i1 false)
  ret void
}

attributes #0 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }

; ALLOC: %mem = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 %sz, ptr addrspace(10) null): icv:0 ici:1
; ALLOC-NEXT: %raw = addrspacecast ptr addrspace(10) %mem to ptr: icv:0 ici:1
; ALLOC-NEXT: call void @llvm.memset.p0.i64(ptr %raw, i8 0, i64 %sz, i1 false): icv:1 ici:1

; ZERO: ptr %arr: icv:0
; ZERO: %buf = load ptr, ptr %arr, align 8{{.*}}: icv:0 ici:1
; ZERO-NEXT: call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 %sz, i1 false): icv:1 ici:1
