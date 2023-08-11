; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

define void @foo(double* nocapture %a0) {
entry:
  %a2 = bitcast double* %a0 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %a2, i8 0, i64 16, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

; Function Attrs: nounwind uwtable
define void @caller(double* %a0, double* %a1) {
entry:
  tail call void (...) @__enzyme_autodiff(void (double*)* @foo, metadata !"enzyme_dupnoneed", double* %a0, double* %a1)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffefoo(double* nocapture %a0, double* nocapture %"a0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a2'ipc" = bitcast double* %"a0'" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %"a2'ipc", i8 0, i64 16, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: } 
