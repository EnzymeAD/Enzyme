; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

define i64 @inner(i64* %ptr) {
entry:
  %rs = cmpxchg i64* %ptr, i64 1, i64 2 acq_rel acquire
  %val = extractvalue { i64, i1 } %rs, 0
  ret i64 %val
}

define void @caller(i64* %ptr) {
entry:
  call i64 (i64 (i64*)*, ...) @__enzyme_autodiff(i64 (i64*)* nonnull @inner, metadata !"enzyme_const", i64* %ptr)
  ret void
}

declare i64 @__enzyme_autodiff(i64 (i64*)*, ...)

; CHECK: define internal void @diffeinner(i64* %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %rs = cmpxchg i64* %ptr, i64 1, i64 2 acq_rel acquire
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
