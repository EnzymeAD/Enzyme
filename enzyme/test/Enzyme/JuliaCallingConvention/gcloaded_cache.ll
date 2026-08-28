; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-julia-addr-load -mem2reg -simplifycfg -adce -S | FileCheck %s; fi

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare noundef nonnull {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* nocapture noundef nonnull readnone, {} addrspace(10)** noundef nonnull readnone) local_unnamed_addr #1

define double @f({} addrspace(10)** addrspace(11)* %slot_ptr, {} addrspace(10)* %obj1, {} addrspace(10)* %obj2, double %val) {
entry:
  ; Load the GC slot (type {} addrspace(10)**, AS 0 pointer pointing to AS 10)
  %slot = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %slot_ptr, align 8
  
  ; GC loaded calls
  %191 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* %obj1, {} addrspace(10)** %slot)
  %194 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* %obj2, {} addrspace(10)** %slot)
  
  ; Comparison
  %.not7 = icmp eq {} addrspace(10)* addrspace(13)* %191, %194
  br i1 %.not7, label %then, label %else

then:
  ; Active float work to force AD to track this branch
  %mul = fmul double %val, %val
  ret double %mul

else:
  %add = fadd double %val, 2.0
  ret double %add
}

declare { i8*, double } @__enzyme_augmentfwd(...)

define { i8*, double } @test({} addrspace(10)** addrspace(11)* %slot_ptr, {} addrspace(10)* %obj1, {} addrspace(10)* %obj2, double %val, double %dval) {
entry:
  %res = call { i8*, double } (...) @__enzyme_augmentfwd(double ({} addrspace(10)** addrspace(11)*, {} addrspace(10)*, {} addrspace(10)*, double)* @f, metadata !"enzyme_const", {} addrspace(10)** addrspace(11)* %slot_ptr, metadata !"enzyme_const", {} addrspace(10)* %obj1, metadata !"enzyme_const", {} addrspace(10)* %obj2, metadata !"enzyme_dup", double %val, double %dval)
  ret { i8*, double } %res
}

attributes #1 = { nofree norecurse nosync nounwind speculatable willreturn "enzyme_nocache" "enzyme_shouldrecompute" }

; We want to make sure the tape does NOT store the GC slot pointer ({} addrspace(10)**).
; CHECK: define internal { i8*, double } @augmented_f({} addrspace(10)** addrspace(11)* %slot_ptr, {} addrspace(10)* %obj1, {} addrspace(10)* %obj2, double %val, double %"val'")
; CHECK-NOT: store {} addrspace(10)**
; CHECK: }
