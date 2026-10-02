; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-global-activity -enzyme-globals-default-inactive -S | FileCheck %s; fi

; A stack temporary (a Fortran array descriptor passed to an optional
; argument) reaches a call through a phi with null:
;   %arg = phi ptr [ %desc, %present ], [ null, %absent ]
; Enzyme moves %desc into the block of inversion allocations, which is not
; an original block. Unwrapping the phi for the reverse pass must still work
; (it used to assert in unwrapM: 'origInstParent').

; The timer and the halo exchange are inactive (registered as such in the
; program, as in ICON on one process); the synchronization around the
; exchange is differentiated.
declare void @timer_start() #0
declare void @exchange(ptr readonly nocapture) #0

define void @sync(ptr readonly nocapture %d) {
  call void @exchange(ptr %d)
  ret void
}

define void @diffusion(ptr %p, i1 %skip, i1 %absent) {
entry:
  %desc = alloca { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }, align 8
  br i1 %skip, label %exit, label %check

check:
  br i1 %absent, label %join, label %present

present:
  store i64 0, ptr %p, align 8
  br label %join

join:
  %arg = phi ptr [ %desc, %present ], [ null, %check ]
  call void @sync(ptr %arg)
  br label %exit

exit:
  ret void
}

define void @step() {
  call void @diffusion(ptr null, i1 false, i1 false)
  call void @timer_start()
  ret void
}

define void @test() {
  call void (...) @__enzyme_autodiff(ptr @step)
  ret void
}

declare void @__enzyme_autodiff(...)

attributes #0 = { nofree "enzyme_inactive" "enzyme_no_escaping_allocation" }

; CHECK: define internal void @diffediffusion(ptr {{.*}}%p, i1 %skip, i1 %absent)
; CHECK: entry:
; CHECK-NEXT:   [[DESC:%.+]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }
; CHECK: invertjoin:
; CHECK-NEXT:   %arg_unwrap = select i1 %absent, ptr null, ptr [[DESC]]
; CHECK-NEXT:   call void @diffesync(ptr %arg_unwrap)
