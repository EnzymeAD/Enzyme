; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare i32 @cuStreamCreate(ptr, i32)
declare i32 @cuStreamSynchronize(ptr) "enzyme_no_escaping_allocation"
declare i32 @cuStreamDestroy(ptr) "enzyme_no_escaping_allocation"

declare double @__enzyme_autodiff(...)
declare { ptr, double } @__enzyme_augmentfwd(...)
declare double @__enzyme_reverse(...)

define double @stream_single(double %x) {
entry:
  %sp = alloca ptr, align 8
  %c = call i32 @cuStreamCreate(ptr %sp, i32 0)
  %s = load ptr, ptr %sp, align 8
  %sy = call i32 @cuStreamSynchronize(ptr %s)
  %d = call i32 @cuStreamDestroy(ptr %s)
  %r = fmul double %x, %x
  ret double %r
}

define double @stream_loop(double %x, i64 %n) {
entry:
  %sp = alloca ptr, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %nsum, %loop ]
  %c = call i32 @cuStreamCreate(ptr %sp, i32 0)
  %s = load ptr, ptr %sp, align 8
  %sy = call i32 @cuStreamSynchronize(ptr %s)
  %d = call i32 @cuStreamDestroy(ptr %s)
  %nsum = fadd double %sum, %x
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret double %nsum
}

define double @test_single(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @stream_single, double %x)
  ret double %r
}

define double @test_loop(double %x, i64 %n) {
  %r = call double (...) @__enzyme_autodiff(ptr @stream_loop, double %x, metadata !"enzyme_const", i64 %n)
  ret double %r
}

define double @test_split(double %x) {
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @stream_single, double %x)
  %tape = extractvalue { ptr, double } %aug, 0
  %r = call double (...) @__enzyme_reverse(ptr @stream_single, double %x, double 1.0, ptr %tape)
  ret double %r
}

; CHECK: define internal { double } @diffestream_single(
; CHECK: entry:
; CHECK-NOT: load ptr, ptr %sp
; CHECK:   %c = call i32 @cuStreamCreate(ptr %sp, i32 0)
; CHECK-NEXT:   %[[HANDLE:.+]] = load ptr, ptr %sp, align 8
; CHECK-NOT: @cuStreamDestroy
; CHECK: invertentry:
; CHECK:   call i32 @cuStreamSynchronize(ptr %s)
; CHECK-NEXT:   call i32 @cuStreamDestroy(ptr %[[HANDLE]])

; CHECK: define internal { double } @diffestream_loop(
; CHECK: loop:
; CHECK:   %c = call i32 @cuStreamCreate(ptr %sp, i32 0)
; CHECK-NEXT:   %[[LHANDLE:.+]] = load ptr, ptr %sp, align 8
; CHECK:   store ptr %[[LHANDLE]], ptr %{{.+}}, align 8
; CHECK-NOT: @cuStreamDestroy
; CHECK: invertloop:
; CHECK:   call i32 @cuStreamSynchronize(ptr %{{.+}})
; CHECK-NOT: load ptr, ptr %sp
; CHECK:   %[[CPTR:.+]] = load ptr, ptr %_cache, align 8
; CHECK-NEXT:   %[[CGEP:.+]] = getelementptr inbounds ptr, ptr %[[CPTR]], i64 %{{.+}}
; CHECK-NEXT:   %[[CDEST:.+]] = load ptr, ptr %[[CGEP]], align 8
; CHECK-NEXT:   call i32 @cuStreamDestroy(ptr %[[CDEST]])

; CHECK: define internal { ptr, double } @augmented_stream_single(
; CHECK:   %c = call i32 @cuStreamCreate(ptr %[[SSP:.+]], i32 0)
; CHECK-NEXT:   %[[SHANDLE:.+]] = load ptr, ptr %[[SSP]], align 8
; CHECK:   store ptr %[[SHANDLE]], ptr %{{.+}}, align 8
; CHECK-NOT: @cuStreamDestroy
; CHECK:   ret { ptr, double }

; CHECK: define internal { double } @diffestream_single{{.*}}(double %x, double %differeturn, ptr %tapeArg)
; CHECK:   %c_psxtmp = extractvalue { ptr, ptr } %truetape, 0
; CHECK:   %s = extractvalue { ptr, ptr } %truetape, 1
; CHECK: invertentry:
; CHECK:   call i32 @cuStreamSynchronize(ptr %s)
; CHECK-NEXT:   call i32 @cuStreamDestroy(ptr %c_psxtmp)
