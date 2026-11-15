; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; A store into an allocation that is rematerialized in the reverse pass is
; replayed there, so the address it writes to is needed in the reverse pass
; too, not only the stored value. Here the index of that store is read from
; another allocation (%ip) that an inactive I/O call writes. As the call
; cannot be repeated, %ip must be cached whole rather than erased from the
; reverse pass. This is the pattern of MITgcm's ctrl_map_ini_genarr.F:
;   READ(xx_genarr3d_file(iarr)(7:7),*) iPtr
;   igen_ptr(iPtr) = iarr
; where flang heap-allocates iPtr, as its address escapes into the runtime.

declare ptr @malloc(i64)
declare void @free(ptr)
declare ptr @_FortranAioBeginInternalListInput(ptr, i64, ptr, i64, ptr, i32)
declare i1 @_FortranAioInputInteger(ptr, ptr, i32)
declare i32 @_FortranAioEndIoStatement(ptr)

define internal void @f(ptr %x, ptr %str) noinline {
entry:
  %ip = call noalias nonnull dereferenceable(4) ptr @malloc(i64 4), !enzyme_fromstack !0
  %gp = call noalias nonnull dereferenceable(8) ptr @malloc(i64 8), !enzyme_fromstack !0
  store i32 0, ptr %gp, align 4
  %gp1 = getelementptr inbounds i32, ptr %gp, i64 1
  store i32 0, ptr %gp1, align 4
  %cookie = call ptr @_FortranAioBeginInternalListInput(ptr %str, i64 1, ptr null, i64 0, ptr null, i32 0) #0
  %ok = call i1 @_FortranAioInputInteger(ptr %cookie, ptr %ip, i32 4) #0
  %e = call i32 @_FortranAioEndIoStatement(ptr %cookie) #0
  %i = load i32, ptr %ip, align 4
  %iz = zext i32 %i to i64
  %g = getelementptr i32, ptr %gp, i64 %iz
  store i32 1, ptr %g, align 4
  %k = load i32, ptr %gp, align 4
  %kz = zext i32 %k to i64
  %xa = getelementptr inbounds double, ptr %x, i64 %kz
  %v = load double, ptr %xa, align 8
  %v2 = fmul double %v, %v
  store double %v2, ptr %xa, align 8
  call void @free(ptr %gp)
  call void @free(ptr %ip)
  ret void
}

define void @outer(ptr %x, ptr %str) {
  call void @f(ptr %x, ptr %str)
  %v = load double, ptr %x, align 8
  %v2 = fmul double %v, %v
  store double %v2, ptr %x, align 8
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test(ptr %x, ptr %dx, ptr %str) {
  call void (...) @__enzyme_autodiff(ptr @outer, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %str)
  ret void
}

attributes #0 = { "enzyme_inactive" }

!0 = !{i64 4}

; The augmented forward pass keeps %ip and hands it to the reverse pass.
; CHECK: define internal { ptr, double } @augmented_f(
; CHECK:   %ip = call noalias nonnull dereferenceable(4) ptr @malloc(i64 4)
; CHECK-NEXT:   store ptr %ip, ptr %0
; CHECK:   %ok = call i1 @_FortranAioInputInteger(ptr %cookie, ptr %ip, i32 4)
; CHECK-NOT: call void @free(ptr %ip)
; CHECK: ret { ptr, double }

; The reverse pass replays igen_ptr(iPtr) = 1 with iPtr from the tape.
; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr %str, { ptr, double } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gp = alloca i8, i64 8, align 4
; CHECK-NEXT:   %ip = extractvalue { ptr, double } %tapeArg, 0
; CHECK:   %i = load i32, ptr %ip, align 4
; CHECK-NEXT:   %iz = zext i32 %i to i64
; CHECK-NEXT:   %g = getelementptr i32, ptr %gp, i64 %iz
; CHECK-NEXT:   store i32 1, ptr %g, align 4
; CHECK-NEXT:   %k = load i32, ptr %gp, align 4
; CHECK:   call void @free(ptr %ip)
; CHECK-NEXT:   ret void
