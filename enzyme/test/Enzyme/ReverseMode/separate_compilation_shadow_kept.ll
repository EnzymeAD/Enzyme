; RUN: if [ %llvmver -ge 21 ]; then echo "user" > %t.exports; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-derivatives=reverse -enzyme-export-list=%t.exports -S | FileCheck %s; fi

; Separate compilation: @ext is defined in another module, and its primal
; does not capture its argument (flang marks every Fortran dummy argument
; captures(none)). Its derivative may still keep the shadow beyond the call,
; as the record of a nonblocking MPI receive does. So the shadow of the
; local %buf must be the same memory in the augmented call and in the
; reverse call: allocated in the augmented primal and kept in the tape, not
; recreated in the reverse pass.

declare void @ext(ptr captures(none))

define double @user(ptr %x) {
entry:
  %buf = alloca [4 x double], align 8
  %v = load double, ptr %x, align 8
  %s = fmul double %v, %v
  store double %s, ptr %buf, align 8
  call void @ext(ptr %buf)
  %r = load double, ptr %buf, align 8
  ret double %r
}


; CHECK: define internal { ptr, double } @augmented_user(
; CHECK: %"buf'mi" = tail call {{.*}}ptr @malloc(i64 32)
; CHECK: store ptr %"buf'mi", ptr
; CHECK: %_augmented = call { ptr } %{{.*}}(ptr %buf, ptr %"buf'mi")

; CHECK: define internal void @diffeuser(
; CHECK-NOT: alloca [4 x double]
; CHECK: %"buf'mi" = extractvalue { ptr, ptr, ptr, double } %truetape, 1
; CHECK: call {} %{{.*}}(ptr %buf, ptr %"buf'mi", ptr %tapeArg1)
