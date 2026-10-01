; RUN: if [ %llvmver -ge 16 ]; then printf "sq 1\next 1\n" > %t.params; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-inactive-params=%t.params -S | FileCheck %s; fi

; Separate compilation: parameters a whole-program plan declares inactive
; (-enzyme-inactive-params) get no shadow, in an exported derivative and at
; a call through another module's derivative alike, and are part of the
; derivative's name (_c<hex mask>), so both sides must agree to link.

define void @sq(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x, ptr %n) "enzyme_export_derivative"="reverse" {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %x
  ret void
}

declare void @ext(ptr, ptr)

define void @user(ptr %x, ptr %n) {
  call void @ext(ptr %x, ptr %n)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx, ptr %n) {
  call void (...) @__enzyme_autodiff(ptr @user, ptr %x, ptr %dx, ptr %n, ptr %n)
  ret void
}

; CHECK-DAG: @__enzyme_sep_rev_w1_c2_ext = external constant { ptr, ptr }
; CHECK-DAG: @__enzyme_sep_rev_w1_c2_sq = constant { ptr, ptr } { ptr @augmented_sq, ptr @diffesq }

; CHECK: define internal void @diffeuser(ptr %x, ptr %"x'", ptr %n, ptr %"n'")
; CHECK: %_augmented = call { ptr } %{{[0-9]+}}(ptr %x, ptr %"x'", ptr %n)
; CHECK: call {} %{{[0-9]+}}(ptr %x, ptr %"x'", ptr %n, ptr %subcache)

; CHECK: define internal ptr @augmented_sq(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}"enzyme_inactive" %n)
; CHECK: define internal void @diffesq(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}"enzyme_inactive" %n, ptr %tapeArg)
