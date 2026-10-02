; RUN: if [ %llvmver -ge 16 ]; then printf "ext\t0\t{[-1]:Pointer, [-1,0]:Float@double}\next\t1\t{[-1]:Pointer, [-1,0]:Float@double}\n" > %t.types; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-param-types=%t.types -S | FileCheck %s; fi

; Separate compilation: the types declared on the parameters of a function
; defined in another module (-enzyme-param-types, from a whole-program plan)
; go on the calls that pass it local memory of the caller. Here they are the only thing saying that the local
; %acc holds a double: it is read before it is written (as after LICM
; promoted it) and then only passed to @ext. Without them the phi merging
; the load with a float is untyped ("Cannot deduce type of phi", MITgcm
; mon_ke).

declare void @ext(ptr, ptr)

define void @user(ptr %x, i1 %c) {
entry:
  %acc = alloca double, align 8
  %init = load double, ptr %acc, align 1
  br i1 %c, label %body, label %exit

body:
  %v = load double, ptr %x, align 8
  %s = fmul double %v, %v
  br label %exit

exit:
  %r = phi double [ %init, %entry ], [ %s, %body ]
  store double %r, ptr %acc, align 1
  call void @ext(ptr %acc, ptr %x)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx, i1 %c) {
  call void (...) @__enzyme_autodiff(ptr @user, ptr %x, ptr %dx, i1 %c)
  ret void
}

; CHECK: define void @user(ptr %x, i1 %c)
; CHECK: call void @ext(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %acc, ptr %x)
; CHECK: define internal void @diffeuser(ptr %x, ptr %"x'", i1 %c)
