; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-global-activity -enzyme-globals-default-inactive -S | FileCheck %s; fi

; Separate compilation, reverse mode: a local object holding only constants
; (the descriptor flang builds to pass on a section of an inactive array)
; gets no shadow at a call through another module's derivative, also once
; reverse mode has moved it to the heap to keep it for the reverse pass.

@table = global [4 x double] zeroinitializer

declare void @ext(ptr, ptr)

define void @user(ptr %x) "enzyme_export_derivative"="reverse" {
  %box = alloca { ptr, i64 }
  store ptr @table, ptr %box
  %len = getelementptr inbounds { ptr, i64 }, ptr %box, i32 0, i32 1
  store i64 4, ptr %len
  call void @ext(ptr %x, ptr %box)
  ret void
}

; CHECK-DAG: @__enzyme_sep_rev_w1_c2_ext = external constant { ptr, ptr }
; CHECK: define internal ptr @augmented_user(
; CHECK: %box = {{.*}}@malloc({{.*}}!enzyme_fromstack
; CHECK: call { ptr } %{{[0-9]+}}(ptr %x, ptr %"x'", ptr %{{[^,']+}})
; CHECK: define internal void @diffeuser(
; CHECK: call {} %{{[0-9]+}}(ptr %x, ptr %"x'", ptr %{{[^,']+}}, ptr
