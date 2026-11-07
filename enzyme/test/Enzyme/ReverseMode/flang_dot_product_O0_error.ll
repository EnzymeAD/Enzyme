; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S 2>&1 | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-unsupported-diag-fallback -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-unsupported-diag-fallback -passes="enzyme" -S 2>&1 | FileCheck %s

%Descriptor = type { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }

@source_file = private constant [9 x i8] c"test.f90\00"

declare float @_FortranADotProductReal4(ptr, ptr, ptr, i32)
declare void @__enzyme_autodiff(...)

define float @dot(ptr %n_ptr, ptr %a, ptr %b, ptr %c_ptr) {
entry:
  %desc_a = alloca %Descriptor, align 8
  %desc_b = alloca %Descriptor, align 8
  store ptr %a, ptr %desc_a, align 8
  store ptr %b, ptr %desc_b, align 8
  %prod = call float @_FortranADotProductReal4(ptr %desc_a, ptr %desc_b, ptr @source_file, i32 10)
  %c = load float, ptr %c_ptr, align 4
  %res = fadd float %prod, %c
  ret float %res
}

define void @caller(ptr %n_ptr, ptr %a, ptr %da, ptr %b, ptr %db, ptr %c, ptr %dc) {
entry:
  call void (...) @__enzyme_autodiff(ptr @dot, metadata !"enzyme_const", ptr %n_ptr, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %b, ptr %db, metadata !"enzyme_dup", ptr %c, ptr %dc)
  ret void
}

; CHECK: error: {{.*}}Enzyme: No reverse pass found for _FortranADotProductReal4
