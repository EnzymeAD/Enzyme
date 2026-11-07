; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::__1::basic_string" = type { ptr, i64, [8 x i8] }
; Kokkos::Impl::ParallelConstructName { const std::string *label; std::string default_name; }
%"struct.Kokkos::Impl::ParallelConstructName" = type { ptr, %"class.std::__1::basic_string" }
; Kokkos::Impl::ViewCtorProp<ExecSpace, std::string> { ExecSpace space; std::string label; }
%"struct.Kokkos::Impl::ViewCtorProp" = type { [16 x i8], %"class.std::__1::basic_string" }

@.label = private unnamed_addr constant [38 x i8] c"buf_with_non_sso_heap_allocated_label\00", align 1

declare noalias nonnull ptr @_Znwm(i64)
declare void @_ZdlPv(ptr)
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2EPKc(ptr, ptr)
declare void @consume_label(ptr nocapture readonly) nofree nounwind willreturn
declare void @begin_parallel_for(ptr nocapture) #0
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare double @__enzyme_autodiff(...)

; An in-function `operator new` for the long branch: the deallocation is paired
; with a visible allocation and handled by the regular allocation machinery.
define double @sso_string_label_kernel(double %x, i1 %is_long) {
entry:
  %str = alloca %"class.std::__1::basic_string", align 8
  %tag_ptr = getelementptr inbounds i8, ptr %str, i64 23
  br i1 %is_long, label %long.init, label %short.init

long.init:
  %heap = call noalias nonnull ptr @_Znwm(i64 32)
  store i8 65, ptr %heap, align 1
  %buf_slot = getelementptr inbounds %"class.std::__1::basic_string", ptr %str, i64 0, i32 0
  store ptr %heap, ptr %buf_slot, align 8
  store i8 -128, ptr %tag_ptr, align 1
  br label %use.str

short.init:
  store i8 0, ptr %tag_ptr, align 1
  br label %use.str

use.str:
  call void @consume_label(ptr readonly %str)
  %tag = load i8, ptr %tag_ptr, align 1
  %long_cond = icmp slt i8 %tag, 0
  br i1 %long_cond, label %long.dtor, label %done

long.dtor:
  %del_slot = getelementptr inbounds %"class.std::__1::basic_string", ptr %str, i64 0, i32 0
  %del_ptr = load ptr, ptr %del_slot, align 8
  call void @_ZdlPv(ptr %del_ptr)
  br label %done

done:
  %res = fmul double %x, %x
  ret double %res
}

; Out-of-line `basic_string(const char *)` constructors allocate the heap
; buffer inside the callee, while the inlined destructors free it here. The
; frees must stay in the forward pass (no allocation to pair them with exists
; in this function), including when
;  - the string lives at a non-zero offset of a wrapper struct (`%prop`),
;  - the string is copy-constructed (inlined SSO path) from a `const
;    std::string &` argument (`%copy`),
;  - the address of the string escapes into another local struct and is
;    reloaded from it (`%name`, as in Kokkos::Impl::ParallelConstructName).
define double @heap_string_label_kernel(double %x, ptr %arg_label) {
entry:
  %str = alloca %"class.std::__1::basic_string", align 8
  %copy = alloca %"class.std::__1::basic_string", align 8
  %prop = alloca %"struct.Kokkos::Impl::ViewCtorProp", align 8
  %name = alloca %"struct.Kokkos::Impl::ParallelConstructName", align 8
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2EPKc(ptr %str, ptr @.label)
  %prop_label = getelementptr inbounds %"struct.Kokkos::Impl::ViewCtorProp", ptr %prop, i64 0, i32 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2EPKc(ptr %prop_label, ptr @.label)
  call void @llvm.memcpy.p0.p0.i64(ptr %copy, ptr %arg_label, i64 24, i1 false)
  store ptr %str, ptr %name, align 8
  call void @begin_parallel_for(ptr %name)
  %reloaded = load ptr, ptr %name, align 8
  call void @consume_label(ptr %reloaded)
  call void @consume_label(ptr %copy)
  call void @consume_label(ptr %prop)
  %res = fmul double %x, %x
  %tag_ptr = getelementptr inbounds i8, ptr %str, i64 23
  %tag = load i8, ptr %tag_ptr, align 1
  %long = icmp slt i8 %tag, 0
  br i1 %long, label %dtor.str, label %dtor.copy.check

dtor.str:
  %buf = load ptr, ptr %str, align 8
  call void @_ZdlPv(ptr %buf)
  br label %dtor.copy.check

dtor.copy.check:
  %ctag_ptr = getelementptr inbounds i8, ptr %copy, i64 23
  %ctag = load i8, ptr %ctag_ptr, align 1
  %clong = icmp slt i8 %ctag, 0
  br i1 %clong, label %dtor.copy, label %dtor.prop.check

dtor.copy:
  %cbuf = load ptr, ptr %copy, align 8
  call void @_ZdlPv(ptr %cbuf)
  br label %dtor.prop.check

dtor.prop.check:
  %ptag_ptr = getelementptr inbounds i8, ptr %prop, i64 39
  %ptag = load i8, ptr %ptag_ptr, align 1
  %plong = icmp slt i8 %ptag, 0
  br i1 %plong, label %dtor.prop, label %done

dtor.prop:
  %pbuf = load ptr, ptr %prop_label, align 8
  call void @_ZdlPv(ptr %pbuf)
  br label %done

done:
  ret double %res
}

define double @caller(double %x, i1 %is_long) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @sso_string_label_kernel, metadata !"enzyme_out", double %x, metadata !"enzyme_const", i1 %is_long)
  ret double %dx
}

define double @caller2(double %x, ptr %label) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @heap_string_label_kernel, metadata !"enzyme_out", double %x, metadata !"enzyme_const", ptr %label)
  ret double %dx
}

attributes #0 = { nofree nounwind willreturn "enzyme_inactive" "enzyme_no_escaping_allocation" }

; CHECK-LABEL: define internal { double } @diffesso_string_label_kernel(
; CHECK: call {{.*}}ptr @_Znwm(i64 {{.*}}32)
; CHECK: call void @_ZdlPv(
; CHECK-NOT: call void @_ZdlPv(
; CHECK: fadd fast double

; CHECK-LABEL: define internal { double } @diffeheap_string_label_kernel(
; CHECK-NOT: @nofree_
; CHECK: call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2EPKc(ptr %str, ptr @.label)
; CHECK: call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2EPKc(ptr %prop_label, ptr @.label)
; CHECK: store ptr %str, ptr %name
; CHECK: %reloaded = load ptr, ptr %name
; CHECK: call void @_ZdlPv(ptr %buf)
; CHECK: call void @_ZdlPv(ptr %cbuf)
; CHECK: call void @_ZdlPv(ptr %pbuf)
; CHECK: fadd fast double
