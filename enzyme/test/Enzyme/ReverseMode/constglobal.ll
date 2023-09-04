; RUN: if [ %llvmver -eq 15 ]; then %opt < %s %loadEnzyme -enzyme -opaque-pointers=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -opaque-pointers=1 -S | FileCheck %s; fi

%"class.std::ios_base::Init" = type { i8 }
%class.Test = type { ptr }

$_ZN4TestC2Ev = comdat any

$_ZN4Test12test_virtualEv = comdat any

$_ZTV4Test = comdat any

$_ZTS4Test = comdat any

$_ZTI4Test = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZTV4Test = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Test, ptr @_ZN4Test12test_virtualEv] }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS4Test = linkonce_odr dso_local constant [6 x i8] c"4Test\00", comdat, align 1
@_ZTI4Test = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS4Test }, comdat, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_issue1394.cpp, ptr null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitC1Ev(ptr noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %0 = call i32 @__cxa_atexit(ptr @_ZNSt8ios_base4InitD1Ev, ptr @_ZStL8__ioinit, ptr @__dso_handle) #3
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2

; Function Attrs: nounwind
declare i32 @__cxa_atexit(ptr, ptr, ptr) #3

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef double @_Z8simulatev() #4 {
entry:
  %sys = alloca %class.Test, align 8
  call void @_ZN4TestC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %sys) #3
  ret double 0.000000e+00
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4TestC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV4Test, i32 0, inrange i32 0, i32 2), ptr %this1, align 8
  ret void
}

define dso_local void @main() #6 {
entry:
  %call = call noundef double @_Z17__enzyme_autodiffPv(ptr noundef @_Z8simulatev)
  ret void
}

declare noundef double @_Z17__enzyme_autodiffPv(ptr noundef) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4Test12test_virtualEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_issue1394.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { noinline uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }
attributes #4 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 16.0.6 (https://github.com/llvm/llvm-project.git 7cbf1a2591520c2491aa35339f227775f4d3adf6)"}

; CHECK: define internal void @diffe_Z8simulatev(double %differeturn) #7 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"sys'ipa" = alloca %class.Test, align 8
; CHECK-NEXT:   store %class.Test zeroinitializer, ptr %"sys'ipa", align 8
; CHECK-NEXT:   %sys = alloca %class.Test, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   call void @diffe_ZN4TestC2Ev(ptr align 8 %sys, ptr align 8 %"sys'ipa")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffe_ZN4TestC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this, ptr align 8 %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV4Test_shadow, i32 0, inrange i32 0, i32 2), ptr %"this'", align 8, !alias.scope !7, !noalias !10
; CHECK-NEXT:   store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV4Test, i32 0, inrange i32 0, i32 2), ptr %this, align 8, !alias.scope !10, !noalias !7
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal ptr @augmented__ZN4Test12test_virtualEv(ptr noundef nonnull align 8 dereferenceable(8) %this, ptr align 8 %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca ptr, align 8
; CHECK-NEXT:   store ptr null, ptr %0, align 8
; CHECK-NEXT:   %1 = load ptr, ptr %0, align 8
; CHECK-NEXT:   ret ptr %1
; CHECK-NEXT: }

; CHECK: define internal void @diffe_ZN4Test12test_virtualEv(ptr align 8 dereferenceable(8) %this, ptr align 8 %"this'", ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
