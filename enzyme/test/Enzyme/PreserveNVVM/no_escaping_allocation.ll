; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm" -S | FileCheck %s

; A function registered through an __enzyme_no_escaping_allocation global,
; or annotated enzyme_no_escaping_allocation by the clang plugin, gets the
; enzyme_no_escaping_allocation attribute, and the registration is removed.

source_filename = "no_escaping_allocation.ll"

@.str.ann = private unnamed_addr constant [30 x i8] c"enzyme_no_escaping_allocation\00", section "llvm.metadata"
@.str.file = private unnamed_addr constant [26 x i8] c"no_escaping_allocation.ll\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { i8*, i8*, i8*, i32, i8* }] [
  { i8*, i8*, i8*, i32, i8* } { i8* bitcast (void (double*)* @annotated to i8*), i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.ann, i32 0, i32 0), i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.file, i32 0, i32 0), i32 1, i8* null }
], section "llvm.metadata"

declare void @registered(double*)

declare void @registered_array(double*)

declare void @annotated(double*)

declare void @unmarked(double*)

@__enzyme_no_escaping_allocation = global i8* bitcast (void (double*)* @registered to i8*)
@__enzyme_no_escaping_allocation_arr = global [1 x i8*] [i8* bitcast (void (double*)* @registered_array to i8*)]

define void @use(double* %x) {
entry:
  call void @registered(double* %x)
  call void @registered_array(double* %x)
  call void @annotated(double* %x)
  call void @unmarked(double* %x)
  ret void
}

; CHECK-NOT: @__enzyme_no_escaping_allocation
; CHECK: declare void @registered(double*) #[[ATTR:[0-9]+]]
; CHECK: declare void @registered_array(double*) #[[ATTR]]
; CHECK: declare void @annotated(double*) #[[ATTR]]
; CHECK: declare void @unmarked(double*){{$}}
; CHECK: attributes #[[ATTR]] = { {{.*}}"enzyme_no_escaping_allocation"
