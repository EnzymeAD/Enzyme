// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

func.func private @callee(%ptr : !llvm.ptr) 

// CHECK: points-to-pointer sets
// CHECK-NEXT: other points to unknown: 1
// CHECK-LABEL @fully_opaque_call
func.func @fully_opaque_call(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_rw
func.func @call_other_none_arg_rw(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = read,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_ro
func.func @call_other_none_arg_ro(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = write,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_wo
func.func @caller(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.nocapture}) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = write,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_wo_nocapture
func.func @call_other_none_arg_wo_nocapture(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.nocapture}) attributes {
  memory = #llvm.memory_effects<other = read, 
                                argMem = write,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {<unknown>}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_read_arg_wo_nocapture
func.func @call_other_read_arg_wo_nocapture(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = read, 
                                argMem = write,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {<unknown>}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_read_arg_wo
func.func @call_other_read_arg_wo(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.readonly}) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_rw_readonly
func.func @call_other_none_arg_rw_readonly(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}


// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.writeonly}) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL @call_other_none_arg_rw_writeonly
func.func @call_other_none_arg_rw_writeonly(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr, %ptr2 : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// TODO: the DAG below is due to using DenseMap and printing in no particular
// order, this should be fixed to have a deterministic order in tests.
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-2"> points to {distinct{{.*}}, distinct{{.*}}}
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-1"> points to {distinct{{.*}}, distinct{{.*}}}
// CHECK-NEXT: other points to unknown: 0
func.func @call_two_pointers_other_none_arg_rw_simple(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr, %ptr2 : !llvm.ptr {llvm.nocapture}) attributes {
  memory = #llvm.memory_effects<other = none, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// TODO: the DAG below is due to using DenseMap and printing in no particular
// order, this should be fixed to have a deterministic order in tests.
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-2"> points to {distinct{{\[}}[[ID]]]<"alloca-1">}
// CHECK-DAG: distinct{{\[}}[[ID]]]<"alloca-1"> points to {distinct{{\[}}[[ID]]]<"alloca-1">}
// CHECK-NEXT: other points to unknown: 0
// CHECK-LABEL: @call_two_pointers_other_none_arg_rw_nocapture
func.func @call_two_pointers_other_none_arg_rw_nocapture(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr {llvm.readonly}, %ptr2 : !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = read, 
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct[{{.+}}]<"alloca-2"> points to {<unknown>}
// CHECK-NEXT: other points to unknown: 0
func.func @call_two_pointers_other_read_arg_rw(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee() -> !llvm.ptr attributes {
  memory = #llvm.memory_effects<other = read,
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK: other points to unknown: 1
func.func @func_return_simple() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}) attributes {
  memory = #llvm.memory_effects<other = read,
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct[{{.+}}]<"func-return"> points to {<unknown>}
// CHECK-NEXT: other points to unknown: 0
func.func @func_return_noalias() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

// CHECK: tag "func-return" Unknown AC
// CHECK: "func-return" and "func-return": MayAlias
// CHECK: points-to-pointer sets
// CHECK-NEXT: other points to unknown: 1
func.func private @callee() -> (!llvm.ptr {llvm.noalias}, !llvm.ptr) attributes {
  memory = #llvm.memory_effects<other = read,
                                argMem = readwrite,
                                inaccessibleMem = none>
}

func.func @caller() -> !llvm.ptr {
  %0:2 = call @callee() {tag = "func-return"} : () -> (!llvm.ptr, !llvm.ptr)
  return %0#0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}) attributes {
  memory = #llvm.memory_effects<other = read,
                                argMem = readwrite,
                                inaccessibleMem = none>
}

// CHECK: "func-return-1" and "func-return-2": NoAlias
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"func-return-1"> points to {<unknown>}
// CHECK-DAG: distinct[{{.+}}]<"func-return-2"> points to {<unknown>}
// CHECK-NEXT: other points to unknown: 0
func.func @caller() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return-1"} : () -> !llvm.ptr
  %1 = call @callee() {tag = "func-return-2"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}


// -----

// CHECK: points-to-pointer sets
// CHECK-NOT: points to {<unknown>}
// CHECK: other points to unknown: 0
func.func private @callee(!llvm.ptr {llvm.readnone}) attributes {
    memory = #llvm.memory_effects<other = read,
                                  argMem = readwrite,
                                  inaccessibleMem = none>
}

func.func @caller(%arg0: !llvm.ptr {enzyme.tag = "argument"}) {
  call @callee(%arg0) : (!llvm.ptr) -> ()
  return
}
