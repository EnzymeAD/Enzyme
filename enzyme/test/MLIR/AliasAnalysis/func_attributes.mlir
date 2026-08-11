// RUN: %eopt --test-print-alias-analysis --split-input-file %s | FileCheck %s

func.func private @callee(%ptr : !llvm.ptr) 

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct[{{.*}}]<"entry"> points to {<unknown>}
// CHECK-LABEL @fully_opaque_call
func.func @fully_opaque_call(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-LABEL @call_other_none_arg_rw
func.func @call_other_none_arg_rw(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = read,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: <empty>
// CHECK-LABEL @call_other_none_arg_ro
func.func @call_other_none_arg_ro(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = write,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-LABEL @call_other_none_arg_wo
func.func @call_other_none_arg_wo(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.nocapture}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = write,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: <empty>
// CHECK-LABEL @call_other_none_arg_wo_nocapture
func.func @call_other_none_arg_wo_nocapture(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.nocapture}) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = write,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {<unknown>}
// CHECK-LABEL @call_other_read_arg_wo_nocapture
func.func @call_other_read_arg_wo_nocapture(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = write,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {<unknown>}
// CHECK-LABEL @call_other_read_arg_wo
func.func @call_other_read_arg_wo(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.readonly}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK: <empty>
// CHECK-LABEL @call_other_none_arg_rw_readonly
func.func @call_other_none_arg_rw_readonly(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}


// -----

func.func private @callee(%ptr : !llvm.ptr {llvm.writeonly}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct{{\[}}[[ID:.+]]]<"entry"> points to {distinct{{\[}}[[ID]]]<"entry">}
// CHECK-LABEL @call_other_none_arg_rw_writeonly
func.func @call_other_none_arg_rw_writeonly(%input: !llvm.ptr {enzyme.tag = "input"}) {
  call @callee(%input) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr, %ptr2 : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// TODO: the DAG below is due to using DenseMap and printing in no particular
// order, this should be fixed to have a deterministic order in tests.
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-2"> points to {distinct{{.*}}, distinct{{.*}}}
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-1"> points to {distinct{{.*}}, distinct{{.*}}}
func.func @call_two_pointers_other_none_arg_rw_simple(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr, %ptr2 : !llvm.ptr {llvm.nocapture}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// TODO: the DAG below is due to using DenseMap and printing in no particular
// order, this should be fixed to have a deterministic order in tests.
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct{{\[}}[[ID:.+]]]<"alloca-2"> points to {distinct{{\[}}[[ID]]]<"alloca-1">}
// CHECK-DAG: distinct{{\[}}[[ID]]]<"alloca-1"> points to {distinct{{\[}}[[ID]]]<"alloca-1">}
// CHECK-LABEL: @call_two_pointers_other_none_arg_rw_nocapture
func.func @call_two_pointers_other_none_arg_rw_nocapture(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee(%ptr1 : !llvm.ptr {llvm.readonly}, %ptr2 : !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// TODO: the DAG below is due to using DenseMap and printing in no particular
// order, this should be fixed to have a deterministic order in tests.
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"alloca-2"> points to {<unknown>}
// CHECK-DAG: distinct[{{.+}}]<"alloca-1"> points to {}
func.func @call_two_pointers_other_read_arg_rw(%sz: i64) {
  %0 = llvm.alloca %sz x i8 { tag = "alloca-1" } : (i64) -> !llvm.ptr
  %1 = llvm.alloca %sz x i8 { tag = "alloca-2" } : (i64) -> !llvm.ptr
  call @callee(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// -----

func.func private @callee() -> !llvm.ptr attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK: <empty>
func.func @func_return_simple() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct[{{.+}}]<"func-return0"> points to {<unknown>}
func.func @func_return_noalias() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

// CHECK: tag "func-return" Unknown AC
// CHECK: "func-return" and "func-return": MayAlias
// CHECK: points-to-pointer sets
// CHECK-NEXT: distinct[{{.*}}]<"func-return0"> points to {<unknown>}
func.func private @callee() -> (!llvm.ptr {llvm.noalias}, !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

func.func @caller() -> !llvm.ptr {
  %0:2 = call @callee() {tag = "func-return"} : () -> (!llvm.ptr, !llvm.ptr)
  return %0#0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: "func-1-return" and "func-2-return": NoAlias
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"func-1-return0"> points to {<unknown>}
// CHECK-DAG: distinct[{{.+}}]<"func-2-return0"> points to {<unknown>}
func.func @caller() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-1-return"} : () -> !llvm.ptr
  %1 = call @callee() {tag = "func-2-return"} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}


// -----

// CHECK: points-to-pointer sets
// CHECK: <empty>
func.func private @callee(!llvm.ptr {llvm.readnone}) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

func.func @caller(%arg0: !llvm.ptr {enzyme.tag = "argument"}) {
  call @callee(%arg0) : (!llvm.ptr) -> ()
  return
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"alloca"> points to {distinct[{{.+}}]<"func-return0">}
// CHECK-DAG: distinct[{{.+}}]<"func-return0"> points to {<unknown>}
func.func @func_return_noalias_stored() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca"}: (i64) -> !llvm.ptr
  llvm.store %0, %1 : !llvm.ptr, !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = read,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: points-to-pointer sets
// CHECK: distinct[{{.+}}]<"alloca"> points to {<unknown>}
func.func @func_return_stored() -> !llvm.ptr {
  %0 = call @callee() {tag = "func-return"} : () -> !llvm.ptr
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca"}: (i64) -> !llvm.ptr
  llvm.store %0, %1 : !llvm.ptr, !llvm.ptr
  return %0 : !llvm.ptr
}

// -----


func.func private @callee() -> (!llvm.ptr, !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// TODO: The two results may alias, but we can't really
// differentiate them with current printing.
// CHECK: "func-return" and "func-return": MayAlias

// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"alloca-1"> points to {distinct[{{.+}}]<"func-return-common">}
// CHECK-DAG: distinct[{{.+}}]<"alloca-2"> points to {distinct[{{.+}}]<"func-return-common">}
func.func @func_return_multiple() -> !llvm.ptr {
  %0:2 = call @callee() {tag = "func-return"} : () -> (!llvm.ptr, !llvm.ptr)
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-1"}: (i64) -> !llvm.ptr
  llvm.store %0#0, %1 : !llvm.ptr, !llvm.ptr
  %2 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-2"}: (i64) -> !llvm.ptr
  llvm.store %0#1, %2 : !llvm.ptr, !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

func.func private @callee() -> (!llvm.ptr {llvm.noalias}, !llvm.ptr {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = readwrite,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// TODO: The two results are known not to alias, but we can't really
// differentiate them with current printing.
// CHECK: "func-return" and "func-return": NoAlias

// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.+}}]<"alloca-1"> points to {distinct[{{.+}}]<"func-return0">}
// CHECK-DAG: distinct[{{.+}}]<"alloca-2"> points to {distinct[{{.+}}]<"func-return1">}
func.func @func_return_noalias() -> !llvm.ptr {
  %0:2 = call @callee() {tag = "func-return"} : () -> (!llvm.ptr, !llvm.ptr)
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-1"}: (i64) -> !llvm.ptr
  llvm.store %0#0, %1 : !llvm.ptr, !llvm.ptr
  %2 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-2"}: (i64) -> !llvm.ptr
  llvm.store %0#1, %2 : !llvm.ptr, !llvm.ptr
  return %0 : !llvm.ptr
}

// -----


func.func private @callee(!llvm.ptr, !llvm.ptr {llvm.nocapture}) -> (!llvm.ptr, !llvm.ptr) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = read,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// CHECK: "func-return" and "func-return": MayAlias

// Returned value can only point to the classes of captured pointers, i.e. arg0.
// However, returned value itself may alias with any argument, so pointers that
// stored the return value may point to any of the arg0, arg1 and the returned
// value itself.
//
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.*}}]<"func-return-common"> points to {distinct[{{.*}}]<"arg0">}
// CHECK-DAG: distinct[{{.*}}]<"alloca-1"> points to {
// CHECK-DAG: distinct[{{.*}}]<"func-return-common">
// CHECK-DAG: distinct[{{.*}}]<"arg0">
// CHECK-DAG: distinct[{{.*}}]<"arg1">
func.func @multi_operand_result(%arg0: !llvm.ptr {enzyme.tag = "arg0", llvm.noalias},
                               %arg1: !llvm.ptr {enzyme.tag = "arg1", llvm.nocapture, llvm.noalias}) -> !llvm.ptr {
  %0:2 = call @callee(%arg0, %arg1) {tag = "func-return"} : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-1"}: (i64) -> !llvm.ptr
  llvm.store %0#0, %1 : !llvm.ptr, !llvm.ptr
  %2 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-2"}: (i64) -> !llvm.ptr
  llvm.store %0#1, %2 : !llvm.ptr, !llvm.ptr
  return %0#0 : !llvm.ptr
}

// -----


func.func private @callee(!llvm.ptr, !llvm.ptr {llvm.nocapture}) -> (!llvm.ptr {llvm.noalias}, !llvm.ptr {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = read,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// Returned values can pointer to something that was captured, but belong to
// diferent classes and don't alias operand pointers.
//
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.*}}]<"alloca-1"> points to {distinct[{{.*}}]<"func-return0">}
// CHECK-DAG: distinct[{{.*}}]<"func-return1"> points to {distinct[{{.*}}]<"arg0">}
// CHECK-DAG: distinct[{{.*}}]<"func-return0"> points to {distinct[{{.*}}]<"arg0">}
// CHECK-DAG: distinct[{{.*}}]<"alloca-2"> points to {distinct[{{.*}}]<"func-return1">}
func.func @multi_operand_result(%arg0: !llvm.ptr {enzyme.tag = "arg0", llvm.noalias},
                                %arg1: !llvm.ptr {enzyme.tag = "arg1", llvm.nocapture, llvm.noalias}) -> !llvm.ptr {
  %0:2 = call @callee(%arg0, %arg1) {tag = "func-return"} : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-1"}: (i64) -> !llvm.ptr
  llvm.store %0#0, %1 : !llvm.ptr, !llvm.ptr
  %2 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-2"}: (i64) -> !llvm.ptr
  llvm.store %0#1, %2 : !llvm.ptr, !llvm.ptr
  return %0#0 : !llvm.ptr
}

// -----

func.func private @callee(!llvm.ptr, !llvm.ptr {llvm.nocapture})
    -> (!llvm.ptr, !llvm.ptr  {llvm.noalias}) attributes {
  memory_effects = #llvm.memory_effects<other = none,
                                        argMem = read,
                                        inaccessibleMem = none,
                                        errnoMem = none,
                                        targetMem0 = none,
                                        targetMem1 = none>
}

// Returned values can pointer to something that was captured, but belong to
// diferent classes and don't alias operand pointers.
//
// CHECK: points-to-pointer sets
// CHECK-DAG: distinct[{{.*}}]<"func-return-common"> points to {distinct[{{.*}}]<"arg0">}
// CHECK-DAG: distinct[{{.*}}]<"func-return1"> points to {distinct[{{.*}}]<"arg0">}
// CHECK-DAG: distinct[{{.*}}]<"alloca-1"> points to
// TODO: the current way of checking is fundamentally broken because of printing
// in hashmap order, we'd need a nested CHECK-DAG for this.
// CHECK-DAG: distinct[{{.*}}]<"alloca-2"> points to {distinct[{{.*}}]<"func-return1">}
// CHECK: #distinct
func.func @multi_operand_result(%arg0: !llvm.ptr {enzyme.tag = "arg0", llvm.noalias},
                                %arg1: !llvm.ptr {enzyme.tag = "arg1", llvm.nocapture, llvm.noalias}) -> !llvm.ptr {
  %0:2 = call @callee(%arg0, %arg1) {tag = "func-return"} : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %c1 = arith.constant 1 : i64
  %1 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-1"}: (i64) -> !llvm.ptr
  llvm.store %0#0, %1 : !llvm.ptr, !llvm.ptr
  %2 = llvm.alloca %c1 x !llvm.ptr {tag = "alloca-2"}: (i64) -> !llvm.ptr
  llvm.store %0#1, %2 : !llvm.ptr, !llvm.ptr
  return %0#0 : !llvm.ptr
}
