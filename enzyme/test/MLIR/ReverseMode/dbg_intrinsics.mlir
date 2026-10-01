// RUN: %eopt %s --enzyme-wrap="infn=square outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s

// Debug intrinsics narrate the primal and compute nothing; -g must not stop
// differentiation. The intrinsic stays on the primal values; the adjoint has
// nothing to add for it.

#di_file = #llvm.di_file<"example.c" in "/tmp">
#di_cu = #llvm.di_compile_unit<id = distinct[1]<>, sourceLanguage = DW_LANG_C11, file = #di_file, producer = "clang", emissionKind = Full>
#di_sp = #llvm.di_subprogram<id = distinct[0]<>, compileUnit = #di_cu, scope = #di_file, name = "square", file = #di_file, line = 5, subprogramFlags = Definition>
#di_var = #llvm.di_local_variable<scope = #di_sp, name = "x", file = #di_file, line = 5, arg = 1>
llvm.func @square(%x: f64) -> f64 {
  llvm.intr.dbg.value #di_var = %x : f64
  %r = arith.mulf %x, %x : f64
  llvm.return %r : f64
}

// CHECK-LABEL: llvm.func @square(%arg0: f64, %arg1: f64) -> f64
// CHECK-NEXT:    llvm.intr.dbg.value #di_local_variable = %arg0 : f64
// CHECK-NEXT:    %[[m:.+]] = arith.mulf %arg1, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[a:.+]] = arith.addf %[[m]], %[[m]] fastmath<fast> : f64
// CHECK-NEXT:    llvm.return %[[a]] : f64
