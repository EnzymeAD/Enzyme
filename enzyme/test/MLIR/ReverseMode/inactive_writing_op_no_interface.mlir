// RUN: %eopt %s --enzyme-wrap="infn=square outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

func.func @square(%x: f64) -> f64 {
  llvm.inline_asm has_side_effects "exit", "" : () -> ()
  %y = arith.mulf %x, %x : f64
  return %y : f64
}

// CHECK-LABEL: func.func @square(
// CHECK-SAME:      %[[X:.+]]: f64, %[[DY:.+]]: f64) -> f64
// CHECK:           llvm.inline_asm has_side_effects "exit", ""
// CHECK-NOT:       llvm.inline_asm
// CHECK:           arith.mulf %[[DY]], %[[X]]
// CHECK:           return
