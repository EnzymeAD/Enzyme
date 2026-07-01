// RUN: %eopt --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s --check-prefix=FWD
// RUN: %eopt --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s --check-prefix=REV

module {
  func.func @main(%arg0: f32) -> f32 {
    %0 = math.exp %arg0 : f32
    return %0 : f32
  }
}

// FWD: func.func @main(%[[x:.+]]: f32, %[[dx:.+]]: f32) -> (f32, f32) {
// FWD-NEXT:   %[[exp:.+]] = math.exp %[[x]] : f32
// FWD-NEXT:   %[[shadow:.+]] = arith.mulf %[[dx]], %[[exp]] : f32
// FWD-NEXT:   return %[[exp]], %[[shadow]] : f32, f32
// FWD-NEXT: }

// REV: func.func @main(%[[x:.+]]: f32, %[[dret:.+]]: f32) -> f32 {
// REV-NEXT:   %[[exp:.+]] = math.exp %[[x]] : f32
// REV-NEXT:   %[[darg:.+]] = arith.mulf %[[dret]], %[[exp]] : f32
// REV-NEXT:   return %[[darg]] : f32
// REV-NEXT: }
