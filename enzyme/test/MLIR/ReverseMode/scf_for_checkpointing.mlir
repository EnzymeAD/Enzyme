// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %sum_iter : f32
      %cos_next = math.cos %sum_next : f32
      scf.yield %cos_next : f32
    } {enzyme.enable_checkpointing = true, enzyme.cache_use_tensor}

    return %sum : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[v0:.+]] = tensor.empty() : tensor<3xf32>
// CHECK-NEXT:    %[[v1:.+]]:2 = scf.for %arg2 = %c0 to %c9 step %c3 iter_args(%arg3 = %arg0, %[[arg5:.+]] = %[[v0]]) -> (f32, tensor<3xf32>) {
// CHECK-NEXT:      %[[idx:.+]] = arith.divui %arg2, %c3 : index
// CHECK-NEXT:      %inserted = tensor.insert %arg3 into %[[arg5]][%[[idx]]] : tensor<3xf32>
// CHECK-NEXT:      %[[v3:.+]] = scf.for %[[arg6:.+]] = %c0 to %c3 step %c1 iter_args(%[[arg7:.+]] = %arg3) -> (f32) {
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %[[arg7]], %[[arg7]] : f32
// CHECK-NEXT:        %[[v6:.+]] = math.cos %[[v5]] : f32
// CHECK-NEXT:        scf.yield %[[v6]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v3]], %inserted : f32, tensor<3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v2:.+]]:4 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %arg1, %arg4 = %[[zero]], %arg5 = %[[zero]], %arg6 = %[[zero]]) -> (f32, f32, f32, f32) {
// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %c2, %arg2 : index
// CHECK-NEXT:      %extracted = tensor.extract %[[v1]]#1[%[[ridx]]] : tensor<3xf32>
// CHECK-NEXT:      %[[v3:.+]] = tensor.empty() : tensor<3xf32>
// CHECK-NEXT:      %[[v5:.+]]:2 = scf.for %[[arg8:.+]] = %c0 to %c3 step %c1 iter_args(%[[arg9:.+]] = %extracted, %[[arg10:.+]] = %[[v3]]) -> (f32, tensor<3xf32>) {
// CHECK-NEXT:        %inserted = tensor.insert %[[arg9]] into %[[arg10:.+]][%[[arg8]]] : tensor<3xf32>
// CHECK-NEXT:        %[[v8:.+]] = arith.mulf %[[arg9]], %[[arg9]] : f32
// CHECK-NEXT:        %[[v9:.+]] = math.cos %[[v8]] : f32
// CHECK-NEXT:        scf.yield %[[v9]], %inserted : f32, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v6:.+]]:4 = scf.for %[[arg8:.+]] = %c0 to %c3 step %c1 iter_args(%[[arg9:.+]] = %arg3, %[[arg10:.+]] = %arg4, %[[arg11:.+]] = %arg5, %[[arg12:.+]] = %arg6) -> (f32, f32, f32, f32) {
// CHECK-NEXT:        %[[ridx2:.+]] = arith.subi %c2, %[[arg8]] : index
// CHECK-NEXT:        %extracted_0 = tensor.extract %[[v5]]#1[%[[ridx2]]] : tensor<3xf32>
// CHECK-NEXT:        %[[r8:.+]] = arith.mulf %extracted_0, %extracted_0 : f32
// CHECK-NEXT:        %[[v8:.+]] = arith.addf %[[arg10]], %[[arg9]] : f32
// CHECK-NEXT:        %[[v9:.+]] = math.sin %[[r8]] : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.negf %[[v9]] : f32
// CHECK-NEXT:        %[[v11:.+]] = arith.mulf %[[v8]], %[[v10]] : f32
// CHECK-NEXT:        %[[v12:.+]] = arith.addf %[[arg11]], %[[v11]] : f32
// CHECK-NEXT:        %[[v13:.+]] = arith.mulf %[[v12]], %extracted_0 : f32
// CHECK-NEXT:        %[[v14:.+]] = arith.addf %[[arg12]], %[[v13]] : f32
// CHECK-NEXT:        %[[v15:.+]] = arith.mulf %[[v12]], %extracted_0 : f32
// CHECK-NEXT:        %[[v16:.+]] = arith.addf %[[v14]], %[[v15]] : f32
// CHECK-NEXT:        scf.yield %[[v16]], %[[zero]], %[[zero]], %[[zero]] : f32, f32, f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v6]]#0, %[[v6]]#1, %[[v6]]#2, %[[v6]]#3 : f32, f32, f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[v2]]#0 : f32
// CHECK-NEXT:  }
