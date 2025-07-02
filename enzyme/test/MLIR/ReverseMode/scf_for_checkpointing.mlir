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
    } {enzyme.enable_checkpointing = true}

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
// CHECK-NEXT:    %[[v1:.+]]:3 = scf.for %arg2 = %c0 to %c9 step %c3 iter_args(%arg3 = %arg0, %arg4 = %c0, %arg5 = %[[v0]]) -> (f32, index, tensor<3xf32>) {
// CHECK-NEXT:      %[[v3:.+]] = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg3) -> (f32) {
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %arg7, %arg7 : f32
// CHECK-NEXT:        %[[v6:.+]] = math.cos %[[v5]] : f32
// CHECK-NEXT:        scf.yield %[[v6]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %inserted = tensor.insert %arg3 into %arg5[%arg4] : tensor<3xf32>
// CHECK-NEXT:      %[[v4:.+]] = arith.addi %arg4, %c1 : index
// CHECK-NEXT:      scf.yield %[[v3]], %[[v4]], %inserted : f32, index, tensor<3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v2:.+]]:5 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %arg1, %arg4 = %[[zero]], %arg5 = %[[zero]], %arg6 = %[[zero]], %arg7 = %c2) -> (f32, f32, f32, f32, index) {
// CHECK-NEXT:      %extracted = tensor.extract %[[v1]]#2[%arg7] : tensor<3xf32>
// CHECK-NEXT:      %[[v3:.+]] = tensor.empty() : tensor<3xf32>
// CHECK-NEXT:      %[[v4:.+]] = tensor.empty() : tensor<3xf32>
// CHECK-NEXT:      %[[v5:.+]]:3 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %extracted, %arg10 = %[[v3]], %arg11 = %[[v4]]) -> (f32, tensor<3xf32>, tensor<3xf32>) {
// CHECK-NEXT:        %inserted = tensor.insert %arg9 into %arg10[%arg8] : tensor<3xf32>
// CHECK-NEXT:        %[[v8:.+]] = arith.mulf %arg9, %arg9 : f32
// CHECK-NEXT:        %inserted_0 = tensor.insert %[[v8]] into %arg11[%arg8] : tensor<3xf32>
// CHECK-NEXT:        %[[v9:.+]] = math.cos %[[v8]] : f32
// CHECK-NEXT:        scf.yield %[[v9]], %inserted, %inserted_0 : f32, tensor<3xf32>, tensor<3xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v6:.+]]:5 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %arg3, %arg10 = %arg4, %arg11 = %arg5, %arg12 = %arg6, %arg13 = %c2) -> (f32, f32, f32, f32, index) {
// CHECK-NEXT:        %[[v8:.+]] = arith.addf %arg10, %arg9 : f32
// CHECK-NEXT:        %extracted_0 = tensor.extract %[[v5]]#2[%arg13] : tensor<3xf32>
// CHECK-NEXT:        %[[v9:.+]] = math.sin %extracted_0 : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.negf %[[v9]] : f32
// CHECK-NEXT:        %[[v11:.+]] = arith.mulf %[[v8]], %[[v10]] : f32
// CHECK-NEXT:        %[[v12:.+]] = arith.addf %arg11, %[[v11]] : f32
// CHECK-NEXT:        %extracted_1 = tensor.extract %[[v5]]#1[%arg13] : tensor<3xf32>
// CHECK-NEXT:        %[[v13:.+]] = arith.mulf %[[v12]], %extracted_1 : f32
// CHECK-NEXT:        %[[v14:.+]] = arith.addf %arg12, %[[v13]] : f32
// CHECK-NEXT:        %[[v15:.+]] = arith.mulf %[[v12]], %extracted_1 : f32
// CHECK-NEXT:        %[[v16:.+]] = arith.addf %[[v14]], %[[v15]] : f32
// CHECK-NEXT:        %[[v17:.+]] = arith.subi %arg13, %c1 : index
// CHECK-NEXT:        scf.yield %[[v16]], %[[zero]], %[[zero]], %[[zero]], %[[v17]] : f32, f32, f32, f32, index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v7:.+]] = arith.subi %arg7, %c1 : index
// CHECK-NEXT:      scf.yield %[[v6]]#0, %[[v6]]#1, %[[v6]]#2, %[[v6]]#3, %[[v7]] : f32, f32, f32, f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[v2]]#0 : f32
// CHECK-NEXT:  }
