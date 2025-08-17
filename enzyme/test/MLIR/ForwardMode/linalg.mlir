// RUN: %eopt --enzyme %s | FileCheck %s

module {

  // Example 1: Element-wise addition (C = A + B)
  func.func @elementwise_add(
      %A: tensor<4xf32>,
      %B: tensor<4xf32>) -> tensor<4xf32> {
    
    %empty = tensor.empty() : tensor<4xf32>
    
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,    // Input A mapping
                       affine_map<(d0) -> (d0)>,    // Input B mapping  
                       affine_map<(d0) -> (d0)>],   // Output mapping
      iterator_types = ["parallel"]                 // Parallel iteration
    } ins(%A, %B : tensor<4xf32>, tensor<4xf32>) 
      outs(%empty : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):               // Block arguments
      %sum = arith.addf %a, %b : f32               // Element operation
      linalg.yield %sum : f32                      // Yield result
    } -> tensor<4xf32>
    
    func.return %result : tensor<4xf32>
  }

  func.func @dadd(%A: tensor<4xf32>, %dA: tensor<4xf32>,  %B: tensor<4xf32>, %dB: tensor<4xf32>) -> tensor<4xf32> {
    %r = enzyme.fwddiff @elementwise_add(%A, %dA, %B, %dB) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
    return %r : tensor<4xf32>
  }
}

// CHECK:  func.func private @fwddiffeelementwise_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %1:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%0, %cst : tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
// CHECK-NEXT:      %2 = arith.addf %in_0, %in_2 : f32
// CHECK-NEXT:      %3 = arith.addf %in, %in_1 : f32
// CHECK-NEXT:      linalg.yield %3, %2 : f32, f32
// CHECK-NEXT:    } -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NEXT:    return %1#1 : tensor<4xf32>
// CHECK-NEXT:  }
