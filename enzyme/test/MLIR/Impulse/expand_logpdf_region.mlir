// RUN: %eopt --expand-impulse %s | FileCheck %s

// The infer_region is written in generic op form because its pretty assembly
// format does not round-trip without a `fn`. The logpdf body uses

module {
  func.func private @model_logpdf(%position : tensor<8xf64>,
                                  %B : tensor<8xf64>) -> tensor<f64> {
    %nh = arith.constant dense<-5.000000e-01> : tensor<f64>
    %s = arith.mulf %position, %B : tensor<8xf64>
    %d = impulse.dot %s, %s {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %r = arith.mulf %d, %nh : tensor<f64>
    return %r : tensor<f64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK-DAG: %[[MAGIC:.+]] = arith.constant dense<3.300000e+00> : tensor<8xf64>
  // CHECK: %[[U0RS:.+]] = impulse.reshape %{{.+}} : (tensor<1x8xf64>) -> tensor<8xf64>
  // CHECK-NEXT: %[[U0S:.+]] = arith.mulf %[[U0RS]], %arg1 : tensor<8xf64>
  // CHECK-NEXT: %[[U0B:.+]] = arith.mulf %[[U0S]], %[[MAGIC]] : tensor<8xf64>
  // CHECK-NEXT: %[[U0D:.+]] = impulse.dot %[[U0B]], %[[U0B]]
  // CHECK-NEXT: %[[U0M:.+]] = arith.mulf %[[U0D]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %{{.+}} = arith.negf %[[U0M]] : tensor<f64>
  // CHECK: enzyme.autodiff_region
  // CHECK-NEXT: ^bb0(%[[QARG:.+]]: tensor<1x8xf64>):
  // CHECK-NEXT: %[[GSL:.+]] = impulse.slice %[[QARG]]
  // CHECK-NEXT: %[[GRS:.+]] = impulse.reshape %[[GSL]] : (tensor<1x8xf64>) -> tensor<8xf64>
  // CHECK-NEXT: %[[GS:.+]] = arith.mulf %[[GRS]], %arg1 : tensor<8xf64>
  // CHECK-NEXT: %[[GB:.+]] = arith.mulf %[[GS]], %[[MAGIC]] : tensor<8xf64>
  // CHECK-NEXT: %[[GD:.+]] = impulse.dot %[[GB]], %[[GB]]
  // CHECK-NEXT: %[[GM:.+]] = arith.mulf %[[GD]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[GN:.+]] = arith.negf %[[GM]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[GN]]
  // CHECK: impulse.while
  // CHECK-NOT: @model_logpdf
  func.func @main(%rng : tensor<2xui64>,
                  %B : tensor<8xf64>)
      -> (tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<0.0> : tensor<1x8xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %imm = arith.constant dense<1.0> : tensor<8xf64>
    %res:8 = "impulse.infer_region"(%rng, %B, %imm, %step_size, %init_pos) <{
      all_addresses = [],
      logpdf_fn = @model_logpdf,
      name = "",
      num_position_args = 1 : i64,
      num_samples = 1 : i64,
      num_warmup = 0 : i64,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1.000000e+03 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      operandSegmentSizes = array<i32: 2, 0, 0, 1, 1, 1, 0, 0, 0>,
      position_size = 8 : i64,
      selection = [],
      thinning = 1 : i64
    }> ({
    }, {
    ^bb0(%pos : tensor<8xf64>):
      %magic = arith.constant dense<3.300000e+00> : tensor<8xf64>
      %scaled = arith.mulf %pos, %B : tensor<8xf64>
      %biased = arith.mulf %scaled, %magic : tensor<8xf64>
      %d = impulse.dot %biased, %biased {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
      %nh = arith.constant dense<-5.000000e-01> : tensor<f64>
      %r = arith.mulf %d, %nh : tensor<f64>
      "impulse.yield"(%r) : (tensor<f64>) -> ()
    }) : (tensor<2xui64>, tensor<8xf64>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x8xf64>, tensor<1x8xf64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
