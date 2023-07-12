#map = affine_map<() -> ()>
module {
  // Test that AA is detecting that linalg.generic propagates activity from %x to %out
  // TODO: won't work because of a bug with DenseDataFlowAnalysis
  func.func @linalg_memref(%x: f64) -> f64 {
    %out = memref.alloca() : memref<f64>
    linalg.generic { indexing_maps = [#map], iterator_types = [] } outs(%out: memref<f64>) {
      ^bb0(%bb0: f64):
        linalg.yield %x : f64
    }
    %res = memref.load %out[] : memref<f64>
    return %res : f64
  }

  func.func @dlinalg_memref(%x: f64) -> f64 {
    %dx = "enzyme.autodiff"(%x) {activity = [#enzyme<activity enzyme_out>], fn = @linalg_memref} : (f64) -> f64
    return %dx : f64
  }
}
