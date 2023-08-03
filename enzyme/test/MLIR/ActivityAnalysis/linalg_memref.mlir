#map = affine_map<() -> ()>
// Test that AA is detecting that linalg.generic propagates activity from %x to %out
// Requires a version of MLIR that has RegionBranchOpInterface removed from linalg.generic.
func.func @linalg_memref_freevar(%x: f64) -> f64 {
  %out = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%out: memref<f64>) {
    ^bb0(%bb0: f64):
      linalg.yield %x : f64
  }
  %res = memref.load %out[] : memref<f64>
  return %res : f64
}

// -----

// Test propagation from linalg.generic ins to outs
#map = affine_map<() -> ()>
func.func @linalg_memref_ins(%x: memref<f64>) -> f64 {
  %out = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%x: memref<f64>) outs(%out: memref<f64>) {
    ^bb0(%in: f64, %bbout: f64):
      linalg.yield %in : f64
  }
  %res = memref.load %out[] : memref<f64>
  return %res : f64
}
