// RUN: %eopt --enzyme --split-input-file --verify-diagnostics %s

// Float8 is a valid memref element type, but has no AutoDiffTypeInterface model.
// Refuse the active store rather than silently omitting its derivative.
func.func @store_float8(%m: memref<1xf8E4M3FN>, %v: f32) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{AutoDiffTypeInterface not implemented for active type 'f8E4M3FN'}}
  %n = arith.truncf %v : f32 to f8E4M3FN
  memref.store %n, %m[%c0] : memref<1xf8E4M3FN>
  return
}

func.func @dstore_float8(%m: memref<1xf8E4M3FN>, %dm: memref<1xf8E4M3FN>, %v: f32) -> f32 {
  %r = enzyme.autodiff @store_float8(%m, %dm, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (memref<1xf8E4M3FN>, memref<1xf8E4M3FN>, f32) -> f32
  return %r : f32
}

// -----

func.func @store_float8(%m: memref<1xf8E4M3FN>, %v: f32) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{AutoDiffTypeInterface not implemented for active type 'f8E4M3FN'}}
  %n = arith.truncf %v : f32 to f8E4M3FN
  memref.store %n, %m[%c0] : memref<1xf8E4M3FN>
  return
}

func.func @fstore_float8(%m: memref<1xf8E4M3FN>, %dm: memref<1xf8E4M3FN>, %v: f32, %dv: f32) {
  enzyme.fwddiff @store_float8(%m, %dm, %v, %dv) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[] }> : (memref<1xf8E4M3FN>, memref<1xf8E4M3FN>, f32, f32) -> ()
  return
}
