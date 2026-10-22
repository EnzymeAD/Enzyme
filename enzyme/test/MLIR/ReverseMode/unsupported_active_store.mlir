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

// -----

// A vector interface alone does not supply an interface for its Float8 element.
func.func @store_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{AutoDiffTypeInterface not implemented for active type 'vector<4xf8E4M3FN>'}}
  %n = arith.truncf %v : vector<4xf32> to vector<4xf8E4M3FN>
  memref.store %n, %m[%c0] : memref<1xvector<4xf8E4M3FN>>
  return
}

func.func @dstore_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %dm: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf32>) -> vector<4xf32> {
  %r = enzyme.autodiff @store_vector_float8(%m, %dm, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (memref<1xvector<4xf8E4M3FN>>, memref<1xvector<4xf8E4M3FN>>, vector<4xf32>) -> vector<4xf32>
  return %r : vector<4xf32>
}

// -----

func.func @store_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{AutoDiffTypeInterface not implemented for active type 'vector<4xf8E4M3FN>'}}
  %n = arith.truncf %v : vector<4xf32> to vector<4xf8E4M3FN>
  memref.store %n, %m[%c0] : memref<1xvector<4xf8E4M3FN>>
  return
}

func.func @fstore_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %dm: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf32>, %dv: vector<4xf32>) {
  enzyme.fwddiff @store_vector_float8(%m, %dm, %v, %dv) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[] }> : (memref<1xvector<4xf8E4M3FN>>, memref<1xvector<4xf8E4M3FN>>, vector<4xf32>, vector<4xf32>) -> ()
  return
}

// -----

// Validate block arguments as well as intermediate results.
// expected-error @below {{AutoDiffTypeInterface not implemented for active type 'vector<4xf8E4M3FN>'}}
func.func @store_vector_float8_arg(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) {
  %c0 = arith.constant 0 : index
  memref.store %v, %m[%c0] : memref<1xvector<4xf8E4M3FN>>
  return
}

func.func @dstore_vector_float8_arg(%m: memref<1xvector<4xf8E4M3FN>>, %dm: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) -> vector<4xf8E4M3FN> {
  %r = enzyme.autodiff @store_vector_float8_arg(%m, %dm, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (memref<1xvector<4xf8E4M3FN>>, memref<1xvector<4xf8E4M3FN>>, vector<4xf8E4M3FN>) -> vector<4xf8E4M3FN>
  return %r : vector<4xf8E4M3FN>
}

// -----

// Unsupported inactive vector elements must not require a derivative model.
func.func @store_inactive_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) {
  %c0 = arith.constant 0 : index
  memref.store %v, %m[%c0] : memref<1xvector<4xf8E4M3FN>>
  return
}

func.func @dstore_inactive_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) {
  enzyme.autodiff @store_inactive_vector_float8(%m, %v) <{ activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (memref<1xvector<4xf8E4M3FN>>, vector<4xf8E4M3FN>) -> ()
  return
}

// -----

// LLVM pointer vectors are valid IR but have no vector null-value model.
// expected-error @below {{AutoDiffTypeInterface not implemented for active type 'vector<4x!llvm.ptr>'}}
func.func @store_vector_pointer(%p: !llvm.ptr, %v: vector<4x!llvm.ptr>) {
  llvm.store %v, %p : vector<4x!llvm.ptr>, !llvm.ptr
  return
}

func.func @dstore_vector_pointer(%p: !llvm.ptr, %dp: !llvm.ptr, %v: vector<4x!llvm.ptr>) -> vector<4x!llvm.ptr> {
  %r = enzyme.autodiff @store_vector_pointer(%p, %dp, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (!llvm.ptr, !llvm.ptr, vector<4x!llvm.ptr>) -> vector<4x!llvm.ptr>
  return %r : vector<4x!llvm.ptr>
}

// -----

func.func @store_inactive_vector_pointer(%p: !llvm.ptr, %v: vector<4x!llvm.ptr>) {
  llvm.store %v, %p : vector<4x!llvm.ptr>, !llvm.ptr
  return
}

func.func @dstore_inactive_vector_pointer(%p: !llvm.ptr, %v: vector<4x!llvm.ptr>) {
  enzyme.autodiff @store_inactive_vector_pointer(%p, %v) <{ activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!llvm.ptr, vector<4x!llvm.ptr>) -> ()
  return
}

// -----

// Constant vector values remain valid when the destination is duplicated.
func.func @store_constant_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) {
  %c0 = arith.constant 0 : index
  memref.store %v, %m[%c0] : memref<1xvector<4xf8E4M3FN>>
  return
}

func.func @dstore_constant_vector_float8(%m: memref<1xvector<4xf8E4M3FN>>, %dm: memref<1xvector<4xf8E4M3FN>>, %v: vector<4xf8E4M3FN>) {
  enzyme.autodiff @store_constant_vector_float8(%m, %dm, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (memref<1xvector<4xf8E4M3FN>>, memref<1xvector<4xf8E4M3FN>>, vector<4xf8E4M3FN>) -> ()
  return
}

// -----

func.func @store_constant_vector_pointer(%p: !llvm.ptr, %v: vector<4x!llvm.ptr>) {
  llvm.store %v, %p : vector<4x!llvm.ptr>, !llvm.ptr
  return
}

func.func @dstore_constant_vector_pointer(%p: !llvm.ptr, %dp: !llvm.ptr, %v: vector<4x!llvm.ptr>) {
  enzyme.autodiff @store_constant_vector_pointer(%p, %dp, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!llvm.ptr, !llvm.ptr, vector<4x!llvm.ptr>) -> ()
  return
}
