// RUN: %eopt %s -test-print-base-object -o /dev/null | FileCheck %s --check-prefix=DEFAULT
// RUN: %eopt %s -test-print-base-object='offset-allowed=false' -o /dev/null | FileCheck %s --check-prefix=NOOFFSET

// Each line shows a returned value followed by its computed base object.
// The default mode finds the allocation. The second mode preserves offsets.

// Casts preserve the address. Both modes reach the original pointer.
// DEFAULT: @pointer_casts return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @pointer_casts return 0: %{{[^ ]+}} -> %arg0
func.func @pointer_casts(%base: !llvm.ptr) -> !llvm.ptr<1> {
  %cast = llvm.bitcast %base : !llvm.ptr to !llvm.ptr
  %space = llvm.addrspacecast %cast : !llvm.ptr to !llvm.ptr<1>
  return %space : !llvm.ptr<1>
}

// A zero GEP does not move the pointer.
// DEFAULT: @zero_gep return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @zero_gep return 0: %{{[^ ]+}} -> %arg0
func.func @zero_gep(%base: !llvm.ptr) -> !llvm.ptr {
  %same = llvm.getelementptr %base[0] : (!llvm.ptr) -> !llvm.ptr, f32
  return %same : !llvm.ptr
}

// A nonzero GEP selects another element. Preserve that GEP in the second mode.
// DEFAULT: @nonzero_gep return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @nonzero_gep return 0: %[[SHIFTED:[^ ]+]] -> %[[SHIFTED]]
func.func @nonzero_gep(%base: !llvm.ptr) -> !llvm.ptr {
  %shifted = llvm.getelementptr %base[1] : (!llvm.ptr) -> !llvm.ptr, f32
  return %shifted : !llvm.ptr
}

// An unknown index can move the pointer. Do not assume that it is zero.
// DEFAULT: @dynamic_gep return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @dynamic_gep return 0: %[[DYNAMIC:[^ ]+]] -> %[[DYNAMIC]]
func.func @dynamic_gep(%base: !llvm.ptr, %index: i64) -> !llvm.ptr {
  %shifted = llvm.getelementptr %base[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  return %shifted : !llvm.ptr
}

// A dynamic GEP operand can still be a known zero constant.
// DEFAULT: @constant_zero_gep return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @constant_zero_gep return 0: %{{[^ ]+}} -> %arg0
func.func @constant_zero_gep(%base: !llvm.ptr) -> !llvm.ptr {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %same = llvm.getelementptr %base[%zero] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  return %same : !llvm.ptr
}

// Every index must be zero, including indices into an aggregate.
// DEFAULT: @multi_index_gep return 0: %{{[^ ]+}} -> %arg0
// DEFAULT-NEXT: @multi_index_gep return 1: %{{[^ ]+}} -> %arg0
// NOOFFSET: @multi_index_gep return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET-NEXT: @multi_index_gep return 1: %[[SHIFTED:[^ ]+]] -> %[[SHIFTED]]
func.func @multi_index_gep(%base: !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
  %same = llvm.getelementptr %base[0, 0] :
      (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x f32>
  %shifted = llvm.getelementptr %base[0, 1] :
      (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x f32>
  return %same, %shifted : !llvm.ptr, !llvm.ptr
}

// Follow the zero GEP, then stop at the nonzero GEP beneath it.
// DEFAULT: @gep_chain return 0: %{{[^ ]+}} -> %arg0
// DEFAULT-NEXT: @gep_chain return 1: %{{[^ ]+}} -> %arg0
// NOOFFSET: @gep_chain return 0: %[[OFFSET:[^ ]+]] -> %[[OFFSET]]
// NOOFFSET-NEXT: @gep_chain return 1: %{{[^ ]+}} -> %[[OFFSET]]
func.func @gep_chain(%base: !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
  %shifted = llvm.getelementptr %base[4] : (!llvm.ptr) -> !llvm.ptr, f32
  %same = llvm.getelementptr %shifted[0] : (!llvm.ptr) -> !llvm.ptr, f32
  return %shifted, %same : !llvm.ptr, !llvm.ptr
}

// Memref casts change type information but preserve the buffer address.
// DEFAULT: @memref_casts return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @memref_casts return 0: %{{[^ ]+}} -> %arg0
func.func @memref_casts(%base: memref<8xf32>) -> memref<?xf32, 1> {
  %dynamic = memref.cast %base : memref<8xf32> to memref<?xf32>
  %space = memref.memory_space_cast %dynamic : memref<?xf32> to memref<?xf32, 1>
  return %space : memref<?xf32, 1>
}

// A zero-offset subview starts at the source address, even with a larger stride.
// This option preserves the address. It does not preserve the entire layout.
// DEFAULT: @zero_offset_subview return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @zero_offset_subview return 0: %{{[^ ]+}} -> %arg0
func.func @zero_offset_subview(%base: memref<8xf32>) -> memref<4xf32, strided<[2]>> {
  %view = memref.subview %base[0] [4] [2] :
      memref<8xf32> to memref<4xf32, strided<[2]>>
  return %view : memref<4xf32, strided<[2]>>
}

// A nonzero subview offset changes the address.
// DEFAULT: @nonzero_subview return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @nonzero_subview return 0: %[[VIEW:[^ ]+]] -> %[[VIEW]]
func.func @nonzero_subview(%base: memref<8xf32>) -> memref<4xf32, strided<[1], offset: 1>> {
  %view = memref.subview %base[1] [4] [1] :
      memref<8xf32> to memref<4xf32, strided<[1], offset: 1>>
  return %view : memref<4xf32, strided<[1], offset: 1>>
}

// Stop when the subview offset is unknown.
// DEFAULT: @dynamic_subview return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @dynamic_subview return 0: %[[VIEW:[^ ]+]] -> %[[VIEW]]
func.func @dynamic_subview(%base: memref<8xf32>, %offset: index) -> memref<4xf32, strided<[1], offset: ?>> {
  %view = memref.subview %base[%offset] [4] [1] :
      memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
  return %view : memref<4xf32, strided<[1], offset: ?>>
}

// A byte view can change element type without changing the address.
// DEFAULT: @zero_byte_shift return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @zero_byte_shift return 0: %{{[^ ]+}} -> %arg0
func.func @zero_byte_shift(%base: memref<32xi8>) -> memref<4xf32> {
  %zero = arith.constant 0 : index
  %view = memref.view %base[%zero][] : memref<32xi8> to memref<4xf32>
  return %view : memref<4xf32>
}

// A nonzero byte shift must remain in the base object.
// DEFAULT: @nonzero_byte_shift return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @nonzero_byte_shift return 0: %[[VIEW:[^ ]+]] -> %[[VIEW]]
func.func @nonzero_byte_shift(%base: memref<32xi8>) -> memref<4xf32> {
  %four = arith.constant 4 : index
  %view = memref.view %base[%four][] : memref<32xi8> to memref<4xf32>
  return %view : memref<4xf32>
}

// An unknown byte shift also stops the address-preserving walk.
// DEFAULT: @dynamic_byte_shift return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @dynamic_byte_shift return 0: %[[VIEW:[^ ]+]] -> %[[VIEW]]
func.func @dynamic_byte_shift(%base: memref<32xi8>, %shift: index) -> memref<4xf32> {
  %view = memref.view %base[%shift][] : memref<32xi8> to memref<4xf32>
  return %view : memref<4xf32>
}

// Reinterpret offsets are absolute. Resetting offset one to zero changes the
// address, although the reinterpret_cast itself has a zero offset.
// DEFAULT: @reset_offset return 0: %{{[^ ]+}} -> %arg0
// DEFAULT-NEXT: @reset_offset return 1: %{{[^ ]+}} -> %arg0
// NOOFFSET: @reset_offset return 0: %[[SHIFTED:[^ ]+]] -> %[[SHIFTED]]
// NOOFFSET-NEXT: @reset_offset return 1: %[[RESET:[^ ]+]] -> %[[RESET]]
func.func @reset_offset(%base: memref<8xf32>) ->
    (memref<4xf32, strided<[1], offset: 1>>, memref<4xf32>) {
  %shifted = memref.subview %base[1] [4] [1] :
      memref<8xf32> to memref<4xf32, strided<[1], offset: 1>>
  %reset = memref.reinterpret_cast %shifted to offset: [0], sizes: [4], strides: [1] :
      memref<4xf32, strided<[1], offset: 1>> to memref<4xf32>
  return %shifted, %reset : memref<4xf32, strided<[1], offset: 1>>, memref<4xf32>
}

// ViewLikeOpInterface alone does not prove equal addresses. Extracting the
// underlying buffer removes the source offset, so preserve this operation.
// DEFAULT: @extract_base_buffer return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET: @extract_base_buffer return 0: %[[EXTRACTED:[^ ]+]] -> %[[EXTRACTED]]
func.func @extract_base_buffer(%source: memref<4xf32, strided<[1], offset: 1>>) -> memref<f32> {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %source :
      memref<4xf32, strided<[1], offset: 1>> -> memref<f32>, index, index, index
  return %base : memref<f32>
}

// Reshapes and transposes change metadata without moving the buffer address.
// DEFAULT: @shape_changes return 0: %{{[^ ]+}} -> %arg0
// DEFAULT-NEXT: @shape_changes return 1: %{{[^ ]+}} -> %arg0
// DEFAULT-NEXT: @shape_changes return 2: %{{[^ ]+}} -> %arg0
// NOOFFSET: @shape_changes return 0: %{{[^ ]+}} -> %arg0
// NOOFFSET-NEXT: @shape_changes return 1: %{{[^ ]+}} -> %arg0
// NOOFFSET-NEXT: @shape_changes return 2: %{{[^ ]+}} -> %arg0
func.func @shape_changes(%base: memref<8xf32>, %shape: memref<2xindex>) ->
    (memref<8xf32>, memref<4x2xf32, strided<[1, 4]>>, memref<2x4xf32>) {
  %expanded = memref.expand_shape %base [[0, 1]] output_shape [2, 4] :
      memref<8xf32> into memref<2x4xf32>
  %collapsed = memref.collapse_shape %expanded [[0, 1]] :
      memref<2x4xf32> into memref<8xf32>
  %transposed = memref.transpose %expanded (d0, d1) -> (d1, d0) :
      memref<2x4xf32> to memref<4x2xf32, strided<[1, 4]>>
  %reshaped = memref.reshape %base(%shape) :
      (memref<8xf32>, memref<2xindex>) -> memref<2x4xf32>
  return %collapsed, %transposed, %reshaped :
      memref<8xf32>, memref<4x2xf32, strided<[1, 4]>>, memref<2x4xf32>
}
