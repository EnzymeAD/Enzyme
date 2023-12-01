// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s

// Test that a call to an external function produces the correct value activity.

// CHECK-LABEL: @d_tanh
// CHECK:         "tanh_1": Active
// CHECK:         "tanh_2": Active
func.func @d_tanh(%x: f64) -> f64 {
  %tanh = llvm.call @tanh(%x) {tag = "tanh_1", fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  %tanh2 = llvm.call @tanh(%tanh) {tag = "tanh_2", fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  return %tanh2 : f64
}

llvm.func local_unnamed_addr @tanh(f64 {llvm.noundef}) -> f64 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "nosync", "nounwind", "willreturn", ["approx-func-fp-math", "true"], ["frame-pointer", "non-leaf"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["target-features", "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"], ["unsafe-fp-math", "true"]], sym_visibility = "private"}
