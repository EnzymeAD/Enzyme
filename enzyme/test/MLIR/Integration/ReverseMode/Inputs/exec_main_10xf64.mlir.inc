memref.global "private" @__buf : memref<10xf64> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]>
memref.global "private" @__dbuf : memref<10xf64> = dense<0.0>

func.func @main() {
  %buf = memref.get_global @__buf : memref<10xf64>
  %dbuf = memref.get_global @__dbuf : memref<10xf64>
  %seed = arith.constant 1.0 : f64
  call @reduce_sum(%buf, %dbuf, %seed) : (memref<10xf64>, memref<10xf64>, f64) -> ()
  %ubuf = memref.cast %buf : memref<10xf64> to memref<*xf64>
  %udbuf = memref.cast %dbuf : memref<10xf64> to memref<*xf64>
  call @printMemrefF64(%ubuf) : (memref<*xf64>) -> ()
  call @printMemrefF64(%udbuf) : (memref<*xf64>) -> ()
  return
}

func.func private @printMemrefF64(%ptr : memref<*xf64>)
