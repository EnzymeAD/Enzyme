func.func @activereturn(%x: memref<f64>) -> memref<f64> {
    %y = memref.alloca() : memref<memref<f64>>
    memref.store %x, %y[] : memref<memref<f64>>
    %u = memref.load %y[] : memref<memref<f64>>
    return %u : memref<f64>
}
