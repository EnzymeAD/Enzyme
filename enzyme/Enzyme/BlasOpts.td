
class BlasOptPattern< list<dag> _inputs, list<string> _tmps, list<dag> _outputs> {
  list<dag> inputs = _inputs;
  // tmp variables will dissapear during the transformation 
  // and therefore are not allowed to be read elsewhere
  list<string> tmps = _tmps;
  list<dag> outputs = _outputs;
}

class Value<int _i> {
  int i = _i;
}

def first : BlasOptPattern<
                  [
                  (b<"ger"> $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $A, $lda),
                  (b<"ger"> $layout, $n, $k, $beta, $v, $incv, $w, $incw, $B, $ldb),
                  (b<"gemm"> $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ],
                  ["A", "B"],
                  [
                  (Value<1> (b<"dot"> $layout, $n, $v, $incv, $y, $incy)),
                  (Value<2> (FMul $alpha, Value<1>)),
                  (Value<3> (FMul $beta, Value<2>)),
                  (b<"ger"> $layout, $m, $k, Value<3>, $x, $incx, $w, $incw, $C, $ldc),
                  ]
                  >;
