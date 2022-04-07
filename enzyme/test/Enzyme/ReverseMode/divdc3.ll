; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define { double, double } @square(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double } %call
}

declare { double, double } @__divdc3(double, double, double, double)

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double, double, double)* @square to i8*), double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double } %call
}

declare { double, double } @__enzyme_autodiff(i8*, ...)