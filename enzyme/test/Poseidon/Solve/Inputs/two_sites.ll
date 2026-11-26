define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @tester_b(double %x, double %y) #0 {
entry:
  %mul = fmul fast double %x, %y
  %sub = fsub fast double %mul, %y
  %div = fdiv fast double %sub, %x
  ret double %div
}

define double @site_a(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y)
  ret double %0
}

define double @site_b(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_b, double %x, double %y)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)
attributes #0 = { "target-cpu"="x86-64" }
