#include <stdio.h>
extern void site_frem(double, double, double *, double *);
extern void site_fma(double, double, double *, double *);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 0.5 + i * 0.25, y = 0.5 + (i % 7) * 0.25;
    double o, d = 1.0;
    site_frem(x, y, &o, &d);
    s += o;
    d = 1.0;
    site_fma(x, y, &o, &d);
    s += o;
  }
  printf("%.17g\n", s);
  return 0;
}
