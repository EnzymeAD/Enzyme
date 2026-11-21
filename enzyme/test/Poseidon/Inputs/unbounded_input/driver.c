#include <math.h>
#include <stdio.h>
extern void site(double, double, double *, double *);
extern void site_nan(double, double, double *, double *);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 1.0 + i, y = 0.5 + (i % 7) * 0.25;
    double o, d = 1.0;
    site(i == 0 ? INFINITY : x, y, &o, &d);
    s += o;
    d = 1.0;
    site_nan(-x, y, &o, &d);
    s += o;
  }
  printf("%.17g\n", s);
  return 0;
}
