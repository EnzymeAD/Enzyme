#include <stdio.h>
extern void site(double, double, double *, double *, double *, double *);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25;
    double o1, o2, d1 = 1.0, d2 = 0.0;
    site(x, y, &o1, &d1, &o2, &d2);
    s += o1 + o2;
  }
  printf("%.17g\n", s);
  return 0;
}
