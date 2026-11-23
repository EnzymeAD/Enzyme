#include <stdio.h>
extern void site(double, double, int, double *, double *, double *);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25;
    double o, d = 1.0;
    site(x, y, 0, 0, &o, &d);
    s += o;
  }
  printf("%.17g\n", s);
  return 0;
}
