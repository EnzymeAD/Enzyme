#include <stdio.h>
extern void site_undef(double, double, double *, double *, double *,
                       double *);
extern void site_ce(double, double, double *, double *);
extern void site_powi(double, double, int, double *, double *);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 0.5 + i * 0.25, y = 0.5 + (i % 7) * 0.25;
    double o, d = 1.0, u, du = 0.0;
    site_undef(x, y, &o, &d, &u, &du);
    s += o;
    d = 1.0;
    site_ce(x, y, &o, &d);
    s += o;
    d = 1.0;
    site_powi(x, y, 1 + i % 3, &o, &d);
    s += o;
  }
  printf("%.17g\n", s);
  return 0;
}
