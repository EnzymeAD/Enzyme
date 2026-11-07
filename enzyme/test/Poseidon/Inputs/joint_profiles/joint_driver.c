#include <stdio.h>
extern double site_a(double, double);
extern double site_b(double, double);
int main(void) {
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25;
    s += site_a(x, y) + site_b(x, y);
  }
  printf("%.17g\n", s);
  return 0;
}
