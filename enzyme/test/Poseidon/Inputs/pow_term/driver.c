#include <stdio.h>
#include <stdlib.h>
extern void site(double, double, double *, double *, double *, double *);
/* argv[1]: which output's shadow is seeded (0, 1; absent = both). */
int main(int argc, char **argv) {
  int only = argc > 1 ? atoi(argv[1]) : -1;
  double s = 0.0;
  for (int i = 0; i < 64; ++i) {
    double T = 250.0 + i * 150.0, y = 0.5 + (i % 7) * 0.25;
    double o1, o2, d1 = only != 1, d2 = only != 0;
    site(T, y, &o1, &d1, &o2, &d2);
    s += o1 + o2;
  }
  printf("%.17g\n", s);
  return 0;
}
