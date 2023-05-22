#include <stdio.h>

extern double __enzyme_autodiff(void*, double);

double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff((void*) square, x);
}

int main() {
  for(double i=1; i<5; i++) {
    printf("%f %f\n", square(i), dsquare(i));
  }
}
