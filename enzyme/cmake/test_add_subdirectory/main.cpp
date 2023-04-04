#include <iostream>

extern double __enzyme_autodiff(void*, double);

double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff((void*) square, x);
}

int main() {
  for(double i=1; i<5; i++) {
    std::cout << square(i) << " " << dsquare(i) << std::endl;
  }
}