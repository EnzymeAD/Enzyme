#include <iostream>

#include "minimal_test_framework.hpp"

#include <enzyme/enzyme>

double square(double x) { return x * x; }

double dsquare(double x) { 
  return enzyme::autodiff<enzyme::Forward>((void*) square, enzyme::Duplicated{x, 1.0}); 
//  return 1.0;
}

int main() {
  for(double i=1; i<5; i++) {
    EXPECT(dsquare(i) == 2 * i);
  }
  return any_tests_failed;
}
