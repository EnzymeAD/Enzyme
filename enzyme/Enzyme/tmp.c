#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <string.h>

// Get the ULP of a double by looking at the difference
// when the last bit of the mantissa is changed.
double ulp(double res) {
  double nres = res;
  (*(uint64_t *)&nres) = 0x1 ^ *(uint64_t *)&nres;
  return std::abs(nres - res);
}

// Struct for forward, gradient return
struct pair {
  double prim;
  double grad;
};

template <typename... Args> double __enzyme_fwddiff(Args...);

// Automatically register and create the condition
// propagation as a custom derivative, using fwddiff
// to get the actual derivative of the function for use
// in the computation.

#define CONCAT(X, Y) X##Y
#define UNARY(x)                                                               \
  __attribute__((noinline)) double CONCAT(x, _)(double v) { return x(v); }     \
  double CONCAT(x, _wrap)(double v) { return x(v); }                           \
  __attribute__((always_inline)) struct pair CONCAT(x, _err)(double v,         \
                                                             double err) {     \
    double res = x(v);                                                         \
    return {res, std::max(ulp(res),                                            \
                          std::abs(v *                                         \
                                   __enzyme_fwddiff((void *)CONCAT(x, _wrap),  \
                                                    v, err) /                  \
                                   res))};                                     \
  }                                                                            \
  void *CONCAT(__enzyme_register_derivative_,                                  \
               x)[2] = {(void *)CONCAT(x, _), (void *)CONCAT(x, _err)};

#define BINARY(x)                                                              \
  __attribute__((noinline)) double CONCAT(x, _)(double v1, double v2) {        \
    return x(v1, v2);                                                          \
  }                                                                            \
  double CONCAT(x, _wrap)(double v1, double v2) { return x(v1, v2); }          \
  __attribute__((always_inline)) struct pair CONCAT(x, _err)(                  \
      double v1, double v1err, double v2, double v2err) {                      \
    double res = x(v1, v2);                                                    \
    return (struct pair){                                                      \
        res, std::max(ulp(res),                                                \
                      std::abs(v1 *                                            \
                               __enzyme_fwddiff((void *)CONCAT(x, _wrap), v1,  \
                                                v1err, v2, 0.0) /              \
                               res) +                                          \
                          std::abs(v2 *                                        \
                                   __enzyme_fwddiff((void *)CONCAT(x, _wrap),  \
                                                    v1, 0.0, v2, v2err) /      \
                                   res))};                                     \
  }                                                                            \
  void *CONCAT(__enzyme_register_derivative_,                                  \
               x)[2] = {(void *)CONCAT(x, _), (void *)CONCAT(x, _err)};

double add(double x, double y) { return x + y; }
double sub(double x, double y) { return x - y; }
double mul(double x, double y) { return x * y; }
double div(double x, double y) { return x / y; }

// Register condition propagation for following functions
UNARY(sin)
UNARY(sqrt)
UNARY(cos)
UNARY(exp)
BINARY(add)
BINARY(sub)
BINARY(mul)
BINARY(div)
BINARY(pow)

// The 4 stages of figure 1:
double s1(double x) { return cos_(x); }
double s2(double x) { return sub_(1, cos_(x)); }
double s3(double x) { return mul_(x, x); }
double s4(double x) { return div_(sub_(1, cos_(x)), mul_(x, x)); }

int main() {
  // 1.110223e-16
  printf("%e\n", __enzyme_fwddiff(s1, 1e-7, 0.0));
  // 2.222222e-02
  printf("%e\n", __enzyme_fwddiff(s2, 1e-7, 0.0));
  // 1.577722e-30
  printf("%e\n", __enzyme_fwddiff(s3, 1e-7, 0.0));
  // 2.222222e-02
  printf("%e\n", __enzyme_fwddiff(s4, 1e-7, 0.0));
}