#include <adept.h>
#include <adept_source.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
using adept::adouble;

template <typename Return, typename... T> Return __enzyme_autodiff(T...);

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

#include "fft.h"

void foobar(double *data, size_t len) {
  fft(data, len);
  ifft(data, len);
}

void afoobar(aVector &data, size_t len) {
  fft(data, len);
  ifft(data, len);
}

extern "C" {
int enzyme_dupnoneed;
}

extern "C" void rust_unsafe_dfoobar(size_t n, double *data, double *ddata);
extern "C" void rust_unsafe_foobar(size_t n, double *data);
extern "C" void rust_dfoobar(size_t n, double *data, double *ddata);
extern "C" void rust_foobar(size_t n, double *data);

static double rust_unsafe_foobar_and_gradient(size_t len) {
  double *inp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    inp[i] = 2.0;
  double *dinp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    dinp[i] = 1.0;
  rust_unsafe_dfoobar(len, inp, dinp);
  double res = dinp[0];
  delete[] dinp;
  delete[] inp;
  return res;
}

static double rust_foobar_and_gradient(size_t len) {
  double *inp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    inp[i] = 2.0;
  double *dinp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    dinp[i] = 1.0;
  rust_dfoobar(len, inp, dinp);
  double res = dinp[0];
  delete[] dinp;
  delete[] inp;
  return res;
}

__attribute__((noinline)) static double foobar_and_gradient(size_t len) {
  double *inp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    inp[i] = 2.0;
  double *dinp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    dinp[i] = 1.0;
  __enzyme_autodiff<void>(foobar, enzyme_dupnoneed, inp, dinp, len);
  double res = dinp[0];
  delete[] dinp;
  delete[] inp;
  return res;
}

static double afoobar_and_gradient(size_t len) {
  adept::Stack stack;

  aVector x(2 * len);
  for (size_t i = 0; i < 2 * len; i++)
    x(i) = 2.0;
  stack.new_recording();
  afoobar(x, len);
  for (size_t i = 0; i < 2 * len; i++)
    x(i).set_gradient(1.0);
  stack.compute_adjoint();

  double *dinp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    dinp[i] = x(i).get_gradient();
  double res = dinp[0];
  delete[] dinp;
  return res;
}

static double tfoobar_and_gradient(size_t len) {
  double *inp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    inp[i] = 2.0;
  double *dinp = new double[2 * len];
  for (size_t i = 0; i < 2 * len; i++)
    dinp[i] = 1.0;
  foobar_b(inp, dinp, len);
  double res = dinp[0];
  delete[] dinp;
  delete[] inp;
  return res;
}

static void adept_sincos(double inp, size_t len) {
  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    foobar(x, len);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Adept real %0.6f res=%f\n", tdiff(&start, &end), res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    adept::Stack stack;

    aVector x(2 * len);
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    // stack.new_recording();
    afoobar(x, len);
    double res = x(0).value();

    gettimeofday(&end, NULL);
    printf("Adept forward %0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res2 = afoobar_and_gradient(len);

    gettimeofday(&end, NULL);
    printf("Adept combined %0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}

static void tapenade_sincos(double inp, size_t len) {

  // {
  // struct timeval start, end;
  // gettimeofday(&start, NULL);

  // double *x = new double[2*len];
  // for(size_t i=0; i<2*len; i++) x[i] = 2.0;
  // foobar(x, len);
  // double res = x[0];

  // gettimeofday(&end, NULL);
  // printf("Tapenade real %0.6f res=%f\n", tdiff(&start, &end), res);
  // delete[] x;
  // }

  // {
  // struct timeval start, end;
  // gettimeofday(&start, NULL);

  // double* x = new double[2*len];
  // for(size_t i=0; i<2*len; i++) x[i] = 2.0;
  // foobar(x, len);
  // double res = x[0];

  // gettimeofday(&end, NULL);
  // printf("Tapenade forward %0.6f res=%f\n", tdiff(&start, &end), res);
  // delete[] x;
  // }

  // {
  // struct timeval start, end;
  // gettimeofday(&start, NULL);

  // double res2 = tfoobar_and_gradient(len);

  // gettimeofday(&end, NULL);
  // printf("Tapenade combined %0.6f res'=%f\n", tdiff(&start, &end), res2);
  // }
}

static void enzyme_sincos(double inp, size_t len) {

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    foobar(x, len);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme real %0.6f res=%f\n", tdiff(&start, &end), res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    foobar(x, len);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme forward %0.6f res=%f\n", tdiff(&start, &end), res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res2 = foobar_and_gradient(len);

    gettimeofday(&end, NULL);
    printf("Enzyme combined %0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}

static void enzyme_unsafe_rust_sincos(double inp, size_t len) {

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    rust_unsafe_foobar(len, x);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme (unsafe Rust) real %0.6f res=%f\n", tdiff(&start, &end),
           res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    rust_unsafe_foobar(len, x);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme (unsafe Rust) forward %0.6f res=%f\n", tdiff(&start, &end),
           res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res2 = rust_unsafe_foobar_and_gradient(len);

    gettimeofday(&end, NULL);
    printf("Enzyme (unsafe Rust) combined %0.6f res'=%f\n", tdiff(&start, &end),
           res2);
  }
}

static void enzyme_rust_sincos(double inp, size_t len) {

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    rust_foobar(len, x);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme (Rust) real %0.6f res=%f\n", tdiff(&start, &end), res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double *x = new double[2 * len];
    for (size_t i = 0; i < 2 * len; i++)
      x[i] = 2.0;
    rust_foobar(len, x);
    double res = x[0];

    gettimeofday(&end, NULL);
    printf("Enzyme (Rust) forward %0.6f res=%f\n", tdiff(&start, &end), res);
    delete[] x;
  }

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double res2 = rust_foobar_and_gradient(len);

    gettimeofday(&end, NULL);
    printf("Enzyme (Rust) combined %0.6f res'=%f\n", tdiff(&start, &end), res2);
  }
}

/* Function to check if x is power of 2*/
bool isPowerOfTwo(size_t x) {
  /* First x in the below expression is for the case when x is 0 */
  return x && (!(x & (x - 1)));
}

size_t max(size_t A, size_t B) {
  if (A > B)
    return A;
  return B;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    printf("usage %s n [must be power of 2]\n", argv[0]);
    return 1;
  }
  size_t N = atol(argv[1]);
  if (!isPowerOfTwo(N)) {
    printf("usage %s n [must be power of 2]\n", argv[0]);
    return 1;
  }
  double inp = -2.1;

  for (size_t iters = max(1, N >> 0); iters <= N; iters *= 2) {
    printf("iters=%zu\n", iters);
    // adept_sincos(inp, iters);
    tapenade_sincos(inp, iters);
    enzyme_sincos(inp, iters);
    enzyme_rust_sincos(inp, iters);
    enzyme_unsafe_rust_sincos(inp, iters);
  }
}
