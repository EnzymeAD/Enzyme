#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#if 0

/*
__attribute__((noinline))
double times2(double x) {
    return 2 * sqrt(x);
}

double squarea(double x) {
    return x * times2(x);
}

double min(double x, double y) {
    return (x < y) ? x : y;
}

double relu(double x) {
    return (x > 0) ? x : 0;
}

double squa(double x) {
    return relu(min(x, x*x/2));
}

double squarez(double x) {
    if (x > 0) {
        //return cos(x * sin(x));
        return sqrt(x * sin(x));
    } else {
        return 0;
    }
}

__attribute__((noinline))
double ptr(double* __restrict x) {
    return (*x) * (*x);
}
*/

/*
#define LEN 3
__attribute__((noinline))
double sumsquare(double* __restrict x ) {
    double sum = 0;
    for(int i=0; i < LEN; i++) {
        //sum += x[i];
        sum += x[i] * x[i];
    }
    return sum;
}

double square(double x) {
   double ar[LEN] = {x};
   return sumsquare(ar);
}
*/


/*
__attribute__((noinline))
double times(double x, int y) {
    return x * y;
}

double square(double x) {
    return times(x, 2);
}
*/

/*
__attribute__((noinline))
double sumsquare(double* __restrict x, int n) {
    double sum = 0;
    #pragma clang loop vectorize(disable)
    #pragma clang loop unroll(disable)
    for(int i=0; i < n; i++) {
        //sum += x[i];
        //printf("running iteration %d\n", i);
        sum += x[i] * x[i];
    }
    //printf("returning sum\n");
    return sum;
}

double square(double x) {
   int n = 6;
   double ar[6] = {1, x, x*x, x*x*x, x*x*x*x, x*x*x*x*x};//, x*x*x*x*x*x};
   return sumsquare(ar, n);
}
*/

/*
__attribute__((noinline))
double foo(double* __restrict matrix) {//, double* __restrict vector) {
  printf("begin foo\n");
  double output = 0;//{0};

  for (int idx = 0; idx < 10*10; idx++) {
    //printf("foo idx=%d\n", idx);
    int i = idx%10;
    int j = idx/10;
    output += matrix[j*10 + i]//vector[i];//matrix[j*100+i] + vector[i];
    ;
  }
  //printf("ended foo\n");
  return output;
}

double square(double x) {
  printf("starting square\n");
  double vector[10] = {0};
    //#pragma clang loop vectorize(disable)
    //#pragma clang loop unroll(disable)
  for (int i = 0; i < 10; i++) {
    vector[i] = (1.0*i)/10;
    //printf("vector[%d]\n", i);
  }
  //printf("set vector\n");
  double matrix_weights[10*10] = {0};

    //#pragma clang loop vectorize(disable)
    //#pragma clang loop unroll(disable)
  for (int idx = 0; idx < 10*10; idx++) {
    int i = idx%10;
    int j = idx/10;
    //printf("matrix[%d]\n", idx);
    matrix_weights[j*10+i] = 1.0*(j+i) + 1e-20;
  }

  printf("calling foo\n");
  return foo(matrix_weights);//, vector);
}
*/

/*
__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};

  #pragma clang loop unroll(disable)
  for (int idx = 0; idx < len*len; idx++) {
    //printf("foo idx=%d\n", idx);
    int i = idx%len;
    int j = idx/len;
    output += matrix[j*len + i] + vector[i];
    ;
  }
  //printf("ended foo\n");
  return output;
}
*/

__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};

  #pragma clang loop unroll(disable)
  for (int i = 0; i < len; i++) {
    //printf("foo idx=%d\n", idx);
  #pragma clang loop unroll(disable)
  for (int j = 0; j < len; j++) {
    printf("foo idx=(i=%d,j=%d)\n", i, j);
    output += matrix[j*len + i] + vector[i];
    ;
  }
  }
  //printf("ended foo\n");
  return output;
}

double square(double x) {
  #define len 100
  double vector[len] = {0};
  for (int i = 0; i < len; i++) {
    vector[i] = (1.0*i)/len
     //* x
     ;
  }
  double matrix_weights[len*len] = {0};

  for (int idx = 0; idx < len*len; idx++) {
    int i = idx%len;
    int j = idx/len;
    matrix_weights[j*len+i] =
    //x *
    1.0*(j+i) + 1e-20;
  }

  printf("calling foo\n");
  return foo(matrix_weights, vector, len);
}

#endif

__attribute__((noinline))
double foo(double* __restrict matrix, double* __restrict vector, size_t len) {
  double output = 0;//{0};

  #pragma clang loop unroll(disable)
  for (int i = 0; i < len*len; i++) {
    //printf("foo(%i) precond\n", i);
    if (vector[i % len] > 0) {
      output += matrix[i];
      //printf("  foo(%i) incond\n", i);
    }
    //printf("foo(%i) endcond\n", i);
    //else {
    //  output += matrix[i] * matrix[i];
    //}
  }
  //printf("ended foo\n");
  return output;
}

double square(double x) {
  #define len 10
  double vector[len] = {0};
  for (int i = 0; i < len; i++) {
    vector[i] = (1.0*i)/len
     - x
     ;
  }
  double matrix_weights[len*len] = {0};

  for (int idx = 0; idx < len*len; idx++) {
    int i = idx%len;
    int j = idx/len;
    matrix_weights[j*len+i] =
    x *
    1.0*(j-i) + 1e-20;
  }

  printf("calling foo\n");
  return foo(matrix_weights, vector, len);
}

int main(int argc, char** argv) {
    double f = atof(argv[1]);

    printf("now executing square\n");
    double res0 = square(f);
    printf("finished executing square\n");
    printf("f(x=%lf) = %lf\n", f, res0);
    printf("now executing builtin autodiff\n");
    double res = __builtin_autodiff(square, f);
    printf("finished executing autodiff\n");
    printf("d/dx f(x=%lf) = %lf\n", f, res);
    //printf("d/dx sqrt(x) | x=%lf  = %lf | eval=%lf\n", f, __builtin_autodiff(ptr, f), ptr(f));
}

#if 0

double f(double x) {
  for(int i=1; i<5; i++) {
    x = sin(cos(x));
  }
  return x;
}

__attribute__((noinline))
double loop(double x, int n) {
  double r = x/x;

  #pragma clang loop unroll(disable)
  for(int i=1; i<n; i++) {
    r *= f(x);
  }
  return sin(cos(r));
}

double test(double x) {
  return loop(x, 3);
}


double max(double x, double y) {
    return (x > y) ? x : y;
}

__attribute__((noinline))
double logsumexp(double *x, int n) {
  double A = x[0];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++) {
    A = max(A, x[i]);
  }
  double ema[n];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++) {
    ema[i] = exp(x[i] - A);
  }
  double sema = 0;
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<n; i++)
    sema += ema[i];
  return log(sema) + A;
}


double test2(double x) {
  double rands[100000];
  #pragma clang loop unroll(disable)
  #pragma clang loop vectorize(disable)
  for(int i=0; i<100000; i++) {
    rands[i] = i * x;
  }
  return logsumexp(rands, 100000);
}

int main0(int argc, char** argv) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = test(2);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(test, 2);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}

int main2(int argc, char** argv) {

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = test2(2);

  gettimeofday(&end, NULL);
  printf("%0.6f res=%f\n", tdiff(&start, &end), res);
  }

  {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  double res = __builtin_autodiff(test2, 2);

  gettimeofday(&end, NULL);
  printf("%0.6f res'=%f\n", tdiff(&start, &end), res);
  }
}

#endif
