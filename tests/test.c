#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
TODO FIX BELOW
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

__attribute__((noinline)) 
double foo(double* __restrict matrix) {//, double* __restrict vector) {
  printf("begin foo\n");
  double output = 0;//{0};

    #pragma clang loop vectorize(disable)
    #pragma clang loop unroll(disable)
  for (int idx = 0; idx < 10*10; idx++) {
    printf("foo idx=%d\n", idx);
    int i = idx%10;
    int j = idx/10;
    output += matrix[j*10 + i]//vector[i];//matrix[j*100+i] + vector[i];
    ;
  }
  printf("ended foo\n");
  return output;
}

double square(double x) {
  printf("starting square\n");
  double vector[10] = {0};
    #pragma clang loop vectorize(disable)
    #pragma clang loop unroll(disable)
  for (int i = 0; i < 10; i++) {
    vector[i] = (1.0*i)/10;
  }
  printf("set vector\n");
  double matrix_weights[10*10] = {0};

  /*
    #pragma clang loop vectorize(disable)
    #pragma clang loop unroll(disable)
  for (int idx = 0; idx < 100*100; idx++) {
    int i = idx%100;
    int j = idx/100;
    matrix_weights[j*100+i] = 1.0*(j+i) + 1e-20;
  }
    */
  printf("calling foo\n");
  return foo(matrix_weights);//, vector);
}

int main(int argc, char** argv) {
    double f = atof(argv[1]);
    printf("f(x=%lf) = %lf\n", f, square(f));
    printf("now executing builtin autodiff\n");
    double res = __builtin_autodiff(square, f); 
    printf("finished executing autodiff\n");
    printf("d/dx f(x=%lf) = %lf\n", f, res);
    //printf("d/dx sqrt(x) | x=%lf  = %lf | eval=%lf\n", f, __builtin_autodiff(ptr, f), ptr(f));
}
