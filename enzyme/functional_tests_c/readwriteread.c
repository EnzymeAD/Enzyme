#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);
int counter = 0;
double recurse_max_helper(float* a, float* b, int N) {
  if (N <= 0) {
    return *a + *b;
  }
  return recurse_max_helper(a,b,N-1) + recurse_max_helper(a,b,N-2);
}


double f_read(double* x) {
  double product = (*x) * (*x);
  return product;
}


void g_write(double* x, double product) {
  *x = (*x) * product;
}

double h_read(double* x) {
  return *x;
}


double readwriteread_helper(double* x) {
  double product = f_read(x);
  g_write(x, product);
  double ret = h_read(x);
  return ret; 
}

void readwriteread(double*__restrict x, double*__restrict ret) {
  *ret = readwriteread_helper(x);
  //*ret = (*x) * (*x) * (*x);
}



int main(int argc, char** argv) {



  double ret = 0;
  double dret = 1.0;
  double* x = (double*) malloc(sizeof(double));
  double* dx = (double*) malloc(sizeof(double));
  *x = 2.0;
  *dx = 0.0;

  __builtin_autodiff(readwriteread, x, dx, &ret, &dret);

  
  printf("dx is %f ret is %f\n", *dx, ret);
  assert(*dx == 3*2.0*2.0);
  //assert(db == 17711.0*2);



  //printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, ret, da,db);
  return 0;
}
