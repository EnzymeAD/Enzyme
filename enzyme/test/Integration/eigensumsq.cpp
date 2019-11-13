// RUN: clang++ -fno-exceptions -O3 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O2 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O0 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O3 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O2 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O1 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | lli - 
// RUN: clang++ -fno-exceptions -O0 %s -S -emit-llvm -o - | opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | lli - 

#define EIGEN_DONT_ALIGN 1
#define EIGEN_NO_DEBUG 1
#define EIGEN_UNROLLING_LIMIT 0
#define EIGEN_DONT_VECTORIZE 1

#include "test_utils.h"
#include <eigen3/Eigen/Dense>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
    extern double __enzyme_autodiff(void*, ...);
}

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;

constexpr size_t IN = 4, OUT = 4, NUM = 5;
__attribute__((noinline))
static double matvec(const Matrix<double, IN, OUT>* __restrict W, const Matrix<double, IN, OUT>* __restrict M) {
  MatrixXd diff = *W-*M;
  return (diff*diff).sum();
}

int main(int argc, char** argv) {

    //size_t IN = 40, OUT = 30, NUM = 50;

    Matrix<double, IN, OUT>  W(IN, OUT);
    Matrix<double, IN, OUT> Wp(IN, OUT);
    
    Matrix<double, IN, OUT> M(IN, OUT);
    Matrix<double, IN, OUT> Mp(IN, OUT);

    W = Eigen::Matrix<double, IN, OUT>::Constant(IN, OUT, 1.0);
    M = Eigen::Matrix<double, IN, OUT>::Constant(IN, OUT, 2.0);
    
    Wp = Eigen::Matrix<double, IN, OUT>::Constant(IN, OUT, 0.0);
    Mp = Eigen::Matrix<double, IN, OUT>::Constant(IN, OUT, 0.0);
    
    __enzyme_autodiff((void*)matvec, &W, &Wp, &M, &Mp);
    

    for(int o=0; o<OUT; o++)
    for(int i=0; i<IN; i++) {
        APPROX_EQ( Wp(i, o), -8, 1e-10);
        fprintf(stderr, "Wp(o=%d, i=%d)=%f\n", i, o, Wp(i, o));
    }
     
    for(int o=0; o<OUT; o++)
    for(int i=0; i<IN; i++) {
        APPROX_EQ( Mp(i, o), 8, 1e-10);
        fprintf(stderr, "Mp(o=%d, i=%d)=%f\n", i, o, Mp(i, o));
    }
}
