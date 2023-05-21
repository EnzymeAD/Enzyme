// RUN: %clang++ -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S -enzyme-loose-types | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S -enzyme-loose-types | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S -enzyme-loose-types | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S -enzyme-loose-types | %lli - 
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S -enzyme-loose-types | %lli - 
// TODO THIS IS BROKEN on 16+ ?
// RUN: if [ %llvmver -lt 16 ]; then %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -lt 16 ]; then %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - ; fi

#include <eigen3/Eigen/Dense>
#include "test_utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//constexpr size_t IN = 3, OUT = 7;
constexpr size_t IN = 2, OUT = 2;

__attribute__((noinline))
static void matvec(const MatrixXd* __restrict W, const VectorXd* __restrict b, VectorXd* __restrict output) {
  *output = *W * *b;
  /*
  for (int r = 0; r < W->rows(); r++) {
    (*output)(r) = 0;

    for (int c = 0; c < W->cols(); c++) {
      (*output)(r) += (*W)(r, c) * (*b)(c);
    }
  }
  */
}

extern "C" {
double __enzyme_autodiff(void*, void*, void*, void*, void*, void*, void*);
}

int main(int argc, char** argv) {

    MatrixXd W = Eigen::MatrixXd::Constant(IN, OUT, 3.0);
    VectorXd M = Eigen::VectorXd::Constant(OUT, 2.0);
    VectorXd O = Eigen::VectorXd::Constant(IN, 0.0);
    
    MatrixXd Wp = Eigen::MatrixXd::Constant(IN, OUT, 0.0);
    VectorXd Mp = Eigen::VectorXd::Constant(OUT, 0.0);
    VectorXd Op = Eigen::VectorXd::Constant(IN, 1.0);
    VectorXd Op_orig = Op;
    
    __enzyme_autodiff((void*)matvec, &W, &Wp, &M, &Mp, &O, &Op);
    
    for(int o=0; o<OUT; o++)
    for(int i=0; i<IN; i++) {
        fprintf(stderr, "W(o=%d, i=%d)=%f\n", i, o, W(i, o));
    }
     
    for(int o=0; o<OUT; o++) {
        fprintf(stderr, "M(o=%d)=%f\n", o, M(o));
    }
    
    for(int i=0; i<IN; i++) {
        fprintf(stderr, "O(i=%d)=%f\n", i, O(i));
    }

    for(int o=0; o<OUT; o++)
    for(int i=0; i<IN; i++) {
        APPROX_EQ( Wp(i, o), M(o) * Op_orig(i) , 1e-10);
        fprintf(stderr, "Wp(o=%d, i=%d)=%f\n", i, o, Wp(i, o));
    }
     
    for(int o=0; o<OUT; o++) {
        double res = 0.0;
        for(int i=0; i<IN; i++) res += W(i, o) * Op_orig(i); 
        APPROX_EQ( Mp(o), res, 1e-10);
        fprintf(stderr, "Mp(o=%d)=%f\n", o, Mp(o));
    }
    
    for(int i=0; i<IN; i++) {
        APPROX_EQ( Op(i), 0., 1e-10);
        fprintf(stderr, "Op(i=%d)=%f\n", i, Op(i));
    }
    //assert(0 && "false");
}
