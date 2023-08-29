// RUN: if [ %llvmver -eq 15 ]; then %clang -std=c11 -O0 -fno-unroll-loops -fno-tree-vectorize %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi


void f(double x[2])
{
   double t;
A:
   t    = 4*__builtin_cos(x[0] + x[1]);
   x[0] = 3*__builtin_sin(2*x[0] + x[1]);
   x[1] = t;
   if (x[0]*x[0] < x[1])
      return;
   else if (x[0] > 0)
      goto A;
   else
      goto B;

B:
   t    = 4*__builtin_sin(2*x[0] + x[1]);
   x[0] = 4*__builtin_cos(x[0] + 2*x[1]);
   x[1] = t;
   if (x[0]*x[0] < x[1])
      return;
   else if (x[0] > 0)
      goto A;
   else
      goto B;
}

extern void* __enzyme_augmentfwd(void*, double*, double*);
extern void __enzyme_reverse(void*, double*, double*, void*);

int main()
{
   double x[2], x_b[2];
   x[0] = 0.9;
   x[1] = 0.7;
   x_b[0] = 1;
   x_b[1] = 2;

   void* tape = __enzyme_augmentfwd((void*)f, x, x_b);
   __enzyme_reverse((void*)f, x, x_b, tape);
}
