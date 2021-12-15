// RUN: if [ %llvmver -ge 13 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | FileCheck - ; fi
int enzyme_dup;

double __enzyme_autodiff(void*, double, double*, double*);

void stress_calculation(double du_dx, double* sigma) {
  *sigma = du_dx;
}

void func1(double* sigma, double* dir) {
        __enzyme_autodiff((void*)stress_calculation,
            2.0,
            sigma, dir);
}

void func2(double* sigma, double* dir) {
      __enzyme_autodiff((void*)stress_calculation,
          2.0,
        sigma, dir);

}

// CHECK: void @func1(double* nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %sigma, double* nocapture nofree noundef nonnull align 8 dereferenceable(8) %dir)
// CHECK: void @func2(double* nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %sigma, double* nocapture nofree noundef nonnull align 8 dereferenceable(8) %dir)

