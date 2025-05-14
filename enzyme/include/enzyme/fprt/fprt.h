#ifndef _ENZYME_FPRT_FPRT_H_
#define _ENZYME_FPRT_FPRT_H_

#include <stdint.h>

#define __ENZYME_MPFR_ATTRIBUTES                                               \
  [[maybe_unused]] __attribute__((weak)) __attribute__((used))
#define __ENZYME_MPFR_ORIGINAL_ATTRIBUTES                                      \
  __attribute__((weak)) __attribute__((used))
#define __ENZYME_MPFR_DEFAULT_ROUNDING_MODE GMP_RNDN

#define __ENZYME_MPFR_MALLOC_FAILURE_EXIT_STATUS 114

#ifdef __cplusplus
extern "C" {
#endif

// User-facing API
double __enzyme_fprt_64_52_get(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc);
double __enzyme_fprt_64_52_new(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc);
void __enzyme_fprt_64_52_delete(double a, int64_t exponent, int64_t significand,
                                int64_t mode, const char *loc);
double __enzyme_truncate_mem_value_d(double, int, int);
float __enzyme_truncate_mem_value_f(float, int, int);
double __enzyme_expand_mem_value_d(double, int, int);
float __enzyme_expand_mem_value_f(float, int, int);
void __enzyme_fprt_delete_all();

// For internal use
struct __enzyme_fp;
__enzyme_fp *__enzyme_fprt_64_52_new_intermediate(int64_t exponent,
                                                  int64_t significand,
                                                  int64_t mode,
                                                  const char *loc);
double __enzyme_fprt_64_52_const(double _a, int64_t exponent,
                                 int64_t significand, int64_t mode,
                                 const char *loc);

__ENZYME_MPFR_ATTRIBUTES bool __enzyme_fprt_is_mem_mode(int64_t mode) {
  return mode & 0b0001;
}
__ENZYME_MPFR_ATTRIBUTES bool __enzyme_fprt_is_op_mode(int64_t mode) {
  return mode & 0b0010;
}
__ENZYME_MPFR_ATTRIBUTES double __enzyme_fprt_idx_to_double(uint64_t p) {
  return *((double *)(&p));
}
__ENZYME_MPFR_ATTRIBUTES uint64_t __enzyme_fprt_double_to_idx(double d) {
  return *((uint64_t *)(&d));
}
__ENZYME_MPFR_ATTRIBUTES double __enzyme_fprt_ptr_to_double(__enzyme_fp *p) {
  return *((double *)(&p));
}
__ENZYME_MPFR_ATTRIBUTES __enzyme_fp *__enzyme_fprt_double_to_ptr(double d) {
  return *((__enzyme_fp **)(&d));
}

#ifdef __cplusplus
}
#endif

#endif // _ENZYME_FPRT_FPRT_H_
