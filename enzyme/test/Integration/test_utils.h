

#if defined(__cplusplus) || defined(__APPLE__) || defined(ENZ_HEADERS)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#else
struct _IO_FILE;
extern struct _IO_FILE* stderr;
extern int fprintf(struct _IO_FILE *, const char*, ...);
extern int fflush(struct _IO_FILE *stream);
extern int printf(const char*, ...);
extern void abort();
extern void free(void *);
extern void* malloc(unsigned long);
extern void *realloc( void *ptr, unsigned long new_size );
extern void* memcpy( void* dest, const void* src, unsigned long count );
extern void* memset( void* dest, int, unsigned long count );
#endif

extern
#ifdef __cplusplus
"C"
#endif
int enzyme_allocated, enzyme_const, enzyme_dup, enzyme_dupnoneed, enzyme_out,
    enzyme_tape;

/*
#ifdef __cplusplus
extern "C" {
#endif
static inline bool approx_fp_equality_float(float f1, float f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}

static inline bool approx_fp_equality_double(double f1, double f2, double threshold) {
  if (fabs(f1-f2) > threshold) return false;
  return true;
}
#ifdef __cplusplus
}
#endif
*/

#define APPROX_EQ(LHS, RHS, THRES)                                    \
    {                                                                \
      if (__builtin_fabs((LHS) - (RHS)) > THRES) {                                               \
        fprintf(stderr, "Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\n", #LHS, (double)(LHS), #RHS, (double)(RHS), THRES, \
                __FILE__, __LINE__, __PRETTY_FUNCTION__);               \
        abort();                                                        \
      }                                                                 \
    };

#define TEST_EQ(LHS, RHS)                                    \
    {                                                                \
      if ((LHS) != (RHS)) {\
        fprintf(stderr, "Assertion Failed: [%s = %d] != [%s = %d] at %s:%d (%s)\n", #LHS, (int)(LHS), #RHS, (int)(RHS), \
                __FILE__, __LINE__, __PRETTY_FUNCTION__);               \
        abort();                                                        \
      }                                                                 \
    };
