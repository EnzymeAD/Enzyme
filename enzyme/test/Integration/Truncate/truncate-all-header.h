#ifndef TRUNCATE_ALL_HEADER_H_
#define TRUNCATE_ALL_HEADER_H_

#include <math.h>

#define N 6

#define floatty double

__attribute__((noinline)) static floatty intrinsics2(floatty a, floatty b) {
  return sin(a) * cos(b);
}

#endif // TRUNCATE_ALL_HEADER_H_
