// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

#define __builtin_autodiff __enzyme_autodiff
double __enzyme_autodiff(void*, ...);


typedef struct {
  int i;
  float value1;
  float value2;
} structtest2;


typedef struct {
  long long int R_start;
  float* data;
} WINDOW_FRAME;

float tile_multiply(const WINDOW_FRAME* __restrict frame) {
  //long long int R_midpoint = (frame->R_start) + (frame->R_end-frame->R_start)/2;
  if (frame->R_start < 8) {
      float sum = frame->data[frame->R_start];//*window[(r+c)%10];
      //WINDOW_FRAME* frame1 = malloc(sizeof(WINDOW_FRAME));
      WINDOW_FRAME* frame2 = malloc(sizeof(WINDOW_FRAME));
      //frame1->data = frame->data;
      frame2->data = frame->data;
      //frame1->R_end = R_midpoint;
      frame2->R_start = frame->R_start + 1;//R_midpoint;

      //float sum1 = tile_multiply(frame1);
      float sum2 = tile_multiply(frame2);
      //free(frame1);
      free(frame2);
      return sum + sum2;
  } else {
    return 0;
  }
}

float tile_multiply_helper(WINDOW_FRAME* frame) {
  return tile_multiply(frame);
}

int main(int argc, char** argv) {

  float* data = (float*) malloc(sizeof(float) * 8);
  float* d_data = (float*) malloc(sizeof(float) * 8);
  for (int i = 0; i < 8; i++) {
    data[i] = i;
    d_data[i] = 0.0;
  }

  float loss = 0.0;
  float d_loss = 1.0;


  int R_start = 0;
  int R_end = 8;

  WINDOW_FRAME* frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  WINDOW_FRAME* d_frame = (WINDOW_FRAME*) malloc(sizeof(WINDOW_FRAME));
  frame->R_start = R_start;
  frame->data = data;
  d_frame->data = d_data;

  //float res = tile_multiply_helper(frame);
  //printf("result=%f\n", res);
  __enzyme_autodiff(tile_multiply_helper, frame, d_frame);

  for (int i = 0; i < 8; i++) {
    printf("gradient for %d is %f\n", i, d_frame->data[i]);
    APPROX_EQ(d_frame->data[i], 1.0, 1e-10);
  }

  return 0;
}
