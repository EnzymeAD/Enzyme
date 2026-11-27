// The runtime declarations a kernel stub needs, for a compile without the CUDA
// or ROCm headers.
struct dim3 {
  unsigned x, y, z;
};
#ifdef __HIP__
typedef struct ihipStream_t *hipStream_t;
extern "C" int hipLaunchKernel(const void *, dim3, dim3, void **, unsigned long,
                               hipStream_t);
extern "C" int __hipPushCallConfiguration(dim3, dim3, unsigned long,
                                          hipStream_t);
extern "C" int __hipPopCallConfiguration(dim3 *, dim3 *, unsigned long *,
                                         hipStream_t *);
#else
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *, dim3, dim3, void **,
                                unsigned long, cudaStream_t);
#endif
