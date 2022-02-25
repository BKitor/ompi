#include "bkpap_kernel.h"
#include "stdio.h"
#include "math.h"
#include "cuda_runtime.h"

#define NUM_STREAMS 5

cudaStream_t pStreams[NUM_STREAMS];
int stream_idx = 0;
int initalized = 0;

static inline cudaStream_t get_stream() {
  if (!initalized) {
    for (int i=0; i<NUM_STREAMS; i++) {
      cudaStreamCreate(&pStreams[i]);
    }
    initalized = 1;
  }
  stream_idx = (stream_idx + 1) % NUM_STREAMS;
  return pStreams[stream_idx];
}

// Calculated A = A + B
__global__ void vecAddImpl(float *a, float *b, int n)
{
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    float tmp_buf = a[id] + b[id];
    a[id] = tmp_buf;
    b[id] = tmp_buf;
  }
}

extern "C" void vecAdd(float *a, float *b, int n) {
  int Db = n < 1024 ? n : 1024;
  int Dg = ceil((float) n / (float) Db);

  int Ns = n * sizeof(float) < 48 * 1024
         ? n * sizeof(float)
         : 48 * 1024;

  cudaStream_t stream = get_stream();
  vecAddImpl<<<Dg, Db, Ns, stream>>>(a, b, n);
  cudaStreamSynchronize(stream);
}
