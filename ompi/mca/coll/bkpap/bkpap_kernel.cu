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
__global__ void vec_add_float_impl(float *in, float *in_out, int count)
{
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < count) {
    in_out[id] = in[id] + in_out[id];
  }
}

extern "C" void vec_add_float(float *in, float *in_out, int count) {
  int Db = count < 1024 ? count : 1024;
  int Dg = ceil((float) count / (float) Db);

  int Ns = count * sizeof(float) < 48 * 1024
         ? count * sizeof(float)
         : 48 * 1024;

  cudaStream_t stream = get_stream();
  vec_add_float_impl<<<Dg, Db, Ns, stream>>>(in, in_out, count);
  cudaStreamSynchronize(stream);
}
