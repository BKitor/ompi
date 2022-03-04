#ifndef BKPAP_KERNEL_H
#define BKPAP_KERNEL_H

#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"

/* Function: vecAdd
 * ------------------------------------------------------------
 * A blocking host function that will calculate a vector
 * addition (A = A + B) within the Open MPI library.
 *
 * In/Output:
 *  a: A device pointer of vector A
 *
 * Inputs:
 *  b: A device pointer of vector B
 *  n: The number of elements in vector A and B. It is
 *     predicted that n shall not be much larger than
 *     16777216 = (16 * 1024 * 1024).
 *
 * returns: nothing
 *
 * ----------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif
  void vec_add_float(float* in, float* in_out, int count);

  static inline void bk_gpu_op_reduce(ompi_op_t* op, void* source,void* target, size_t full_count, ompi_datatype_t* dtype){
    if(OPAL_LIKELY( MPI_FLOAT == dtype && MPI_SUM == op )){ // is sum float
      vec_add_float(source, target, full_count);
    } else {
      BKPAP_ERROR("Falling back to ompi impl");
      // FULL SEND TO A SEGV !!!
      ompi_op_reduce(op, source, target, full_count, dtype);
    }
  }

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
/* Function: vecAddImpl
 * ------------------------------------------------------------
 * The kernel function which implements the vector addition.
 * The actual function signature can change, if required, this
 * is only an example.
 * ----------------------------------------------------------*/

__global__ void vec_add_float_impl(float* in, float* in_out, int count);

#endif

#endif // BKPAP_KERNEL_H
