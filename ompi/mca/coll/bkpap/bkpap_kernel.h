#ifndef BKPAP_KERNEL_H
#define BKPAP_KERNEL_H

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
