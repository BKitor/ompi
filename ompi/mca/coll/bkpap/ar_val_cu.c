#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "mpi.h"

#define BK_NUM_ITERS 3

int main(int argc, char* argv[]) {
    int rank, size;
    float* snd_bff, * rcv_bff, *loc_tmp;
    int g_err = 0;
    int count = 1 << 23;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)printf("BK VALIDATE ALLREDUCE, wsize %d\n", size);
    
    cudaSetDevice(rank);

    size_t bff_size = count * sizeof(*snd_bff);
    cudaMallocManaged((void*)&snd_bff, bff_size, cudaMemAttachGlobal);
    cudaMallocManaged((void*)&rcv_bff, bff_size, cudaMemAttachGlobal);
    loc_tmp = malloc(bff_size);

    float g_sum = 0.0;
    for (int i = 0; i < size; i++) {
        g_sum += (float)i;
    }
    
	struct cudaPointerAttributes cu_attrs;
	cudaPointerGetAttributes(&cu_attrs, rcv_bff);
    int is_cuda_rbuf = (cudaMemoryTypeDevice  == cu_attrs.type || cudaMemoryTypeManaged == cu_attrs.type);
    printf("INTERNAL CHECK %d\n", is_cuda_rbuf);
    fflush(stdout);

    for (int i = 0; i < BK_NUM_ITERS; i++) {
        int err = 0;
        for (int j = 0; j < count; j++) {
            loc_tmp[j] = (float)rank * i;
        }

        cudaMemcpy(rcv_bff, loc_tmp, bff_size, cudaMemcpyHostToDevice);
        cudaMemcpy(snd_bff, loc_tmp, bff_size, cudaMemcpyHostToDevice);
        
        MPI_Allreduce(snd_bff, rcv_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // MPI_Allreduce(MPI_IN_PLACE, snd_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        cudaMemcpy(loc_tmp, rcv_bff, bff_size, cudaMemcpyDeviceToHost);

        for (int j = 0; j < count; j++) {
            if (loc_tmp[j] != g_sum * i)
                err += 1;
        }

        if (err) {
            printf("ERROR: rank:%d reult %.2f not equal round %d, shoudl be %.2f\n", rank, loc_tmp[0], i, g_sum * i);
            g_err = 69;
        }
        else {
            if (rank == 0)printf("VAL SUCCESS: round %d, should be: %.2f, is: %.2f\n", i, g_sum * i, loc_tmp[0]);
        }

        fflush(stdout);
        fflush(stderr);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    cudaFree(snd_bff);
    cudaFree(rcv_bff);
    free(loc_tmp);
    MPI_Finalize();
    return g_err;
}