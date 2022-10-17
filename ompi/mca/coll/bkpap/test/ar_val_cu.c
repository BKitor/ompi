#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "mpi.h"

#define BK_NUM_ITERS 5

int main(int argc, char* argv[]) {
    int rank, size;
    float* snd_bff, * rcv_bff, * loc_tmp;
    int g_err = 0;
    int count = 1 << 23;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaSetDevice(rank);

    size_t bff_size = count * sizeof(*snd_bff);
    cudaMalloc((void*)&snd_bff, bff_size);
    cudaMalloc((void*)&rcv_bff, bff_size);
    loc_tmp = malloc(bff_size);

    float g_sum = 0.0;
    for (int i = 0; i < size; i++) {
        g_sum += (float)i;
    }

    // if (rank == 0)printf("BK VALIDATE ALLREDUCE, wsize %d, vec_size: %d/%ld, %p \n", size, count, bff_size, rcv_bff);
    printf("BK VALIDATE ALLREDUCE, %d of %d, vec count: %d, bff_size: %ld, rcv_ptr: [%p] \n", rank, size, count, bff_size, rcv_bff);

    for (int i = 0; i < BK_NUM_ITERS; i++) {
        int err = 0, f_err = -1;
        for (int j = 0; j < count; j++) {
            loc_tmp[j] = (float)rank * i;
        }

        cudaMemcpy(rcv_bff, loc_tmp, bff_size, cudaMemcpyHostToDevice);
        cudaMemcpy(snd_bff, loc_tmp, bff_size, cudaMemcpyHostToDevice);

        MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Allreduce(snd_bff, rcv_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, snd_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // cudaMemcpy(loc_tmp, rcv_bff, bff_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(loc_tmp, snd_bff, bff_size, cudaMemcpyDeviceToHost);
        
        MPI_Barrier(MPI_COMM_WORLD);

        for (int j = 0; j < count; j++) {
            if (loc_tmp[j] != g_sum * i) {
                if (-1 == f_err)f_err = j;
                err = j;
            }
        }

        if (err) {
            printf("ERROR: rank:%d range %d:%d samples: %.2f/%.2f not equal, should be %.2f\n", rank, f_err, err, loc_tmp[f_err], loc_tmp[err], g_sum * i);
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
