#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
    int rank, size;
    float* snd_bff, * rcv_bff;
    int g_err = 0;
    int count = 1 << 23;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)printf("BK VALIDATE ALLREDUCE, wsize %d\n", size);

    snd_bff = malloc(count * sizeof(*snd_bff));
    rcv_bff = malloc(count * sizeof(*rcv_bff));

    float g_sum = 0.0;
    for (int i = 0; i < size; i++) {
        g_sum += (float)i;
    }

    for (int i = 0; i < 8; i++) {
        int err = 0;
        for (int j = 0; j < count; j++) {
            rcv_bff[j] = 0;
            snd_bff[j] = (float)rank * i;
        }

        // MPI_Allreduce(snd_bff, rcv_bff, memsize, MPI_FLOAT, MPI_BXOR, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD); // TODO: Remove this when ss leaving is better implimented
        MPI_Allreduce(MPI_IN_PLACE, snd_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        for (int j = 0; j < count; j++) {
            if (snd_bff[j] != g_sum * i)
                err = j;
            if (err)
                break;
        }

        if (err) {
            printf("ERROR: rank:%d snd_buff %.2f not equal round %d, err_pos %d, should be %.2f\n", rank, snd_bff[err], i, err, g_sum * i);
            g_err = 69;
        }
        else {
            if (rank == 0)printf("VAL SUCCESS: round %d, should be: %.2f, is: %.2f\n", i, g_sum * i, snd_bff[0]);
        }

        fflush(stdout);
        fflush(stderr);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(snd_bff);
    MPI_Finalize();
    return g_err;
}