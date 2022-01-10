#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank, size;
    int *snd_bff, memsize;
    int g_err = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    memsize = 1 << 22;

    snd_bff = malloc(memsize * sizeof(int));

    int g_sum = 0;
    for(int i = 0; i<size; i++)
        g_sum += i;

    for (int i = 0; i < 8; i++)
    {
        int err = 0;
        for (int j = 0; j < memsize; j++)
        {
            snd_bff[j] = rank*i;
        }

        MPI_Allreduce(MPI_IN_PLACE, snd_bff, memsize, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int j = 0; j < memsize; j++)
        {
            if (snd_bff[j] != g_sum*i)
                err += 1;
        }

        if (err)
        {
            printf("ERROR: rank:%d snd_buff %d not equal round %d, shoudl be %d\n", rank, snd_bff[0], i, g_sum*i);
            g_err = 69;
        }else{
            if(rank == 0)printf("VAL SUCCESS: round %d, should be: %d, is: %d\n", i, g_sum*i, snd_bff[0]);
        }

        fflush(stdout);
        fflush(stderr);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(snd_bff);
    MPI_Finalize();
    return g_err;
}