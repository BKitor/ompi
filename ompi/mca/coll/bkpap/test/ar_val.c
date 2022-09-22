#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define BK_CHK_MPI_GT_LBL(_ret, _msg, _lbl)if(MPI_SUCCESS != _ret){fprintf(stderr, _msg); goto _lbl;}
#define BK_CHK_NULL_GT_LBL(_ptr, _lbl)if(NULL == _ptr){fprintf(stderr, "ptr %s is NULL", #_ptr); goto _lbl;}
#define BK_VAL_ERROR(_msg) fprintf(stderr, "%s %s %d ERROR: %s\n", __FILE__, __FUNCTION__, __LINE__, _msg);

static inline void print_f_array(int rank, float* buffer, size_t size) {
    printf("Rank: %d:  ", rank);
    for (int i = 0; i < size; i++)
        printf("%.2f\t", buffer[i]);

    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    int ret = MPI_SUCCESS;
    float* snd_bff = NULL, * rcv_bff = NULL;
    int g_err = 0;
    int count = 1 << 23;

    ret = MPI_Init(&argc, &argv);
    BK_CHK_MPI_GT_LBL(ret, "MPI_Init failed", main_abort);

    ret = MPI_Comm_size(MPI_COMM_WORLD, &size);
    BK_CHK_MPI_GT_LBL(ret, "MPI_Comm_size failed", main_abort);
    ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    BK_CHK_MPI_GT_LBL(ret, "MPI_Comm_rank failed", main_abort);
    if (rank == 0)printf("BK VALIDATE ALLREDUCE, wsize %d\n", size);

    snd_bff = malloc(count * sizeof(*snd_bff));
    BK_CHK_NULL_GT_LBL(snd_bff, main_abort);
    rcv_bff = malloc(count * sizeof(*rcv_bff));
    BK_CHK_NULL_GT_LBL(rcv_bff, main_abort);

    float g_sum = 0.0;
    for (int i = 0; i < size; i++) {
        g_sum += (float)i;
    }

    for (int i = 0; i < 8; i++) {
        int err = 0, f_err = -1;
        for (int j = 0; j < count; j++) {
            rcv_bff[j] = 0;
            snd_bff[j] = (float)rank * i;
        }

        ret = MPI_Barrier(MPI_COMM_WORLD); // TODO: Remove this when ss leaving is better implimented
        BK_CHK_MPI_GT_LBL(ret, "MPI_Barrier failed", main_abort);
        ret = MPI_Allreduce(MPI_IN_PLACE, snd_bff, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        BK_CHK_MPI_GT_LBL(ret, "MPI_Allreduce failed", main_abort);

        for (int j = 0; j < count; j++) {
            if (snd_bff[j] != g_sum * i){
				if(-1 == f_err) f_err = j;
                err = j+1;
			}
            if (err)
                break;
        }

        if (err) {
            printf("ERROR: rank:%d result %.2f not equal round %d, err_pos %d, should be %.2f\n", rank, snd_bff[err-1], i, err-1, g_sum * i);
            g_err = 69;
        }
        else {
            if (rank == 0)printf("VAL SUCCESS: round %d, should be: %.2f, is: %.2f\n", i, g_sum * i, snd_bff[0]);
        }

        fflush(stdout);
        fflush(stderr);
        ret = MPI_Barrier(MPI_COMM_WORLD); // TODO: Remove this when ss leaving is better implimented
        BK_CHK_MPI_GT_LBL(ret, "MPI_Barrier failed", main_abort);

        // float* pain_buf = malloc(sizeof(*pain_buf) * count * count);
        // MPI_Gather(snd_bff, count, MPI_FLOAT, pain_buf, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // if (0 == rank)
        //     for (int i = 0; i < size; i++)
        //         print_f_array(i, pain_buf + (i * count), count);
        // free(pain_buf);
    }

main_abort:
    free(snd_bff);
    free(rcv_bff);
    ret = MPI_Finalize();
    BK_CHK_MPI_GT_LBL(ret, "MPI_Finalize failed", main_abort);

    return g_err;
}
