#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"
#include "opal/util/bit_ops.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_util.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop


int ompi_coll_bkpap_base_allreduce_intra_redscat_allgather_gpu(
    const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype,
    struct ompi_op_t* op, struct ompi_communicator_t* comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    int* rindex = NULL, * rcount = NULL, * sindex = NULL, * scount = NULL;
    mca_coll_base_module_t* module = &bkpap_module->super;
    int comm_size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    BKPAP_PROFILE("base_rsa_gpu_start_algorithm", rank);
    BKPAP_OUTPUT("coll:base:allreduce_intra_redscat_allgather: rank %d/%d",
        rank, comm_size);

    /* Find nearest power-of-two less than or equal to comm_size */
    int nsteps = opal_hibit(comm_size, comm->c_cube_dim + 1);   /* ilog2(comm_size) */
    assert(nsteps >= 0);
    int nprocs_pof2 = 1 << nsteps;                              /* flp2(comm_size) */

    if (count < nprocs_pof2 || !ompi_op_is_commute(op)) {
        BKPAP_OUTPUT("coll:base:allreduce_intra_redscat_allgather: rank %d/%d "
            "count %d switching to basic linear allreduce",
            rank, comm_size, count);
        return ompi_coll_base_allreduce_intra_basic_linear(sbuf, rbuf, count, dtype,
            op, comm, module);
    }

    int err = MPI_SUCCESS;
    ptrdiff_t lb, extent, dsize, gap = 0;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    dsize = opal_datatype_span(&dtype->super, count, &gap);

    char* tmp_buf = NULL, * tmp_buf_raw = NULL;
    // err = bk_get_pbuff((void**)&tmp_buf_raw, bkpap_module);
    err = bkpap_get_mempool((void**)&tmp_buf_raw, dsize, bkpap_module);
    if (OMPI_SUCCESS != err)
        return err;
    tmp_buf = tmp_buf_raw - gap;

    if (sbuf != MPI_IN_PLACE) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf,
            (char*)sbuf);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

    /*
     * Step 1. Reduce the number of processes to the nearest lower power of two
     * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
     * 1. In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
     *    the second half of the input vector to their right neighbor (rank + 1)
     *    and all the odd ranks send the first half of the input vector to their
     *    left neighbor (rank - 1).
     * 2. All 2r processes compute the reduction on their half.
     * 3. The odd ranks then send the result to their left neighbors
     *    (the even ranks).
     *
     * The even ranks (0 to 2r - 1) now contain the reduction with the input
     * vector on their right neighbors (the odd ranks). The first r even
     * processes and the p - 2r last processes are renumbered from
     * 0 to 2^{\floor{\log_2 p}} - 1.
     */

    BKPAP_PROFILE("base_rsa_start_phase_1", rank);
    int vrank, step, wsize;
    int nprocs_rem = comm_size - nprocs_pof2;

    if (rank < 2 * nprocs_rem) {
        int count_lhalf = count / 2;
        int count_rhalf = count - count_lhalf;

        if (rank % 2 != 0) {
            /*
             * Odd process -- exchange with rank - 1
             * Send the left half of the input vector to the left neighbor,
             * Recv the right half of the input vector from the left neighbor
             */
            err = ompi_coll_base_sendrecv(rbuf, count_lhalf, dtype, rank - 1,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                (char*)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                count_rhalf, dtype, rank - 1,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* Reduce on the right half of the buffers (result in rbuf) */
            mca_coll_bkpap_reduce_local(op, (char*)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                (char*)rbuf + count_lhalf * extent, count_rhalf, dtype);
            // mca_coll_bkpap_reduce_local

            /* Send the right half to the left neighbor */
            err = MCA_PML_CALL(send((char*)rbuf + (ptrdiff_t)count_lhalf * extent,
                count_rhalf, dtype, rank - 1,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* This process does not pariticipate in recursive doubling phase */
            vrank = -1;

        }
        else {
            /*
             * Even process -- exchange with rank + 1
             * Send the right half of the input vector to the right neighbor,
             * Recv the left half of the input vector from the right neighbor
             */
            err = ompi_coll_base_sendrecv((char*)rbuf + (ptrdiff_t)count_lhalf * extent,
                count_rhalf, dtype, rank + 1,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                tmp_buf, count_lhalf, dtype, rank + 1,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* Reduce on the right half of the buffers (result in rbuf) */
            mca_coll_bkpap_reduce_local(op, tmp_buf, rbuf, count_lhalf, dtype);

            /* Recv the right half from the right neighbor */
            err = MCA_PML_CALL(recv((char*)rbuf + (ptrdiff_t)count_lhalf * extent,
                count_rhalf, dtype, rank + 1,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            vrank = rank / 2;
        }
    }
    else { /* rank >= 2 * nprocs_rem */
        vrank = rank - nprocs_rem;
    }

    BKPAP_PROFILE("base_rsa_end_phase_1", rank);
    /*
     * Step 2. Reduce-scatter implemented with recursive vector halving and
     * recursive distance doubling. We have p' = 2^{\floor{\log_2 p}}
     * power-of-two number of processes with new ranks (vrank) and result in rbuf.
     *
     * The even-ranked processes send the right half of their buffer to rank + 1
     * and the odd-ranked processes send the left half of their buffer to
     * rank - 1. All processes then compute the reduction between the local
     * buffer and the received buffer. In the next \log_2(p') - 1 steps, the
     * buffers are recursively halved, and the distance is doubled. At the end,
     * each of the p' processes has 1 / p' of the total reduction result.
     */
    rindex = malloc(sizeof(*rindex) * nsteps);
    sindex = malloc(sizeof(*sindex) * nsteps);
    rcount = malloc(sizeof(*rcount) * nsteps);
    scount = malloc(sizeof(*scount) * nsteps);
    if (NULL == rindex || NULL == sindex || NULL == rcount || NULL == scount) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        goto cleanup_and_return;
    }

    if (vrank != -1) {
        step = 0;
        wsize = count;
        sindex[0] = rindex[0] = 0;

        for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
            /*
             * On each iteration: rindex[step] = sindex[step] -- begining of the
             * current window. Length of the current window is storded in wsize.
             */
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

            if (rank < dest) {
                /*
                 * Recv into the left half of the current window, send the right
                 * half of the window to the peer (perform reduce on the left
                 * half of the current window)
                 */
                rcount[step] = wsize / 2;
                scount[step] = wsize - rcount[step];
                sindex[step] = rindex[step] + rcount[step];
            }
            else {
                /*
                 * Recv into the right half of the current window, send the left
                 * half of the window to the peer (perform reduce on the right
                 * half of the current window)
                 */
                scount[step] = wsize / 2;
                rcount[step] = wsize - scount[step];
                rindex[step] = sindex[step] + scount[step];
            }

            BKPAP_PROFILE("base_rsa_start_rs_sendrecv", rank);
            /* Send part of data from the rbuf, recv into the tmp_buf */
            err = ompi_coll_base_sendrecv((char*)rbuf + (ptrdiff_t)sindex[step] * extent,
                scount[step], dtype, dest,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                (char*)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                rcount[step], dtype, dest,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            BKPAP_PROFILE("base_rsa_end_rs_sendrecv", rank);
            /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
            mca_coll_bkpap_reduce_local(op, (char*)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                (char*)rbuf + (ptrdiff_t)rindex[step] * extent,
                rcount[step], dtype);
            BKPAP_PROFILE("base_rsa_end_rs_reduce", rank);

            /* Move the current window to the received message */
            if (step + 1 < nsteps) {
                rindex[step + 1] = rindex[step];
                sindex[step + 1] = rindex[step];
                wsize = rcount[step];
                step++;
            }
        }
        /*
         * Assertion: each process has 1 / p' of the total reduction result:
         * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
         */

         /*
          * Step 3. Allgather by the recursive doubling algorithm.
          * Each process has 1 / p' of the total reduction result:
          * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
          * All exchanges are executed in reverse order relative
          * to recursive doubling (previous step).
          */
        BKPAP_PROFILE("base_rsa_end_reduce_scatter", rank);

        step = nsteps - 1;

        for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

            /*
             * Send rcount[step] elements from rbuf[rindex[step]...]
             * Recv scount[step] elements to rbuf[sindex[step]...]
             */
            BKPAP_PROFILE("base_rsa_start_ag_sendrecv", rank);
            err = ompi_coll_base_sendrecv((char*)rbuf + (ptrdiff_t)rindex[step] * extent,
                rcount[step], dtype, dest,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                (char*)rbuf + (ptrdiff_t)sindex[step] * extent,
                scount[step], dtype, dest,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE, rank);
            BKPAP_PROFILE("base_rsa_end_ag_sendrecv", rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }
            step--;
        }
    }

    BKPAP_PROFILE("base_rsa_end_allgather", rank);
    /*
     * Step 4. Send total result to excluded odd ranks.
     */
    if (rank < 2 * nprocs_rem) {
        if (rank % 2 != 0) {
            /* Odd process -- recv result from rank - 1 */
            err = MCA_PML_CALL(recv(rbuf, count, dtype, rank - 1,
                MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                MPI_STATUS_IGNORE));
            if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

        }
        else {
            /* Even process -- send result to rank + 1 */
            err = MCA_PML_CALL(send(rbuf, count, dtype, rank + 1,
                MCA_COLL_BASE_TAG_ALLREDUCE,
                MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }
        }
    }
    BKPAP_PROFILE("base_rsa_end_phase_4", rank);

cleanup_and_return:
    // if (NULL != tmp_buf_raw)
    //     bk_free_pbufft(tmp_buf_raw);
    bkpap_reset_mempool(bkpap_module);
    if (NULL != rindex)
        free(rindex);
    if (NULL != sindex)
        free(sindex);
    if (NULL != rcount)
        free(rcount);
    if (NULL != scount)
        free(scount);
    BKPAP_PROFILE("base_rsa_gpu_end_algorithm", rank);
    return err;
}

/* copied function (with appropriate renaming) ends here */
