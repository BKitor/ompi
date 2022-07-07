#include "ompi_config.h"
#include "mpi.h"

#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/op/op.h"

#include "opal/cuda/common_cuda.h"
#include "opal/util/bit_ops.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop


// Only supports power of 2 numprocs 
static inline int coll_bkpap_papaware_rsa_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module) {

    int ret = OMPI_SUCCESS;
    int inter_size = ompi_comm_size(inter_comm), inter_rank = ompi_comm_rank(inter_comm);
    int intra_rank = ompi_comm_rank(intra_comm);
    int is_inter = (0 == intra_rank);
    int* ex_rank_array = NULL;
    int* recv_idx = NULL, * recv_count = NULL, * send_idx = NULL, * send_count = NULL;
    int num_rounds = opal_hibit(inter_size, inter_comm->c_cube_dim + 1);
    ptrdiff_t lb, extent;
    ompi_datatype_get_extent(dtype, &lb, &extent);

    BKPAP_OUTPUT("ARRIVE_AT_RSA: num_rounds: %d, data_size: 0x%lx, ranks: (inter: %d, intra: %d) rbuf_ptr: [%p]", num_rounds, (extent * count), inter_rank, intra_rank, rbuf);
    if (is_inter)BKPAP_PROFILE("bkpap_rsa_start_algorithm", inter_rank);

    if (OPAL_UNLIKELY(num_rounds > 0 && (1 << num_rounds) != inter_size)) { // only support power of 2 world size
        BKPAP_ERROR("inter size: %d not supported (num_rounds: %d (%d))", inter_size, num_rounds, (1 << num_rounds));
        ret = OMPI_ERR_NOT_SUPPORTED;
        goto bkpap_rsa_allreduce_exit;
    }
    if (OPAL_UNLIKELY(BKPAP_DATAPLANE_TAG != mca_coll_bkpap_component.dataplane_type)) { // only support TAG dataplane
        BKPAP_ERROR("RSA only supports TAG dataplane");
        ret = OMPI_ERR_NOT_SUPPORTED;
        goto bkpap_rsa_allreduce_exit;
    }

    // intra_reduce
    ret = bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_rsa_allreduce_exit);
    if (is_inter)BKPAP_PROFILE("bkpap_rsa_intra_reduce", inter_rank);

    if (is_inter) {
        mca_coll_bkpap_postbuf_memory_t recv_tmp_mem_t = mca_coll_bkpap_component.bk_postbuf_memory_type;
        void* recv_tmp = NULL;
        bkpap_mempool_alloc(&recv_tmp, extent * count, recv_tmp_mem_t, bkpap_module);

        int64_t arrival_pos = -1;
        ret = mca_coll_bkpap_arrive_ss(inter_rank, 0, 0, bkpap_module->remote_syncstructure, bkpap_module, inter_comm, &arrival_pos);
        BKPAP_CHK_MPI_MSG_LBL(ret, "arrive_ss failed", bkpap_rsa_allreduce_exit);
        int arrival = arrival_pos + 1;
        BKPAP_PROFILE("bkpap_rsa_get_arrival", inter_rank);

        // *** Step 1: Recusive doubling reduce scatter
        ex_rank_array = malloc(sizeof(*ex_rank_array) * num_rounds);
        send_count = malloc(sizeof(*send_count) * num_rounds);
        recv_count = malloc(sizeof(*recv_count) * num_rounds);
        send_idx = malloc(sizeof(*send_idx) * num_rounds);
        recv_idx = malloc(sizeof(*recv_idx) * num_rounds);
        if (OPAL_UNLIKELY(NULL == ex_rank_array || NULL == send_count || NULL == recv_count || NULL == send_idx || NULL == recv_idx)) {
            BKPAP_ERROR("ran out of memory allocating working buffers");
            ret = OPAL_ERR_OUT_OF_RESOURCE;
            goto bkpap_rsa_allreduce_exit;
        }
        BKPAP_PROFILE("bkpap_rsa_aloc_tmp_bff", inter_rank);
        memset(ex_rank_array, -1, sizeof(*ex_rank_array) * num_rounds);

        send_idx[0] = recv_idx[0] = 0;
        int round, win_count = count;
        uint64_t tag, tag_mask;
        for (round = 0; round < num_rounds; round++) {

            int exchange_arrival = arrival ^ (1 << round);
            int exchange_rank = -1;
            BKPAP_OUTPUT("RSA_START_ROUND: round: %d, rank: %d, win_count: %d, arrival: %d, exchange_arrival: %d", round, inter_rank, win_count, arrival, exchange_arrival);

            if (arrival < exchange_arrival) { // is early, peer not gaurenteed to have arrived
                recv_count[round] = win_count / 2;
                send_count[round] = win_count - recv_count[round];
                send_idx[round] = recv_idx[round] + recv_count[round];
                void* send_ptr = (int8_t*)rbuf + ((ptrdiff_t)send_idx[round] * extent);
                void* recv_ptr = (int8_t*)rbuf + ((ptrdiff_t)recv_idx[round] * extent);

                BK_RSA_MAKE_TAG(tag, tag_mask, num_rounds, round, 0);
                BKPAP_OUTPUT("RSA_REDUCE_EARLY: round: %d, arrival: %d, send_idx: %d, send_count: %d, recv_idx: %d, recv_count: %d",
                    round, arrival, send_idx[round], send_count[round], recv_idx[round], recv_count[round]);
                ret = bkpap_module->dplane_ftbl.sendrecv_from_late(send_ptr, send_count[round], recv_tmp, recv_count[round],
                    dtype, tag, tag_mask, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_dplane_sendrecv_from_late failed", bkpap_rsa_allreduce_exit);
                ret = mca_coll_bkpap_reduce_local(op, recv_tmp, recv_ptr, recv_count[round], dtype);
                BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_reduce_local failed", bkpap_rsa_allreduce_exit);

                BKPAP_OUTPUT("RSA_REDUCE_EARLY_DONE: round: %d, arrival: %d, exchange_rank: %d",
                    round, arrival, exchange_rank);

            }
            else { // is late, fetch peer, fetch rank from ss
                send_count[round] = win_count / 2;
                recv_count[round] = win_count - send_count[round];
                recv_idx[round] = send_idx[round] + send_count[round];
                void* send_ptr = (int8_t*)rbuf + ((ptrdiff_t)send_idx[round] * extent);
                void* recv_ptr = (int8_t*)rbuf + ((ptrdiff_t)recv_idx[round] * extent);

                while (-1 == exchange_rank) {
                    ret = mca_coll_bkpap_get_rank_of_arrival(exchange_arrival, 0, bkpap_module->remote_syncstructure, bkpap_module, &exchange_rank);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_rsa_allreduce_exit);
                }
                ex_rank_array[round] = exchange_rank;
                BK_RSA_MAKE_TAG(tag, tag_mask, num_rounds, round, 0);
                BKPAP_OUTPUT("RSA_REDUCE_LATE: round: %d, arrival: %d, exchange_rank: %d, send_idx: %d, send_count: %d, recv_idx: %d, recv_count: %d",
                    round, arrival, exchange_rank, send_idx[round], send_count[round], recv_idx[round], recv_count[round]);
                ret = bkpap_module->dplane_ftbl.sendrecv_from_early(send_ptr, send_count[round], recv_tmp, recv_count[round],
                    dtype, exchange_rank, tag, tag_mask, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "mca_coll_bkpap_dplane_sendrecv_from_early failed", bkpap_rsa_allreduce_exit);
                ret = mca_coll_bkpap_reduce_local(op, recv_tmp, recv_ptr, recv_count[round], dtype);
                BKPAP_CHK_MPI_MSG_LBL(ret, "mca_coll_bkpap_reduce_local failed", bkpap_rsa_allreduce_exit);

                BKPAP_OUTPUT("RSA_REDUCE_LATE_DONE: round: %d, arrival: %d, exchange_rank: %d",
                    round, arrival, exchange_rank);

            }
            if (round + 1 < num_rounds) {
                win_count = recv_count[round];
                send_idx[round + 1] = recv_idx[round];
                recv_idx[round + 1] = recv_idx[round];
            }
        }
        BKPAP_OUTPUT("RSA_FINISH_RS_START_A");
        BKPAP_PROFILE("bkpap_rsa_finish_rs", inter_rank);
        for (round = num_rounds - 1; round >= 0; round--) {
            int exchange_rank = ex_rank_array[round];
            void* recv_ptr = (int8_t*)rbuf + ((ptrdiff_t)recv_idx[round] * extent);
            void* send_ptr = (int8_t*)rbuf + ((ptrdiff_t)send_idx[round] * extent);
            BKPAP_OUTPUT("RSA_AG_STEP: round: %d, arrival: %d exchange_rank: %d, recv_idx: %d, recv_count: %d, send_idx: %d, send_count: %d", round, arrival, exchange_rank,
                recv_idx[round], recv_count[round], send_idx[round], send_count[round]);

            while (-1 == exchange_rank) {
                int exchange_arrival = arrival ^ (1 << round);
                ret = mca_coll_bkpap_get_rank_of_arrival(exchange_arrival, 0, bkpap_module->remote_syncstructure, bkpap_module, &exchange_rank);
                BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_rsa_allreduce_exit);
            }

            BK_RSA_MAKE_TAG(tag, tag_mask, num_rounds, round, 1);
            BKPAP_PROFILE("bkpap_rsa_enter_sendrecv", inter_rank);
            // ret = mca_coll_bkpap_sendrecv(recv_ptr, recv_count[round], send_ptr, send_count[round],
            ret = bkpap_module->dplane_ftbl.sendrecv(recv_ptr, recv_count[round], send_ptr, send_count[round],
                dtype, op, exchange_rank, tag, tag_mask, inter_comm, bkpap_module);
            BKPAP_PROFILE("bkpap_rsa_leave_sendrecv", inter_rank);
            BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_sendrecv failed", bkpap_rsa_allreduce_exit);
        }

        // assumes, last process to arrive is the last to leave
        if ((inter_size - 1) == arrival) {
            ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
            BKPAP_PROFILE("bkpap_rsa_reset_ss", inter_rank);
            BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_rsa_allreduce_exit);
        }
        bkpap_mempool_free(recv_tmp, recv_tmp_mem_t, bkpap_module);
    }

    // intra_bcast
    ret = intra_comm->c_coll->coll_bcast(
        rbuf, count, dtype, 0,
        intra_comm,
        intra_comm->c_coll->coll_bcast_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra-stage bcast failed", bkpap_rsa_allreduce_exit);
    if (is_inter)BKPAP_PROFILE("bkpap_rsa_intra_bcast", inter_rank);

bkpap_rsa_allreduce_exit:
    if (is_inter)BKPAP_PROFILE("bkpap_rsa_end_algorithm", inter_rank);
    if (NULL != ex_rank_array) free(ex_rank_array);
    if (NULL != send_idx) free(send_idx);
    if (NULL != recv_idx) free(recv_idx);
    if (NULL != send_count) free(send_count);
    if (NULL != recv_count) free(recv_count);
    return ret;
}

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype,
    struct ompi_op_t* op,
    struct ompi_communicator_t* comm,
    mca_coll_base_module_t* module) {
    mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
    int ret = OMPI_SUCCESS, alg = mca_coll_bkpap_component.allreduce_alg;
    size_t dsize = -1, total_dsize = -1;

    if (OPAL_UNLIKELY(alg >= BKPAP_ALLREDUCE_ALG_COUNT)) {
        BKPAP_ERROR("Selected alg %d not available, change OMPI_MCA_coll_bkpap_allreduce_alg", alg);
        ret = OMPI_ERR_BAD_PARAM;
        goto bkpap_ar_abort;
    }

    if (OPAL_UNLIKELY(!ompi_op_is_commute(op))) {
        BKPAP_ERROR("Commutative operation, going to fallback");
        ret = OMPI_ERR_BAD_PARAM;
        goto bkpap_ar_abort;
    }

    ompi_datatype_type_size(dtype, &dsize);
    total_dsize = dsize * (ptrdiff_t)count;
    if (OPAL_UNLIKELY(total_dsize > mca_coll_bkpap_component.postbuff_size)) {
        BKPAP_ERROR("Message size is bigger than postbuf, falling back");
        ret = OMPI_ERR_BAD_PARAM;
        goto bkpap_ar_abort;
    }

    // if IN_PLACE, rbuf is local contents, and will be used as local buffer 
    // if not IN_PLACE, copy sbuf into rbuf and act as if IN_PLACE
    if (MPI_IN_PLACE != sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, rbuf, (char*)sbuf);
        if (ret != OMPI_SUCCESS) {
            BKPAP_ERROR("Not in place memcpy failed, falling back");
            goto bkpap_ar_abort;
        }
        sbuf = MPI_IN_PLACE;
    }

    // Set up Intra/Inter comms
    if (OPAL_UNLIKELY(NULL == bkpap_module->intra_comm || NULL == bkpap_module->inter_comm)) {
        ret = mca_coll_bkpap_wireup_hier_comms(bkpap_module, comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("inter/intra communicator creation failed");
            goto bkpap_ar_abort;
        }
        BKPAP_OUTPUT("Wireup hier comm SUCCESS");
    }

    int global_wsize = ompi_comm_size(comm);
    int global_rank = ompi_comm_rank(comm);
    int intra_wsize = ompi_comm_size(bkpap_module->intra_comm);

    int is_multinode = (mca_coll_bkpap_component.force_flat) ? 0 : intra_wsize < global_wsize;
    struct ompi_communicator_t* ss_inter_comm = (is_multinode) ? bkpap_module->inter_comm : comm;
    struct ompi_communicator_t* ss_intra_comm = (is_multinode) ? bkpap_module->intra_comm : &ompi_mpi_comm_self.comm;

    BKPAP_OUTPUT("AR comm rank: %d, intra rank: %d, inter rank: %d, count: %d, is_multinode: %d, alg %d, rbuf: [%p]", ompi_comm_rank(comm),
        ompi_comm_rank(ss_intra_comm), ompi_comm_rank(ss_inter_comm), count, is_multinode, alg, rbuf);


    if (OPAL_UNLIKELY(!bkpap_module->ucp_is_initialized)) {
        if (0 == ompi_comm_rank(ss_intra_comm)) { // is internode
            ret = mca_coll_bkpap_lazy_init_module_ucx(bkpap_module, ss_inter_comm, alg);
            BKPAP_CHK_MPI(ret, bkpap_ar_abort);

#if OPAL_ENABLE_DEBUG
            if (0 == ompi_comm_rank(comm) && bkpap_module->num_syncstructures > 0) {
                int64_t* arrival_arr_tmp = bkpap_module->local_syncstructure->arrival_arr_attr.address;
                int64_t* count_arr_tmp = bkpap_module->local_syncstructure->counter_attr.address;
                char arrival_str[128] = { '\0' };
                bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_arrival_arr_len, arrival_arr_tmp, 128, arrival_str);
                char count_str[128] = { '\0' };
                bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_counter_len, count_arr_tmp, 128, count_str);
                BKPAP_OUTPUT("SS initalized at intra %d inter %d global %d, arirval_arr: %s, count_arr: %s", ompi_comm_rank(ss_intra_comm), ompi_comm_rank(ss_inter_comm), ompi_comm_rank(comm), arrival_str, count_str);
            }
#endif

        }
        ret = bkpap_init_mempool(bkpap_module);
        BKPAP_CHK_MPI(ret, bkpap_ar_abort);
        bkpap_module->ucp_is_initialized = 1;
    }

    switch (alg) {
    case BKPAP_ALLREDUCE_ALG_KTREE:
        ret = coll_bkpap_papaware_ktree_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
        ret = coll_bkpap_papaware_ktree_allreduce_pipelined(sbuf, rbuf, count, dtype, op, mca_coll_bkpap_component.pipeline_segment_size, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE:
        ret = coll_bkpap_papaware_ktree_allreduce_fullpipelined(sbuf, rbuf, count, dtype, op, mca_coll_bkpap_component.pipeline_segment_size, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_RSA:
        ret = coll_bkpap_papaware_rsa_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_BASE_RSA_GPU:
        ret = ompi_coll_bkpap_base_allreduce_intra_redscat_allgather_gpu(sbuf, rbuf, count, dtype, op, comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_BINOMIAL:
        ret = coll_bkpap_papaware_binomial_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_CHAIN:
        ret = coll_bkpap_papaware_chain_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    default:
        BKPAP_ERROR("alg %d undefined, falling back", alg);
        goto bkpap_ar_abort;
        break;
    }
    BKPAP_CHK_MPI(ret, bkpap_ar_abort);

    BKPAP_OUTPUT("rank: %d COMPLETE BKPAP ALLREDUCE", global_rank);
    bk_mempool_trim(bkpap_module);
    return ret;

bkpap_ar_abort:
    return ret;
}
