#include "ompi_config.h"
#include "mpi.h"

#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/op/op.h"

#include "opal/cuda/common_cuda.h"
#include "opal/util/bit_ops.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

// returns error if runs out of space
static int _bk_fill_array_str_ld(size_t arr_len, int64_t* arr, size_t str_limit, char* out_str) {
    if (str_limit < 3) return OMPI_ERROR;
    char tmp[16] = { "\0" };
    *out_str = '\0';
    strcat(out_str, "[");
    for (size_t i = 0; i < arr_len; i++) {
        if (i == 0)
            sprintf(tmp, " %ld", arr[i]);
        else
            sprintf(tmp, ", %ld", arr[i]);

        if (strlen(tmp) > (str_limit - strlen(out_str)))
            return OMPI_ERROR;

        strcat(out_str, tmp);
    }

    if (strlen(out_str) > (str_limit + 1))
        return OMPI_ERROR;
    strcat(out_str, " ]");
    return OMPI_SUCCESS;
}

static inline int _bk_int_pow(int base, int exp) {
    int res = 1;
    while (1) {
        if (exp & 1)
            res *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
    return res;
}

// floor of log base k
static inline int _bk_log_k(int k, int n) {
    int ret = 0;
    for (int i = 1; i < n; i *= k)
        ret++;
    return ret;
}

static inline int bk_request_wait_all(ompi_request_t** request_arr, int req_arr_len) {
    int tmp_is_completed;
    ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    while (!tmp_is_completed) {
        ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
        ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    }
    return OMPI_SUCCESS;
}

static inline int _bk_intra_reduce(void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* intra_comm, mca_coll_bkpap_module_t* bkpap_module) {
    int intra_rank = ompi_comm_rank(intra_comm);

    void* intra_reduce_sbuf = (0 == intra_rank) ? MPI_IN_PLACE : rbuf;
    void* intra_reduce_rbuf = (0 == intra_rank) ? rbuf : NULL;

    switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
    case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
        return intra_comm->c_coll->coll_reduce(
            intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0,
            intra_comm,
            intra_comm->c_coll->coll_reduce_module);
        break;
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
        return mca_coll_bkpap_reduce_intra_inplace_binomial(intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0, intra_comm, bkpap_module, 0, 0);
        break;
    default:
        BKPAP_ERROR("Bad memory type, intra-node reduce failed");
        return OMPI_ERROR;
        break;
    }
}

// Only supports power of 2 numprocs 
static inline int _bk_papaware_rsa_allreduce(const void* sbuf, void* rbuf, int count,
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

    BKPAP_OUTPUT("ARRIVE_AT_RSA: num_rounds: %d, data_size: 0x%lx, rbuf_ptr: [0x%p]", num_rounds, (extent * count), rbuf);
    if(is_inter)BKPAP_PROFILE("bkpap_rsa_start_algorithm", inter_rank);

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
    _bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_rsa_allreduce_exit);
    if (is_inter)BKPAP_PROFILE("bkpap_rsa_intra_reduce", inter_rank);

    if (is_inter) {
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
                BKPAP_OUTPUT("RSA_REDUCE_EARLY: round: %d, arrival: %d, send_idx: %d, send_count: %d, recv_idx: %d, recv_count: %d, sample: (%.2f)",
                    round, arrival, send_idx[round], send_count[round], recv_idx[round], recv_count[round], ((float*)recv_ptr)[0]);
                BKPAP_PROFILE("bkpap_rsa_enter_early_exchange", inter_rank);
                ret = mca_coll_bkpap_reduce_early_p2p(send_ptr, send_count[round], recv_ptr, recv_count[round],
                    &exchange_rank, tag, tag_mask, dtype, op, inter_comm, bkpap_module);
                BKPAP_PROFILE("bkpap_rsa_leave_early_exchange", inter_rank);

                BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_reduce_early_p2p failed", bkpap_rsa_allreduce_exit);
                BKPAP_OUTPUT("RSA_REDUCE_EARLY_DONE: round: %d, arrival: %d, exchange_rank: %d, sample: (%.2f)",
                    round, arrival, exchange_rank, ((float*)recv_ptr)[0]);

                ex_rank_array[round] = exchange_rank;
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
                BKPAP_OUTPUT("RSA_REDUCE_LATE: round: %d, arrival: %d, exchange_rank: %d, send_idx: %d, send_count: %d, recv_idx: %d, recv_count: %d, sample: (%.2f)",
                    round, arrival, exchange_rank, send_idx[round], send_count[round], recv_idx[round], recv_count[round], ((float*)recv_ptr)[0]);
                BKPAP_PROFILE("bkpap_rsa_enter_late_exchange", inter_rank);
                ret = mca_coll_bkpap_reduce_late_p2p(send_ptr, send_count[round], recv_ptr, recv_count[round],
                    exchange_rank, tag, tag_mask, dtype, op, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_reduce_late_p2p failed", bkpap_rsa_allreduce_exit);
                BKPAP_PROFILE("bkpap_rsa_leave_late_exchange", inter_rank);
                BKPAP_OUTPUT("RSA_REDUCE_LATE_DONE: round: %d, arrival: %d, exchange_rank: %d, sample: (%.2f)",
                    round, arrival, exchange_rank, ((float*)recv_ptr)[0]);

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

            BK_RSA_MAKE_TAG(tag, tag_mask, num_rounds, round, 1);
            BKPAP_PROFILE("bkpap_rsa_enter_sendrecv", inter_rank);
            ret = mca_coll_bkpap_sendrecv(recv_ptr, recv_count[round], send_ptr, send_count[round],
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
    }

    // intra_bcast
    ret = intra_comm->c_coll->coll_bcast(
        rbuf, count, dtype, 0,
        intra_comm,
        intra_comm->c_coll->coll_bcast_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra-stage bcast failed", bkpap_rsa_allreduce_exit);

    if (is_inter)BKPAP_PROFILE("bkpap_rsa_intra_bcast", inter_rank);

bkpap_rsa_allreduce_exit:
    if(is_inter)BKPAP_PROFILE("bkpap_rsa_end_algorithm", inter_rank);
    if (NULL != ex_rank_array) free(ex_rank_array);
    if (NULL != send_idx) free(send_idx);
    if (NULL != recv_idx) free(recv_idx);
    if (NULL != send_count) free(send_count);
    if (NULL != recv_count) free(recv_count);
    return ret;
}


static inline int _bk_papaware_ktree_allreduce_fullpipelined(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int inter_rank = ompi_comm_rank(inter_comm), inter_size = ompi_comm_size(inter_comm);
    int intra_rank = ompi_comm_rank(intra_comm), intra_size = ompi_comm_size(intra_comm);
    int k = mca_coll_bkpap_component.allreduce_k_value;
    int is_inter = (0 == intra_rank);

    // if(seg_size >  type_size * seg_count) seg_count = count;
    int seg_count = count;
    size_t type_size;
    ptrdiff_t type_extent, type_lb;
    ompi_datatype_get_extent(dtype, &type_lb, &type_extent);
    ompi_datatype_type_size(dtype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int num_segments = (count + seg_count - 1) / seg_count;
    size_t real_seg_size = (ptrdiff_t)seg_count * type_extent;

    BKPAP_OUTPUT("KTREE_FULLPIPE_ARRIVE, rank: %d, count: %d, seg_size: %ld, dtype_size: %ld, num_segments:%d, seg_count: %d, inter_size: %d, intra_size: %d",
        inter_rank, count, seg_size, type_size, num_segments, seg_count, inter_size, intra_size);
    if (is_inter) { // do PAP-Aware stuff
        BKPAP_PROFILE("arrive_at_fullpipe", inter_rank);
        ompi_request_t* inter_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* intra_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* ss_reset_barrier_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* tmp_bcast_wait_arr[3] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

        uint8_t* seg_buf = rbuf;
        int seg_iter_count = seg_count, prev_seg_iter_count = 0;
        int inter_bcast_root = -1;

        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            BKPAP_PROFILE("start_new_segment", inter_rank);
            int64_t arrival_pos = -1;
            int phase_selector = (seg_index % 2);
            mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(bkpap_module->remote_syncstructure[phase_selector]);
            if ((num_segments - 1) == seg_index) { // if last segment, and allignment doesn't line up with seg-count
                seg_iter_count = count - (seg_index * seg_count);
            }

            BKPAP_OUTPUT("START_SEG: seg_index: %d, seg_iter_count: %d, num_segments: %d, rank: %d, phase_selector: %d inv(%d)",
                seg_index, seg_iter_count, num_segments, inter_rank, phase_selector, (phase_selector ^ 1));

            // Wait Inter and Intra ibcasts
            tmp_bcast_wait_arr[0] = intra_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[1] = inter_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[2] = ss_reset_barrier_reqs[phase_selector];
            ret = bk_request_wait_all(tmp_bcast_wait_arr, 3);
            BKPAP_CHK_MPI_MSG_LBL(ret, "leader bk_request_wait_all failed", bkpap_fullpipe_allreduce_exit);

            BKPAP_PROFILE("leave_new_seg_wait", inter_rank);

            BKPAP_OUTPUT("LEAVE_WAIT_ALL: seg_index: %d, rank: %d, phase_selector: %d", seg_index, inter_rank, phase_selector);

            if (seg_index > 1 && intra_size > 1) { // Launch Intra-bcast
                uint8_t* intra_buf = rbuf;
                intra_buf += (seg_index - 2) * real_seg_size;
                BKPAP_OUTPUT("STARTING_INTRA_IBCAST: seg_index: %d, rank: %d, bc_buf: [0x%p]", seg_index, inter_rank, (void*)intra_buf);
                intra_comm->c_coll->coll_ibcast(
                    intra_buf, prev_seg_iter_count, dtype, 0, intra_comm,
                    &intra_bcast_reqs[phase_selector], intra_comm->c_coll->coll_ibcast_module);
            }

            BKPAP_OUTPUT("INTRA_REDUCE_SEGMENT: seg_index: %d, rank: %d, count: %d, red_buf: [0x%p]", seg_index, inter_rank, seg_iter_count, (void*)seg_buf);
            ret = _bk_intra_reduce(seg_buf, seg_iter_count, dtype, op, intra_comm, bkpap_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_fullpipe_allreduce_exit);
            BKPAP_PROFILE("finish_intra_reduce_seg", inter_rank);

            int sync_mask = 1;
            int tree_height = _bk_log_k(k, ompi_comm_size(inter_comm));
            for (int sync_round = 0; sync_round < tree_height; sync_round++) { // pap-aware loop
                BKPAP_PROFILE("starting_new_sync_round", inter_rank);
                uint64_t counter_offset = 0;
                size_t arrival_arr_offset = 0;
                if (-1 == arrival_pos) {
                    ret = mca_coll_bkpap_arrive_ss(inter_rank, counter_offset, arrival_arr_offset, remote_ss_tmp, bkpap_module, inter_comm, &arrival_pos);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "arrive at inter failed", bkpap_fullpipe_allreduce_exit);
                    arrival_pos += 1;

                    BKPAP_OUTPUT("ARRIVED_AT_SS: seg_index: %d, rank: %d, arrival_pos: %ld, inter_bcast_root: %d", seg_index, inter_rank, arrival_pos, inter_bcast_root);
                    while (-1 == inter_bcast_root) {
                        int arrival_round_offset = 0;
                        ret = mca_coll_bkpap_get_rank_of_arrival(0, arrival_round_offset, remote_ss_tmp, bkpap_module, &inter_bcast_root);
                        BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival failed", bkpap_fullpipe_allreduce_exit);
                    }
                    BKPAP_OUTPUT("GOT_TREE_ROOT: seg_index: %d, rank: %d, arrive: %ld, inter_bcast_root:%d ", seg_index, inter_rank, arrival_pos, inter_bcast_root);
                    BKPAP_PROFILE("synced_at_ss", inter_rank);
                }

                sync_mask *= k;

                if (0 == arrival_pos % sync_mask) { // if-parent reduce
                    // TODO: this num_reduction logic only works for (k == 4) and (wsize is a power of 2),
                    // TODO: might want to fix at some point
                    int num_reductions = (sync_mask > inter_size) ? 1 : k - 1;
                    BKPAP_OUTPUT("START_PBUF_REDUCE: seg_index: %d, rank: %d, num_reductions: %d, count: %d", seg_index, inter_rank, num_reductions, seg_iter_count);
                    BKPAP_PROFILE("start_postbuf_reduce", inter_rank);
                    ret = mca_coll_bkpap_reduce_dataplane(seg_buf, dtype, seg_iter_count, op, num_reductions, inter_comm, bkpap_module);
                    BKPAP_PROFILE("leave_postbuf_reduce", inter_rank);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "reduce postbuf failed", bkpap_fullpipe_allreduce_exit);
                    BKPAP_OUTPUT("LEAVE_PBUF_REDUCE: seg_index: %d, rank: %d, sync_round: %d", seg_index, inter_rank, sync_round);
                } // if-parent reduce
                else { // else-child send parent
                    // sending
                    int send_rank = -1;
                    int send_arrival_pos = arrival_pos - (arrival_pos % sync_mask);
                    int arrival_round_offset = 0;
                    BKPAP_OUTPUT("GET_PARENT: seg_index: %d, rank: %d, arrival: %ld, parent_arrival: %d, sync_mask: %d", seg_index, inter_rank, arrival_pos, send_arrival_pos, sync_mask);
                    while (send_rank == -1) {
                        ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, arrival_round_offset, remote_ss_tmp, bkpap_module, &send_rank);
                        BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival faild", bkpap_fullpipe_allreduce_exit);
                    }
                    BKPAP_PROFILE("got_parent_rank", inter_rank);

                    int slot = ((arrival_pos / (sync_mask / k)) % k) - 1;

                    BKPAP_OUTPUT("SEND_PARENT: seg_index: %d, rank: %d, arrival: %ld, send_rank: %d, send_arrival: %d, slot: %d, sync_mask: %d",
                        seg_index, inter_rank, arrival_pos, send_rank, send_arrival_pos, slot, sync_mask);
                    ret = mca_coll_bkpap_send_dataplane(seg_buf, dtype, seg_iter_count, send_rank, slot, inter_comm, bkpap_module);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "write parrent postuf failed", bkpap_fullpipe_allreduce_exit);
                    BKPAP_PROFILE("sent_parent_rank", inter_rank);
                    break;
                } // else-child send parent
            } // pap-aware loop

            if (inter_rank == inter_bcast_root) {
                BKPAP_OUTPUT("RESET_SS: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
                ret = mca_coll_bkpap_reset_remote_ss(remote_ss_tmp, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_fullpipe_allreduce_exit);
                BKPAP_PROFILE("reset_remote_ss", inter_rank);
            }

            BKPAP_OUTPUT("STARTING_INTER_IBCAST: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
            inter_comm->c_coll->coll_ibarrier(inter_comm, &(ss_reset_barrier_reqs[phase_selector]), inter_comm->c_coll->coll_ibarrier_module);
            inter_comm->c_coll->coll_ibcast(
                seg_buf, seg_iter_count, dtype, inter_bcast_root, inter_comm,
                &(inter_bcast_reqs[phase_selector]), inter_comm->c_coll->coll_ibcast_module);
            inter_bcast_root = -1;

            prev_seg_iter_count = seg_iter_count;
            seg_buf += real_seg_size;
        }
        BKPAP_PROFILE("leave_main_loop", inter_rank);

        BKPAP_OUTPUT("STARTING_CLEANUP: rank: %d", inter_rank);

        int init_cleanup_phase = num_segments % 2;

        for (int cleanup_index = 0; cleanup_index < 2; cleanup_index++) {
            if (1 == num_segments)
                cleanup_index++;

            int phase_selector = (init_cleanup_phase + cleanup_index) % 2;

            tmp_bcast_wait_arr[0] = inter_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[1] = intra_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[2] = ss_reset_barrier_reqs[phase_selector];
            ret = bk_request_wait_all(tmp_bcast_wait_arr, 3);
            BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_fullpipe_allreduce_exit);
            BKPAP_PROFILE("leave_cleanup_wait", inter_rank);
            BKPAP_OUTPUT("FINISHED_CLEANUP_WAIT: rank: %d, cleanup_idx: %d", inter_rank, cleanup_index);

            seg_buf = rbuf;
            seg_buf += (real_seg_size * (num_segments - (2 - cleanup_index)));
            int bcast_count = (0 == cleanup_index) ? seg_count : count - ((num_segments - 1) * seg_count);

            intra_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, 0, intra_comm,
                &intra_bcast_reqs[phase_selector], intra_comm->c_coll->coll_ibcast_module
            );

            seg_buf += real_seg_size;
        }

        ret = bk_request_wait_all(intra_bcast_reqs, 2);
        BKPAP_CHK_MPI_MSG_LBL(ret, "leader bk_request_wait_all failed", bkpap_fullpipe_allreduce_exit);
        BKPAP_PROFILE("final_cleanup_wait", inter_rank);
        BKPAP_OUTPUT("CLEANUP_DONE: rank: %d", inter_rank);
    }
    else { // is-intra, do reduce and bcast
        int  red_count = seg_count, bc_count = seg_count;
        uint8_t* red_buf = rbuf, * bc_buf = rbuf;
        ompi_request_t* bc_req = MPI_REQUEST_NULL;

        // start first pipeline segment
        ret = _bk_intra_reduce(red_buf, red_count, dtype, op, intra_comm, bkpap_module);
        BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader intra-node reduce failed", bkpap_fullpipe_allreduce_exit);
        red_buf += real_seg_size;

        for (int seg_index = 1; seg_index < num_segments; seg_index++) {
            if ((num_segments - 1) == seg_index) {
                red_count = count - (seg_index * seg_count);
            }
            BKPAP_OUTPUT("IS INTRA, rank: %d, seg_index: %d, num_segments: %d, red_count: %d, bc_count: %d, red_buf: [0x%p], bc_buf: [0x%p]",
                intra_rank, seg_index, num_segments, red_count, bc_count, (void*)red_buf, (void*)bc_buf);

            ret = intra_comm->c_coll->coll_ibcast(
                bc_buf, bc_count, dtype, 0, intra_comm, &bc_req,
                intra_comm->c_coll->coll_ibcast_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader intra-node ibcast failed", bkpap_fullpipe_allreduce_exit);
            ret = _bk_intra_reduce(red_buf, red_count, dtype, op, intra_comm, bkpap_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader intra-node reduce failed", bkpap_fullpipe_allreduce_exit);

            ret = ompi_request_wait(&bc_req, MPI_STATUS_IGNORE);
            BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader ompi_request_wait_all failed", bkpap_fullpipe_allreduce_exit);
            red_buf += real_seg_size;
            bc_buf += real_seg_size;
        }

        // cleanup last segment
        bc_count = count - ((num_segments - 1) * seg_count);
        ret = intra_comm->c_coll->coll_ibcast(
            bc_buf, bc_count, dtype, 0, intra_comm, &bc_req,
            intra_comm->c_coll->coll_ibcast_module);
        BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader intra-node ibcast failed", bkpap_fullpipe_allreduce_exit);
        ret = ompi_request_wait(&bc_req, MPI_STATUS_IGNORE);
        BKPAP_CHK_MPI_MSG_LBL(ret, "non-leader ompi_request_wait_all failed", bkpap_fullpipe_allreduce_exit);
    }


    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("ktree_fullpipe_leave", inter_rank);

bkpap_fullpipe_allreduce_exit:
    return ret;
}

static inline int _bk_papaware_ktree_allreduce_pipelined(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int inter_rank = ompi_comm_rank(inter_comm), inter_size = ompi_comm_size(inter_comm);
    int intra_rank = ompi_comm_rank(intra_comm), intra_size = ompi_comm_size(intra_comm);
    int k = mca_coll_bkpap_component.allreduce_k_value;
    int is_inter = (0 == intra_rank);

    // if(seg_size >  type_size * seg_count) seg_count = count;
    int seg_count = count;
    size_t type_size;
    ptrdiff_t type_extent, type_lb;
    ompi_datatype_get_extent(dtype, &type_lb, &type_extent);
    ompi_datatype_type_size(dtype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int num_segments = (count + seg_count - 1) / seg_count;
    size_t real_seg_size = (ptrdiff_t)seg_count * type_extent;


    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("ktree_pipeline_arrive", inter_rank);


    ret = _bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra-node reduce failed", bkpap_pipeline_allreduce_exit);

    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("finish_intra_reduce", inter_rank);

    BKPAP_OUTPUT("KTREE_PIPELINE_ARRIVE, rank: %d, count: %d, seg_size: %ld, dtype_size: %ld, num_segments:%d, inter_size: %d, intra_size: %d",
        inter_rank, count, seg_size, type_size, num_segments, inter_size, intra_size);
    if (is_inter) { // Node leader, do pipelineing and pap-awareness
        ompi_request_t* inter_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* intra_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* ss_reset_barrier_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* tmp_bcast_wait_arr[3] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

        uint8_t* seg_buf = rbuf;
        int seg_iter_count = seg_count, prev_seg_iter_count = 0;
        int inter_bcast_root = -1;

        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            BKPAP_PROFILE("start_new_segment", inter_rank);
            int64_t arrival_pos = -1;
            int phase_selector = (seg_index % 2);
            mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(bkpap_module->remote_syncstructure[phase_selector]);
            if ((num_segments - 1) == seg_index) { // if last segment, and allignment doesn't line up with seg-count
                seg_iter_count = count - (seg_index * seg_count);
            }

            BKPAP_OUTPUT("START_SEG: seg_index: %d, num_segments: %d, rank: %d, phase_selector: %d", seg_index, num_segments, inter_rank, phase_selector);

            // Wait Inter and Intra ibcasts
            tmp_bcast_wait_arr[0] = intra_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[1] = inter_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[2] = ss_reset_barrier_reqs[phase_selector];
            bk_request_wait_all(tmp_bcast_wait_arr, 3);

            BKPAP_PROFILE("leave_new_seg_wait", inter_rank);

            BKPAP_OUTPUT("LEAVE_WAIT_ALL: seg_index: %d, rank: %d, phase_selector: %d", seg_index, inter_rank, phase_selector);

            if (seg_index > 1 && intra_size > 1) { // Launch Intra-bcast
                BKPAP_OUTPUT("STARTING_INTRA_IBCAST: seg_index: %d, rank: %d", seg_index, inter_rank);
                uint8_t* intra_buf = rbuf;
                intra_buf += (seg_index - 2) * real_seg_size;
                intra_comm->c_coll->coll_ibcast(
                    intra_buf, prev_seg_iter_count, dtype, 0, intra_comm,
                    &intra_bcast_reqs[phase_selector], intra_comm->c_coll->coll_ibcast_module);
            }

            int sync_mask = 1;
            int tree_height = _bk_log_k(k, ompi_comm_size(inter_comm));
            for (int sync_round = 0; sync_round < tree_height; sync_round++) { // pap-aware loop
                BKPAP_PROFILE("starting_new_sync_round", inter_rank);
                uint64_t counter_offset = 0;
                size_t arrival_arr_offset = 0;
                if (-1 == arrival_pos) {
                    ret = mca_coll_bkpap_arrive_ss(inter_rank, counter_offset, arrival_arr_offset, remote_ss_tmp, bkpap_module, inter_comm, &arrival_pos);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "arrive at inter failed", bkpap_pipeline_allreduce_exit);
                    arrival_pos += 1;

                    BKPAP_OUTPUT("ARRIVED_AT_SS: seg_index: %d, rank: %d, arrival_pos: %ld, inter_bcast_root: %d", seg_index, inter_rank, arrival_pos, inter_bcast_root);
                    while (-1 == inter_bcast_root) {
                        int arrival_round_offset = 0;
                        ret = mca_coll_bkpap_get_rank_of_arrival(0, arrival_round_offset, remote_ss_tmp, bkpap_module, &inter_bcast_root);
                        BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival failed", bkpap_pipeline_allreduce_exit);
                    }
                    BKPAP_OUTPUT("GOT_TREE_ROOT: seg_index: %d, rank: %d, arrive: %ld, inter_bcast_root:%d ", seg_index, inter_rank, arrival_pos, inter_bcast_root);
                    BKPAP_PROFILE("synced_at_ss", inter_rank);
                }

                sync_mask *= k;

                if (0 == arrival_pos % sync_mask) { // if-parent reduce
                    // TODO: this num_reduction logic only works for (k == 4) and (wsize is a power of 2),
                    // TODO: might want to fix at some point
                    int num_reductions = (sync_mask > inter_size) ? 1 : k - 1;
                    BKPAP_OUTPUT("START_PBUF_REDUCE: seg_index: %d, rank: %d, num_reductions: %d, count: %d", seg_index, inter_rank, num_reductions, seg_iter_count);
                    BKPAP_PROFILE("start_postbuf_reduce", inter_rank);
                    ret = mca_coll_bkpap_reduce_dataplane(seg_buf, dtype, seg_iter_count, op, num_reductions, inter_comm, bkpap_module);
                    BKPAP_PROFILE("leave_postbuf_reduce", inter_rank);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "reduce postbuf failed", bkpap_pipeline_allreduce_exit);
                    BKPAP_OUTPUT("LEAVE_PBUF_REDUCE: seg_index: %d, rank: %d, sync_round: %d", seg_index, inter_rank, sync_round);
                } // if-parent reduce
                else { // else-child send parent
                    // sending
                    int send_rank = -1;
                    int send_arrival_pos = arrival_pos - (arrival_pos % sync_mask);
                    int arrival_round_offset = 0;
                    BKPAP_OUTPUT("GET_PARENT: seg_index: %d, rank: %d, arrival: %ld, parent_arrival: %d, sync_mask: %d", seg_index, inter_rank, arrival_pos, send_arrival_pos, sync_mask);
                    while (send_rank == -1) {
                        ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, arrival_round_offset, remote_ss_tmp, bkpap_module, &send_rank);
                        BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival faild", bkpap_pipeline_allreduce_exit);
                    }
                    BKPAP_PROFILE("got_parent_rank", inter_rank);

                    int slot = ((arrival_pos / (sync_mask / k)) % k) - 1;

                    BKPAP_OUTPUT("SEND_PARENT: seg_index: %d, rank: %d, arrival: %ld, send_rank: %d, send_arrival: %d, slot: %d, sync_mask: %d",
                        seg_index, inter_rank, arrival_pos, send_rank, send_arrival_pos, slot, sync_mask);
                    ret = mca_coll_bkpap_send_dataplane(seg_buf, dtype, seg_iter_count, send_rank, slot, inter_comm, bkpap_module);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "write parrent postuf failed", bkpap_pipeline_allreduce_exit);
                    BKPAP_PROFILE("sent_parent_rank", inter_rank);
                    break;
                } // else-child send parent
            } // pap-aware loop

            if (inter_rank == inter_bcast_root) {
                BKPAP_OUTPUT("RESET_SS: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
                ret = mca_coll_bkpap_reset_remote_ss(remote_ss_tmp, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_pipeline_allreduce_exit);
                BKPAP_PROFILE("reset_remote_ss", inter_rank);
            }

            BKPAP_OUTPUT("STARTING_INTER_IBCAST: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
            ret = inter_comm->c_coll->coll_ibarrier(inter_comm, &(ss_reset_barrier_reqs[phase_selector]), inter_comm->c_coll->coll_ibarrier_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "inter ibarrier failed", bkpap_pipeline_allreduce_exit);
            inter_comm->c_coll->coll_ibcast(
                seg_buf, seg_iter_count, dtype, inter_bcast_root, inter_comm,
                &(inter_bcast_reqs[phase_selector]), inter_comm->c_coll->coll_ibcast_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "inter ibcast failed", bkpap_pipeline_allreduce_exit);
            inter_bcast_root = -1;

            prev_seg_iter_count = seg_iter_count;
            seg_buf += real_seg_size;
        }
        BKPAP_PROFILE("leave_main_loop", inter_rank);

        BKPAP_OUTPUT("STARTING_CLEANUP: rank: %d", inter_rank);

        int init_cleanup_phase = num_segments % 2;

        for (int cleanup_index = 0; cleanup_index < 2; cleanup_index++) {
            if (1 == num_segments)
                cleanup_index++;

            int phase_selector = (init_cleanup_phase + cleanup_index) % 2;

            tmp_bcast_wait_arr[0] = inter_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[1] = intra_bcast_reqs[phase_selector];
            tmp_bcast_wait_arr[2] = ss_reset_barrier_reqs[phase_selector];
            ret = bk_request_wait_all(tmp_bcast_wait_arr, 3);
            BKPAP_CHK_MPI_MSG_LBL(ret, "bkpap_req_wait_all failed", bkpap_pipeline_allreduce_exit);
            BKPAP_PROFILE("leave_cleanup_wait", inter_rank);
            BKPAP_OUTPUT("FINISHED_CLEANUP_WAIT: rank: %d, cleanup_idx: %d", inter_rank, cleanup_index);

            seg_buf = rbuf;
            seg_buf += (real_seg_size * (num_segments - (2 - cleanup_index)));
            int bcast_count = (0 == cleanup_index) ? seg_count : count - ((num_segments - 1) * seg_count);

            ret = intra_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, 0, intra_comm,
                &intra_bcast_reqs[phase_selector], intra_comm->c_coll->coll_ibcast_module
            );
            BKPAP_CHK_MPI_MSG_LBL(ret, "intra ibcast failed", bkpap_pipeline_allreduce_exit);

            seg_buf += real_seg_size;
        }

        ret = bk_request_wait_all(intra_bcast_reqs, 2);
        BKPAP_CHK_MPI_MSG_LBL(ret, "final bk_req_wait_all failed", bkpap_pipeline_allreduce_exit);
        BKPAP_PROFILE("final_cleanup_wait", inter_rank);
        BKPAP_OUTPUT("CLEANUP_DONE: rank: %d", inter_rank);
    }
    else {// non-leader, issues intra-bcasts
        int bcast_count = seg_count;
        ompi_request_t* bcast_req = (void*)OMPI_REQUEST_NULL;
        uint8_t* seg_buf = rbuf;
        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            if ((num_segments - 1) == seg_index) {
                bcast_count = count - (seg_index * seg_count);
            }
            BKPAP_OUTPUT("IS INTRA, rank: %d, seg_index: %d, num_segments: %d, count: %d", intra_rank, seg_index, num_segments, count);

            ret = intra_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, 0, intra_comm, &bcast_req,
                intra_comm->c_coll->coll_ibcast_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "intra ibcast failed", bkpap_pipeline_allreduce_exit);
            ret = ompi_request_wait(&bcast_req, MPI_STATUS_IGNORE);
            BKPAP_CHK_MPI_MSG_LBL(ret, "intra ompi_req_wait failed", bkpap_pipeline_allreduce_exit);
            seg_buf += real_seg_size;
        }
    }


    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("ktree_pipeline_leave", inter_rank);

bkpap_pipeline_allreduce_exit:
    return ret;
}

static inline int _bk_papaware_ktree_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int inter_rank = ompi_comm_rank(inter_comm), inter_size = ompi_comm_size(inter_comm);
    int intra_rank = ompi_comm_rank(intra_comm);
    int is_inter = (0 == intra_rank);
    int k = mca_coll_bkpap_component.allreduce_k_value;
    int64_t arrival_pos = -1;

    BKPAP_OUTPUT("ARRIVE AT KTREE, rank (%d, %d) comm: '%s', sbuf: %p rbuf: %p", inter_rank, intra_rank, intra_comm->c_name, sbuf, rbuf);
    if (is_inter)BKPAP_PROFILE("arrive_at_ktree", inter_rank);


    ret = _bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra-node reduce failed", bkpap_ktree_allreduce_exit);

    if (is_inter) {
        BKPAP_PROFILE("finish_intra_reduce", inter_rank);

#if OPAL_ENABLE_DEBUG
        int rbuf_is_cuda = opal_cuda_check_one_buf(rbuf, NULL);
        void* chk_pbuf = (BKPAP_DATAPLANE_RMA == mca_coll_bkpap_component.dataplane_type) ?
            bkpap_module->local_pbuffs.rma.postbuf_attrs.address : bkpap_module->local_pbuffs.tag.buff_arr;
        int pbuf_is_cuda = opal_cuda_check_one_buf(chk_pbuf, NULL);
        char offsets_str[64] = { '\0' };
        _bk_fill_array_str_ld(
            bkpap_module->remote_syncstructure->ss_counter_len, bkpap_module->remote_syncstructure->ss_arrival_arr_offsets, 64, offsets_str);
        BKPAP_OUTPUT("rank: %d, offsets %s, rbuf_is_cuda: %d, pbuf_is_cuda: %d", inter_rank, offsets_str, rbuf_is_cuda, pbuf_is_cuda);
#endif

        for (int sync_round = 0; sync_round < bkpap_module->remote_syncstructure->ss_counter_len; sync_round++) {
            uint64_t counter_offset = sync_round;
            uint64_t arrival_arr_offset = bkpap_module->remote_syncstructure->ss_arrival_arr_offsets[sync_round];

            // BKPAP_OUTPUT("round: %d, rank: %d, counter_offset: %ld, arrival_arr_offset: %ld", sync_round, inter_rank, counter_offset, arrival_arr_offset);
            ret = mca_coll_bkpap_arrive_ss(inter_rank, counter_offset, arrival_arr_offset, bkpap_module->remote_syncstructure, bkpap_module, inter_comm, &arrival_pos);
            BKPAP_CHK_MPI_MSG_LBL(ret, "arrive at inter failed", bkpap_ktree_allreduce_exit);
            arrival_pos += 1;
            BKPAP_OUTPUT("round: %d, rank: %d, arrive: %ld ", sync_round, inter_rank, arrival_pos);

            BKPAP_PROFILE("register_at_ss", inter_rank);

            if (0 == arrival_pos % k) {
                // receiving
                // TODO: this num_reduction logic only works for (k == 4) and (wsize is a power of 2),
                // TODO: might want to fix at some point
                int num_reductions = (_bk_int_pow(k, (sync_round + 1)) <= inter_size) ? k - 1 : 1;
                ret = mca_coll_bkpap_reduce_dataplane(rbuf, dtype, count, op, num_reductions, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "reduce postbuf failed", bkpap_ktree_allreduce_exit);
                BKPAP_PROFILE("reduce_pbuf", inter_rank);
            }
            else {
                // sending
                int send_rank = -1;
                int send_arrival_pos = arrival_pos - (arrival_pos % k);
                int arrival_round_offset = bkpap_module->remote_syncstructure->ss_arrival_arr_offsets[sync_round];
                BKPAP_OUTPUT("rank: %d, arrival: %ld, getting arrival: %d, offset: %ld", inter_rank, arrival_pos, send_arrival_pos, arrival_arr_offset);
                while (send_rank == -1) {
                    ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, arrival_round_offset, bkpap_module->remote_syncstructure, bkpap_module, &send_rank);
                    BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival faild", bkpap_ktree_allreduce_exit);
                }
                BKPAP_PROFILE("get_parent_rank", inter_rank);

                int slot = ((arrival_pos % k) - 1);

                ret = mca_coll_bkpap_send_dataplane(rbuf, dtype, count, send_rank, slot, inter_comm, bkpap_module);
                BKPAP_CHK_MPI_MSG_LBL(ret, "write parrent postuf failed", bkpap_ktree_allreduce_exit);
                BKPAP_PROFILE("send_parent_data", inter_rank);
                break;
            }
            arrival_pos = -1;
        }

        // this is polling over the network, this is gross
        int tree_root = -1;
        while (-1 == tree_root) {
            int arrival_round_offset = bkpap_module->remote_syncstructure->ss_arrival_arr_offsets[bkpap_module->remote_syncstructure->ss_counter_len - 1];
            ret = mca_coll_bkpap_get_rank_of_arrival(0, arrival_round_offset, bkpap_module->remote_syncstructure, bkpap_module, &tree_root);
            BKPAP_CHK_MPI_MSG_LBL(ret, "get rank of arrival failed", bkpap_ktree_allreduce_exit);
            // usleep(10);
        }
        BKPAP_PROFILE("get_leader_of_tree", inter_rank);

        // internode bcast
        ret = inter_comm->c_coll->coll_bcast(rbuf, count, dtype, tree_root, inter_comm, inter_comm->c_coll->coll_bcast_module);
        BKPAP_CHK_MPI_MSG_LBL(ret, "inter-stage bcast failed", bkpap_ktree_allreduce_exit);
        BKPAP_PROFILE("finish_inter_bcast", inter_rank);

        // TODO: desing reset-system that doesn't block 
        // hard-reset by rank 0 or last rank, and  check in arrival that arrival_pos < world_size
        inter_comm->c_coll->coll_barrier(inter_comm, inter_comm->c_coll->coll_barrier_module);
        if (is_inter && inter_rank == tree_root) {
            ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
            BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_ktree_allreduce_exit);
            BKPAP_PROFILE("reset_remote_ss", inter_rank);
        }

    }

    ret = intra_comm->c_coll->coll_bcast(
        rbuf, count, dtype, 0,
        intra_comm,
        intra_comm->c_coll->coll_bcast_module);
    BKPAP_CHK_MPI_MSG_LBL(ret, "intra-stage bcast failed", bkpap_ktree_allreduce_exit);
    if (is_inter) BKPAP_PROFILE("leave_intra_bcast", inter_rank);



#if OPAL_ENABLE_DEBUG
    if (0 == inter_rank && 0 == intra_rank) {
        int64_t* arrival_arr_tmp = bkpap_module->local_syncstructure->arrival_arr_attr.address;
        int64_t* count_arr_tmp = bkpap_module->local_syncstructure->counter_attr.address;
        char arrival_str[128] = { '\0' };
        _bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_arrival_arr_len, arrival_arr_tmp, 128, arrival_str);
        char count_str[128] = { '\0' };
        _bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_counter_len, count_arr_tmp, 128, count_str);
        BKPAP_OUTPUT("rank 0 leaving, arirval_arr: %s, count_arr: %s", arrival_str, count_str);
    }
#endif

bkpap_ktree_allreduce_exit:
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

    // TODO: Think about fastpath optimization

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

    int is_multinode = intra_wsize < global_wsize;
    struct ompi_communicator_t* ss_inter_comm = (is_multinode) ? bkpap_module->inter_comm : comm;
    struct ompi_communicator_t* ss_intra_comm = (is_multinode) ? bkpap_module->intra_comm : &ompi_mpi_comm_self.comm;

    BKPAP_OUTPUT("comm rank: %d, intra rank: %d, inter rank: %d, is_multinode: %d, alg %d", ompi_comm_rank(comm),
        ompi_comm_rank(ss_intra_comm), ompi_comm_rank(ss_inter_comm), is_multinode, alg);


    // if (OPAL_UNLIKELY((is_multinode && intra_rank == 0 && !bkpap_module->ucp_is_initialized)
    //     || (!is_multinode && !bkpap_module->ucp_is_initialized))) {
    if (OPAL_UNLIKELY(!bkpap_module->ucp_is_initialized && 0 == ompi_comm_rank(ss_intra_comm))) {
        ret = mca_coll_bkpap_lazy_init_module_ucx(bkpap_module, ss_inter_comm, alg);
        BKPAP_CHK_MPI(ret, bkpap_ar_abort);
#if OPAL_ENABLE_DEBUG
        if (0 == ompi_comm_rank(comm) && bkpap_module->num_syncstructures > 0) {
            int64_t* arrival_arr_tmp = bkpap_module->local_syncstructure->arrival_arr_attr.address;
            int64_t* count_arr_tmp = bkpap_module->local_syncstructure->counter_attr.address;
            char arrival_str[128] = { '\0' };
            _bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_arrival_arr_len, arrival_arr_tmp, 128, arrival_str);
            char count_str[128] = { '\0' };
            _bk_fill_array_str_ld(bkpap_module->remote_syncstructure->ss_counter_len, count_arr_tmp, 128, count_str);
            BKPAP_OUTPUT("SS initalized at intra %d inter %d global %d, arirval_arr: %s, count_arr: %s", ompi_comm_rank(ss_intra_comm), ompi_comm_rank(ss_inter_comm), ompi_comm_rank(comm), arrival_str, count_str);
        }
#endif
        bkpap_module->ucp_is_initialized = 1;
    }

    switch (alg) {
    case BKPAP_ALLREDUCE_ALG_KTREE:
        ret = _bk_papaware_ktree_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
        ret = _bk_papaware_ktree_allreduce_pipelined(sbuf, rbuf, count, dtype, op, mca_coll_bkpap_component.pipeline_segment_size, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE:
        ret = _bk_papaware_ktree_allreduce_fullpipelined(sbuf, rbuf, count, dtype, op, mca_coll_bkpap_component.pipeline_segment_size, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_RSA:
        ret = _bk_papaware_rsa_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_BASE_RSA_GPU:
        ret = ompi_coll_bkpap_base_allreduce_intra_redscat_allgather_gpu(sbuf, rbuf, count, dtype, op, comm, &bkpap_module->super);
        break;
    default:
        BKPAP_ERROR("alg %d undefined, falling back", alg);
        goto bkpap_ar_abort;
        break;
    }
    BKPAP_CHK_MPI(ret, bkpap_ar_abort);

    BKPAP_OUTPUT("rank: %d COMPLETE BKPAP ALLREDUCE", global_rank);
    return ret;

bkpap_ar_abort:
    return ret;
}
