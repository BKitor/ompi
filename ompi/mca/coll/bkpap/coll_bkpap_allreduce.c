#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/op/op.h"

#include "opal/mca/common/cuda/common_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define _BK_CHK_RET(_ret, _msg) if(OPAL_UNLIKELY(OMPI_SUCCESS != _ret)){BKPAP_ERROR(_msg); return _ret;}

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

static inline int _bk_papaware_rsa_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module) {
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static inline int bk_request_wait_all(ompi_request_t** request_arr, int req_arr_len) {
    int tmp_is_completed;
    ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    while (!tmp_is_completed) {
        ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
        ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    }
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

    int seg_count = count;
    size_t type_size;
    ptrdiff_t type_extent, type_lb;
    ompi_datatype_get_extent(dtype, &type_lb, &type_extent);
    ompi_datatype_type_size(dtype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int num_segments = (count + seg_count - 1) / seg_count;
    size_t real_seg_size = (ptrdiff_t)seg_count * type_extent;

    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("ktree_pipeline_arrive", inter_rank);

    void* intra_reduce_sbuf = (0 == intra_rank) ? MPI_IN_PLACE : rbuf;
    void* intra_reduce_rbuf = (0 == intra_rank) ? rbuf : NULL;

    switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
    case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
        ret = intra_comm->c_coll->coll_reduce(
            intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0,
            intra_comm,
            intra_comm->c_coll->coll_reduce_module);
        break;
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
        ret = mca_coll_bkpap_reduce_intra_inplace_binomial(rbuf, count, dtype, op, 0, intra_comm, intra_comm->c_coll->coll_reduce_module, 0, 0);
        break;
    default:
        BKPAP_ERROR("Bad memory type, intra-node reduce failed");
        return OMPI_ERROR;
        break;
    }
    _BK_CHK_RET(ret, "intra-node reduce failed");


    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("finish_intra_reduce", inter_rank);

    if (OPAL_LIKELY(is_inter)) {
        BKPAP_OUTPUT("KTREE_PIPELINE_ARRIVE, rank: %d, count: %d, seg_size: %ld, dtype_size: %ld, num_segments:%d, inter_size: %d, intra_size: %d",
            inter_rank, count, seg_size, type_size, num_segments, inter_size, intra_size);
        ompi_request_t* inter_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* intra_bcast_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
        ompi_request_t* tmp_bcast_wait_arr[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };

        uint8_t* seg_buf = rbuf;
        int bcast_count = seg_count, prev_bcast_count = 0;
        int inter_bcast_root = -1;

        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            BKPAP_PROFILE("start_new_segment", inter_rank);
            int64_t arrival_pos = -1;
            int phase_selector = (seg_index % 2);
            mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(bkpap_module->remote_syncstructure[phase_selector]);
            if ((num_segments - 1) == seg_index) { // if last segment, and allignment doesn't line up with seg-count
                bcast_count = count - (seg_index * seg_count);
            }

            BKPAP_OUTPUT("START_SEG: seg_index: %d, num_segments: %d, rank: %d, phase_selector: %d", seg_index, num_segments, inter_rank, phase_selector);

            // Wait Inter and Intra ibcasts
            tmp_bcast_wait_arr[0] = (intra_bcast_reqs[phase_selector]);
            tmp_bcast_wait_arr[1] = (inter_bcast_reqs[phase_selector]);
            bk_request_wait_all(tmp_bcast_wait_arr, 2);

            BKPAP_PROFILE("leave_new_seg_wait", inter_rank);

            BKPAP_OUTPUT("LEAVE_WAIT_ALL: seg_index: %d, rank: %d, phase_selector: %d", seg_index, inter_rank, phase_selector);

            if (seg_index > 1 && OPAL_LIKELY(inter_size != intra_size)) { // Launch Intra-bcast
                BKPAP_OUTPUT("STARTING_INTRA_IBCAST: seg_index: %d, rank: %d", seg_index, inter_rank);
                uint8_t* intra_buf = rbuf;
                intra_buf += (seg_index - 2) * real_seg_size;
                intra_comm->c_coll->coll_ibcast(
                    intra_buf, prev_bcast_count, dtype, 0, intra_comm,
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
                    _BK_CHK_RET(ret, "arrive at inter failed");
                    arrival_pos += 1;

                    BKPAP_OUTPUT("ARRIVED_AT_SS: seg_index: %d, rank: %d, arrival_pos: %ld", seg_index, inter_rank, arrival_pos);
                    while (-1 == inter_bcast_root) {
                        int arrival_round_offset = 0;
                        ret = mca_coll_bkpap_get_rank_of_arrival(0, arrival_round_offset, remote_ss_tmp, bkpap_module, &inter_bcast_root);
                        _BK_CHK_RET(ret, "get rank of arrival failed");
                    }
                    BKPAP_OUTPUT("GOT_TREE_ROOT: seg_index: %d, rank: %d, arrive: %ld, inter_bcast_root:%d ", seg_index, inter_rank, arrival_pos, inter_bcast_root);
                    BKPAP_PROFILE("synced_at_ss", inter_rank);
                }

                sync_mask *= k;

                if (0 == arrival_pos % sync_mask) { // if-parent reduce
                    // TODO: this num_reduction logic only works for (k == 4) and (wsize is a power of 2),
                    // TODO: might want to fix at some point
                    int num_reductions = (sync_mask > inter_size) ? 1 : k - 1;
                    BKPAP_OUTPUT("DBG sync_mask: %d, k: %d, inter_size: %d, num_reductions: %d", sync_mask, k, inter_size, num_reductions);
                    BKPAP_OUTPUT("START_PBUF_REDUCE: seg_index: %d, rank: %d, num_reductions: %d, count: %d", seg_index, inter_rank, num_reductions, bcast_count);
                    BKPAP_PROFILE("start_postbuf_reduce", inter_rank);
                    ret = mca_coll_bkpap_reduce_postbufs(seg_buf, dtype, bcast_count, op, num_reductions, inter_comm, bkpap_module);
                    BKPAP_PROFILE("leave_postbuf_reduce", inter_rank);
                    _BK_CHK_RET(ret, "reduce postbuf failed");
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
                        _BK_CHK_RET(ret, "get rank of arrival faild");
                    }
                    BKPAP_PROFILE("got_parent_rank", inter_rank);

                    int slot = ((arrival_pos / (sync_mask / k)) % k) - 1;

                    BKPAP_OUTPUT("SEND_PARENT: seg_index: %d, rank: %d, arrival: %ld, send_rank: %d, send_arrival: %d, slot: %d, sync_mask: %d",
                        seg_index, inter_rank, arrival_pos, send_rank, send_arrival_pos, slot, sync_mask);
                    ret = mca_coll_bkpap_put_postbuf(seg_buf, dtype, bcast_count, send_rank, slot, inter_comm, bkpap_module);
                    _BK_CHK_RET(ret, "write parrent postuf failed");
                    BKPAP_PROFILE("sent_parent_rank", inter_rank);
                    break;
                } // else-child send parent
            } // pap-aware loop

            if (inter_rank == inter_bcast_root) {
                BKPAP_OUTPUT("RESET_SS: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
                ret = mca_coll_bkpap_reset_remote_ss(remote_ss_tmp, inter_comm, bkpap_module);
                _BK_CHK_RET(ret, "reset_remote_ss failed");
                BKPAP_PROFILE("reset_remote_ss", inter_rank);
            }

            BKPAP_OUTPUT("STARTING_INTER_IBCAST: seg_index: %d, rank: %d, arrival: %ld", seg_index, inter_rank, arrival_pos);
            inter_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, inter_bcast_root, inter_comm,
                &(inter_bcast_reqs[seg_index % 2]), inter_comm->c_coll->coll_ibcast_module);
            inter_bcast_root = -1;

            prev_bcast_count = bcast_count;
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
            bk_request_wait_all(tmp_bcast_wait_arr, 2);
            BKPAP_PROFILE("leave_cleanup_wait", inter_rank);
            BKPAP_OUTPUT("FINISHED_CLEANUP_WAIT: rank: %d, cleanup_idx: %d", inter_rank, cleanup_index);

            seg_buf = rbuf;
            seg_buf += (real_seg_size * (num_segments - (2 - cleanup_index)));
            bcast_count = (0 == cleanup_index) ? seg_count : count - ((num_segments - 1) * seg_count);

            intra_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, 0, intra_comm,
                &intra_bcast_reqs[phase_selector], intra_comm->c_coll->coll_ibcast_module
            );

            seg_buf += real_seg_size;
        }

        bk_request_wait_all(intra_bcast_reqs, 2);
        BKPAP_PROFILE("final_cleanup_wait", inter_rank);
        BKPAP_OUTPUT("CLEANUP_DONE: rank: %d", inter_rank);
    }
    else {
        int bcast_count = seg_count;
        ompi_request_t* bcast_req = (void*)OMPI_REQUEST_NULL;
        uint8_t* seg_buf = rbuf;
        for (int seg_index = 0; seg_index < num_segments; seg_index++) {
            if ((num_segments - 1) == seg_index) {
                bcast_count = count - (seg_index * seg_count);
            }
            BKPAP_OUTPUT("in intra, rank: %d, seg_index: %d, num_segments: %d, bcast_count: %d", intra_rank, seg_index, num_segments, count);

            intra_comm->c_coll->coll_ibcast(
                seg_buf, bcast_count, dtype, 0, intra_comm, &bcast_req,
                intra_comm->c_coll->coll_ibcast_module);
            ompi_request_wait(&bcast_req, MPI_STATUS_IGNORE);
            seg_buf += real_seg_size;
        }
    }


    if (OPAL_LIKELY(is_inter))BKPAP_PROFILE("ktree_pipeline_leave", inter_rank);

    return ret;
}

static inline int _bk_papaware_ktree_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int inter_rank = ompi_comm_rank(inter_comm), inter_size = ompi_comm_size(inter_comm);
    int intra_rank = ompi_comm_rank(intra_comm), intra_size = ompi_comm_size(intra_comm);
    int is_inter = (0 == intra_rank);
    int k = mca_coll_bkpap_component.allreduce_k_value;
    int64_t arrival_pos = -1;

    BKPAP_OUTPUT("rank (%d, %d) Arrive at ktree, comm: '%s', sbuf: %p rbuf: %p", inter_rank, intra_rank, intra_comm->c_name, sbuf, rbuf);
    if (is_inter)BKPAP_PROFILE("arrive_at_ktree", inter_rank);

    void* intra_reduce_sbuf = (0 == intra_rank) ? MPI_IN_PLACE : rbuf;
    void* intra_reduce_rbuf = (0 == intra_rank) ? rbuf : NULL;

    switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
    case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
        ret = intra_comm->c_coll->coll_reduce(
            intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0,
            intra_comm,
            intra_comm->c_coll->coll_reduce_module);
        break;
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
        ret = mca_coll_bkpap_reduce_intra_inplace_binomial(rbuf, count, dtype, op, 0, intra_comm, intra_comm->c_coll->coll_reduce_module, 0, 0);
        break;
    default:
        BKPAP_ERROR("Bad memory type, intra-node reduce failed");
        return OMPI_ERROR;
        break;
    }
    // _BK_CHK_RET(ret, "intra-node reduce failed");
    if (OMPI_SUCCESS != ret) {
        BKPAP_ERROR("Intra-reduce failed, rank (%d, %d) returned %d (%s) exiting", inter_rank, intra_rank, ret, opal_strerror(ret));
        return ret;
    }

    BKPAP_PROFILE("finish_intra_reduce", inter_rank);


    if (is_inter) {

#if OPAL_ENABLE_DEBUG
        int rbuf_is_cuda = opal_cuda_check_one_buf(rbuf, NULL);
        int pbuf_is_cuda = opal_cuda_check_one_buf(bkpap_module->local_pbuffs.postbuf_attrs.address, NULL);
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
            _BK_CHK_RET(ret, "arrive at inter failed");
            arrival_pos += 1;
            BKPAP_OUTPUT("round: %d, rank: %d, arrive: %ld ", sync_round, inter_rank, arrival_pos);

            BKPAP_PROFILE("register_at_ss", inter_rank);

            if (0 == arrival_pos % k) {
                // receiving
                // TODO: this num_reduction logic only works for (k == 4) and (wsize is a power of 2),
                // TODO: might want to fix at some point
                int num_reductions = (_bk_int_pow(k, (sync_round + 1)) <= inter_size) ? k - 1 : 1;
                ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_reductions, inter_comm, bkpap_module);
                _BK_CHK_RET(ret, "reduce postbuf failed");
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
                    _BK_CHK_RET(ret, "get rank of arrival faild");
                }
                BKPAP_PROFILE("get_parent_rank", inter_rank);

                int slot = ((arrival_pos % k) - 1);

                ret = mca_coll_bkpap_put_postbuf(rbuf, dtype, count, send_rank, slot, inter_comm, bkpap_module);
                _BK_CHK_RET(ret, "write parrent postuf failed");
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
            _BK_CHK_RET(ret, "get rank of arrival failed");
            // usleep(10);
        }
        BKPAP_PROFILE("get_leader_of_tree", inter_rank);

        // internode bcast
        ret = inter_comm->c_coll->coll_bcast(rbuf, count, dtype, tree_root, inter_comm, inter_comm->c_coll->coll_bcast_module);
        _BK_CHK_RET(ret, "inter-stage bcast failed");
        BKPAP_PROFILE("finish_inter_bcast", inter_rank);

        // TODO: desing reset-system that doesn't block 
        // hard-reset by rank 0 or last rank, and  check in arrival that arrival_pos < world_size
        inter_comm->c_coll->coll_barrier(inter_comm, inter_comm->c_coll->coll_barrier_module);
        if (is_inter && inter_rank == tree_root) {
            ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
            _BK_CHK_RET(ret, "reset_remote_ss failed");
            BKPAP_PROFILE("reset_remote_ss", inter_rank);
        }

    }

    ret = intra_comm->c_coll->coll_bcast(
        rbuf, count, dtype, 0,
        intra_comm,
        intra_comm->c_coll->coll_bcast_module);
    _BK_CHK_RET(ret, "intra-stage bcast failed");
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


    return ret;
}
#undef _BK_CHK_RET

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
        goto bkpap_ar_fallback;
    }

    if (!ompi_op_is_commute(op)) {
        BKPAP_ERROR("Commutative operation, going to fallback");
        goto bkpap_ar_fallback;
    }

    ompi_datatype_type_size(dtype, &dsize);
    total_dsize = dsize * (ptrdiff_t)count;
    if (total_dsize > mca_coll_bkpap_component.postbuff_size) {
        BKPAP_ERROR("Message size is bigger than postbuf, falling back");
        goto bkpap_ar_fallback;
    }

    // if IN_PLACE, rbuf is local contents, and will be used as local buffer 
    // if not IN_PLACE, copy sbuf into rbuf and act as if IN_PLACE
    if (MPI_IN_PLACE != sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, rbuf, (char*)sbuf);
        if (ret != OMPI_SUCCESS) {
            BKPAP_ERROR("Not in place memcpy failed, falling back");
            goto bkpap_ar_fallback;
        }
        sbuf = MPI_IN_PLACE;
    }

    // Set up Intra/Inter comms
    if (OPAL_UNLIKELY(NULL == bkpap_module->intra_comm || NULL == bkpap_module->inter_comm)) {
        ret = mca_coll_bkpap_wireup_hier_comms(bkpap_module, comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("inter/intra communicator creation failed");
            goto bkpap_ar_fallback;
        }
    }

    int global_wsize = ompi_comm_size(comm);
    int global_rank = ompi_comm_rank(comm);
    int intra_wsize = ompi_comm_size(bkpap_module->intra_comm);
    int intra_rank = ompi_comm_rank(bkpap_module->intra_comm);

    int is_multinode = intra_wsize < global_wsize;
    struct ompi_communicator_t* ss_inter_comm = (is_multinode) ? bkpap_module->inter_comm : comm;
    struct ompi_communicator_t* ss_intra_comm = (is_multinode) ? bkpap_module->intra_comm : &ompi_mpi_comm_self.comm;

    BKPAP_OUTPUT("comm rank: %d, intra rank: %d, inter rank: %d, is_multinode: %d, alg %d", ompi_comm_rank(comm),
        ompi_comm_rank(ss_intra_comm), ompi_comm_rank(ss_inter_comm), is_multinode, alg);


    if (OPAL_UNLIKELY((is_multinode && intra_rank == 0 && !bkpap_module->ucp_is_initialized)
        || (!is_multinode && !bkpap_module->ucp_is_initialized))) {


        if (OPAL_UNLIKELY(NULL == mca_coll_bkpap_component.ucp_context)) {
            ret = mca_coll_bkpap_init_ucx(mca_coll_bkpap_component.enable_threads);
            if (OMPI_SUCCESS != ret) {
                BKPAP_ERROR("UCX Initialization Failed");
                goto bkpap_ar_fallback;
            }
            BKPAP_OUTPUT("UCX Initialization SUCCESS");
        }


        ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, ss_inter_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        int num_postbufs = (mca_coll_bkpap_component.allreduce_k_value - 1); // should depend on component.alg
        ret = mca_coll_bkpap_wireup_postbuffs(num_postbufs, bkpap_module, ss_inter_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }


        int k = mca_coll_bkpap_component.allreduce_k_value;
        int num_syncstructures = 1;
        size_t counter_arr_len = 0; // log_k(wsize);
        size_t arrival_arr_len = 0; // wsize + wsize/k + wsize/k^2 + wsize/k^3 ...
        int64_t* arrival_arr_offsets_tmp = NULL;
        switch (alg) {
        case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
            num_syncstructures = 2;
            counter_arr_len = 1;
            arrival_arr_len = ompi_comm_size(ss_inter_comm);
            arrival_arr_offsets_tmp = NULL;
            break;
        case BKPAP_ALLREDUCE_ALG_KTREE:
            for (int i = 1; i < ompi_comm_size(ss_inter_comm); i *= k)
                counter_arr_len++;
            arrival_arr_offsets_tmp = calloc(counter_arr_len, sizeof(*arrival_arr_offsets_tmp));

            for (size_t i = 0; i < counter_arr_len; i++) {
                int k_pow_i = 1;
                for (size_t j = 0; j < i; j++)
                    k_pow_i *= k;
                arrival_arr_len += (ompi_comm_size(ss_inter_comm) / k_pow_i);
                if ((i + 1) != counter_arr_len)
                    arrival_arr_offsets_tmp[i + 1] = arrival_arr_offsets_tmp[i] + (ompi_comm_size(ss_inter_comm) / k_pow_i);
            }
            break;
        case BKPAP_ALLREDUCE_ALG_RSA:
            BKPAP_ERROR("bkpap RSA alg not implemented, aborting");
            return OPAL_ERR_NOT_IMPLEMENTED;
            break;
        default:
            BKPAP_ERROR("Bad algorithms specified, failed to setup syncstructure");
            return OMPI_ERROR;
            break;
        }

        ret = mca_coll_bkpap_wireup_syncstructure(counter_arr_len, arrival_arr_len, num_syncstructures, bkpap_module, ss_inter_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }
        for (int i = 0; i < num_syncstructures; i++) {
            bkpap_module->remote_syncstructure[i].ss_counter_len = counter_arr_len;
            bkpap_module->remote_syncstructure[i].ss_arrival_arr_len = arrival_arr_len;
            bkpap_module->remote_syncstructure[i].ss_arrival_arr_offsets = arrival_arr_offsets_tmp;
        }
        arrival_arr_offsets_tmp = NULL;

#if OPAL_ENABLE_DEBUG
        if (0 == ompi_comm_rank(ss_inter_comm)) {
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
        _bk_papaware_ktree_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
        _bk_papaware_ktree_allreduce_pipelined(sbuf, rbuf, count, dtype, op, mca_coll_bkpap_component.pipeline_segment_size, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_RSA:
        _bk_papaware_rsa_allreduce(sbuf, rbuf, count, dtype, op, ss_intra_comm, ss_inter_comm, bkpap_module);
        break;

    default:
        BKPAP_ERROR("alg %d undefined, falling back", alg);
        goto bkpap_ar_fallback;
        break;
    }


    BKPAP_OUTPUT("rank: %d returning BKPAP ALLREDUCE SUCCESSFULL", global_rank);
    return OMPI_SUCCESS;

bkpap_ar_fallback:

    return bkpap_module->fallback_allreduce(
        sbuf, rbuf, count, dtype, op, comm,
        bkpap_module->fallback_allreduce_module);
}
