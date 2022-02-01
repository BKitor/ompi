
#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/op/op.h"

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

static inline int _bk_papaware_rsa_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

    return OPAL_ERR_NOT_IMPLEMENTED;
}

static inline int _bk_papaware_ktree_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int mpi_rank = ompi_comm_rank(comm), mpi_size = ompi_comm_size(comm);
    int k = mca_coll_bkpap_component.allreduce_k_value;
    int64_t arrival_pos = -1;

    BKPAP_PROFILE("rank: %d, arrive_at_ktree", mpi_rank);
    // DELETEME
    char offsets_str[64] = { '\0' };
    _bk_fill_array_str_ld(
        bkpap_module->ss_counter_len, bkpap_module->ss_arrival_arr_offsets, 64, offsets_str);
    BKPAP_OUTPUT("rank %d, offsets %s", mpi_rank, offsets_str);
    // return OPAL_ERR_NOT_IMPLEMENTED;
    // DELETEME

    for (int sync_round = 0; sync_round < bkpap_module->ss_counter_len; sync_round++) {
        uint64_t counter_offset = (sync_round * sizeof(int64_t));
        size_t arrival_arr_offset = (bkpap_module->ss_arrival_arr_offsets[sync_round] * sizeof(int64_t));

        BKPAP_OUTPUT("round: %d, rank: %d, counter_offset: %ld, arrival_arr_offset: %ld", sync_round, mpi_rank, counter_offset, arrival_arr_offset);
        ret = mca_coll_bkpap_arrive_ss(bkpap_module, mpi_rank, counter_offset, arrival_arr_offset, comm, &arrival_pos);
        _BK_CHK_RET(ret, "arrive at inter failed");
        arrival_pos += 1;
        BKPAP_OUTPUT("round: %d, rank: %d, arrive: %ld ", sync_round, mpi_rank, arrival_pos);

        BKPAP_PROFILE("rank: %d, register_at_ss", mpi_rank);

        if (0 == arrival_pos % k) {
            // receiving
            // TODO: this num_reduction logic only works for k == 4, might want to fix at some point
            int num_reductions = (_bk_int_pow(k, (sync_round + 1)) <= mpi_size) ? k - 1 : 1;
            ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_reductions, bkpap_module);
            _BK_CHK_RET(ret, "reduce postbuf failed");
            BKPAP_PROFILE("rank: %d, reduce_pbuf", mpi_rank);
        }
        else {
            // sending
            int send_rank = -1;
            int send_arrival_pos = arrival_pos - (arrival_pos % k);
            int arrival_round_offset = bkpap_module->ss_arrival_arr_offsets[sync_round];
            while (send_rank == -1) {
                BKPAP_OUTPUT("rank: %d, arrival: %ld, getting arrival: %d, offset: %ld", mpi_rank, arrival_pos, send_arrival_pos, arrival_arr_offset);
                ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, arrival_round_offset, bkpap_module, &send_rank);
                _BK_CHK_RET(ret, "get rank of arrival faild");
                // usleep(10);
                // sleep(2);
            }
            BKPAP_PROFILE("rank: %d, get_parent_rank", mpi_rank);

            int slot = ((arrival_pos % k) - 1);

            ret = mca_coll_bkpap_put_postbuf(rbuf, dtype, count, send_rank, slot, comm, bkpap_module);
            _BK_CHK_RET(ret, "write parrent postuf failed");
            BKPAP_PROFILE("rank: %d, send_parent_data", mpi_rank);
            break;
        }
        arrival_pos = -1;
    }

    // this is polling over the network, this is gross
    int tree_root = -1;
    while (-1 == tree_root) {
        int arrival_round_offset = bkpap_module->ss_arrival_arr_offsets[bkpap_module->ss_counter_len - 1];
        ret = mca_coll_bkpap_get_rank_of_arrival(0, arrival_round_offset, bkpap_module, &tree_root);
        _BK_CHK_RET(ret, "get rank of arrival failed");
        // usleep(10);
    }
    BKPAP_PROFILE("rank: %d, get_leader_of_tree", mpi_rank);

    // intranode bcast
    ret = comm->c_coll->coll_bcast(rbuf, count, dtype, tree_root, comm, comm->c_coll->coll_bcast_module);
    _BK_CHK_RET(ret, "singlenode bcast failed");

    BKPAP_PROFILE("rank: %d, finish_bcast", mpi_rank);

    // sm bcast is non-blocking, need to block before leaving coll
    // TODO: desing reset-system that doesn't block 
        // hard-reset by rank 0 or last rank, and  check in arrival that arrival_pos < world_size

    // DELETEME
    if (mpi_rank == 0) {
        int64_t* arrival_arr_tmp = bkpap_module->local_syncstructure->arrival_arr_attr.address;
        int64_t* count_arr_tmp = bkpap_module->local_syncstructure->counter_attr.address;
        char arrival_str[128] = { '\0' };
        _bk_fill_array_str_ld(bkpap_module->ss_arrival_arr_len, arrival_arr_tmp, 128, arrival_str);
        char count_str[128] = { '\0' };
        _bk_fill_array_str_ld(bkpap_module->ss_counter_len, count_arr_tmp, 128, count_str);
        BKPAP_OUTPUT("rank 0 leaving, arirval_arr: %s, count_arr: %s", arrival_str, count_str);
    }
    // DELETEME

    // find way to safely DELETEME
    comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);

    ret = mca_coll_bkpap_leave_ss(bkpap_module, comm, arrival_pos);
    _BK_CHK_RET(ret, "leave inter failed");

    return ret;
}

static inline int _bk_singlenode_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    mca_coll_bkpap_module_t* bkpap_module, int alg) {

    switch (alg) {
    case BKPAP_ALLREDUCE_ALG_RSA:
        return _bk_papaware_rsa_allreduce(
            sbuf, rbuf, count, dtype, op, bkpap_module->intra_comm, bkpap_module);
        break;
    case BKPAP_ALLREDUCE_ALG_KTREE:
        return _bk_papaware_ktree_allreduce(
            sbuf, rbuf, count, dtype, op, bkpap_module->intra_comm, bkpap_module);
        break;
    default:
        BKPAP_ERROR("singlenode allreduce switch defaulted on alg: %d", alg);
        return OMPI_ERROR;
        break;
    }
}

// Intranode reduce, internode-syncstructure, internode, allreduce 
static inline int _bk_multinode_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op,
    mca_coll_bkpap_module_t* bkpap_module, int alg) {
    int ret = OMPI_SUCCESS;
    int intra_rank = ompi_comm_rank(bkpap_module->intra_comm);

    const void* reduce_sbuf = (intra_rank == 0) ? MPI_IN_PLACE : rbuf;
    void* reduce_rbuf = (intra_rank == 0) ? rbuf : NULL;
    ret = bkpap_module->intra_comm->c_coll->coll_reduce(
        reduce_sbuf, reduce_rbuf, count, dtype, op, 0,
        bkpap_module->intra_comm,
        bkpap_module->intra_comm->c_coll->coll_reduce_module);
    _BK_CHK_RET(ret, "intranode reduce failed");

    if (intra_rank == 0) {
        switch (alg) {
        case BKPAP_ALLREDUCE_ALG_KTREE:
            ret = _bk_papaware_ktree_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module->inter_comm, bkpap_module);
            break;
        case BKPAP_ALLREDUCE_ALG_RSA:
            ret = _bk_papaware_rsa_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module->inter_comm, bkpap_module);
            break;

        default:
            BKPAP_ERROR("multinode allreduce switch defaulted on alg: %d", alg);
            ret = OMPI_ERROR;
            break;
        }
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Multi-node pap-aware stage failed");
            return ret;
        }
    }

    ret = bkpap_module->intra_comm->c_coll->coll_bcast(
        rbuf, count, dtype, 0,
        bkpap_module->intra_comm,
        bkpap_module->intra_comm->c_coll->coll_bcast_module
    );
    _BK_CHK_RET(ret, "intra-stage bcast failed");

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
    // if not IN_PLACE, copy sbuf into rbuf
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

    BKPAP_OUTPUT("comm rank %d, intra rank %d, inter rank %d", ompi_comm_rank(comm),
        ompi_comm_rank(bkpap_module->intra_comm), ompi_comm_rank(bkpap_module->inter_comm));

    int is_multinode = intra_wsize < global_wsize;
    struct ompi_communicator_t* ss_comm = (is_multinode) ? bkpap_module->inter_comm : bkpap_module->intra_comm;

    if (OPAL_UNLIKELY((is_multinode && intra_rank == 0 && !bkpap_module->ucp_is_initialized)
        || (!is_multinode && !bkpap_module->ucp_is_initialized))) {
        ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        int num_postbufs = (mca_coll_bkpap_component.allreduce_k_value - 1); // should depend on component.alg
        ret = mca_coll_bkpap_wireup_postbuffs(num_postbufs, bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        int k = mca_coll_bkpap_component.allreduce_k_value;
        size_t counter_arr_len = 0; // log_k(wsize);
        size_t arrival_arr_len = 0; // wsize + wsize/k + wsize/k^2 + wsize/k^3 ...
        for (int i = 1; i < ompi_comm_size(ss_comm); i *= k)
            counter_arr_len++;
        int64_t* arrival_arr_offsets_tmp = calloc(counter_arr_len, sizeof(*arrival_arr_offsets_tmp));

        for (size_t i = 0; i < counter_arr_len; i++) {
            int k_pow_i = 1;
            for (size_t j = 0; j < i; j++)
                k_pow_i *= k;
            arrival_arr_len += (ompi_comm_size(ss_comm) / k_pow_i);
            if ((i + 1) != counter_arr_len)
                arrival_arr_offsets_tmp[i + 1] = arrival_arr_offsets_tmp[i] + (ompi_comm_size(ss_comm) / k_pow_i);
        }

        bkpap_module->ss_counter_len = counter_arr_len;
        bkpap_module->ss_arrival_arr_len = arrival_arr_len;
        bkpap_module->ss_arrival_arr_offsets = arrival_arr_offsets_tmp;
        arrival_arr_offsets_tmp = NULL;

        ret = mca_coll_bkpap_wireup_syncstructure(counter_arr_len, arrival_arr_len, bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        if (0 == ompi_comm_rank(ss_comm)) {
            int64_t* arrival_arr_tmp = bkpap_module->local_syncstructure->arrival_arr_attr.address;
            int64_t* count_arr_tmp = bkpap_module->local_syncstructure->counter_attr.address;
            char arrival_str[128] = { '\0' };
            _bk_fill_array_str_ld(bkpap_module->ss_arrival_arr_len, arrival_arr_tmp, 128, arrival_str);
            char count_str[128] = { '\0' };
            _bk_fill_array_str_ld(bkpap_module->ss_counter_len, count_arr_tmp, 128, count_str);
            BKPAP_OUTPUT("SS initalized at rank 0, arirval_arr: %s, count_arr: %s", arrival_str, count_str);
        }

        bkpap_module->ucp_is_initialized = 1;
    }


    if (is_multinode) {
        ret = _bk_multinode_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module, alg);
        if (ret != OMPI_SUCCESS) {
            BKPAP_ERROR("multi-node failed, falling back");
            goto bkpap_ar_fallback;
        }
    }
    else {
        ret = _bk_singlenode_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module, alg);
        if (ret != OMPI_SUCCESS) {
            BKPAP_ERROR("single-node failed, falling back");
            goto bkpap_ar_fallback;
        }
    }


    BKPAP_OUTPUT("rank %d returning first val %d BKPAP ALLREDUCE SUCCESSFULL", global_rank, ((int*)rbuf)[0]);
    return OMPI_SUCCESS;

bkpap_ar_fallback:

    return bkpap_module->fallback_allreduce(
        sbuf, rbuf, count, dtype, op, comm,
        bkpap_module->fallback_allreduce_module);
}