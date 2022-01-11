#include "math.h"

#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/op/op.h"

#define _BK_CHK_RET(_ret, _msg) if(OPAL_UNLIKELY(OMPI_SUCCESS != _ret)){BKPAP_ERROR(_msg); return _ret;}

// Just the Suncstructure stuff
static inline int _bk_singlenode_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int64_t arrival_pos = -1;
    struct ompi_communicator_t* comm = bkpap_module->intra_comm;
    int rank = ompi_comm_rank(comm), size = ompi_comm_size(comm);
    int k = mca_coll_bkpap_component.allreduce_k_value;


    ret = mca_coll_bkpap_arrive_at_inter(bkpap_module, rank, &arrival_pos);
    _BK_CHK_RET(ret, "arrive at inter failed");
    arrival_pos += 1;
    BKPAP_OUTPUT("rank %d arrive %ld start val: %x", rank, arrival_pos, ((int*)rbuf)[0]);
    // comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);

    int tmp_k = k;
    while (arrival_pos % tmp_k == 0) {
        int num_buffers = (k - 1);
        // BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
        ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, bkpap_module);
        _BK_CHK_RET(ret, "reduce postbuf failed");

        tmp_k *= k;
        if (tmp_k > size) break;
    }

    if (arrival_pos == 0) {
        if ((tmp_k / k) < size) {  // condition to do final recieve if not power of k
            int num_buffers = 1; // TODO: fix to that it will work for different K values, this only works for k=4
            // BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
            ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, bkpap_module);
            _BK_CHK_RET(ret, "reduce postbuf failed");
        }
    }
    else {
        int send_arrival_pos = arrival_pos - (arrival_pos % tmp_k);
        int send_hrank = -1;
        while (-1 == send_hrank) {
            ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, bkpap_module, &send_hrank);
            _BK_CHK_RET(ret, "get rank of arrival failed");
        }

        // BKPAP_OUTPUT("rank %d arrive %ld send to pos %d (rank %d)", global_rank, arrival_pos, send_arrival_pos, send_hrank);
        ret = mca_coll_bkpap_write_parent_postbuf(rbuf, dtype, count, arrival_pos, tmp_k, send_hrank, comm, bkpap_module);
        _BK_CHK_RET(ret, "write parent postbuf failed");
    }

    // internode bcast
    int root = -1;
    while (-1 == root) {
        ret = mca_coll_bkpap_get_rank_of_arrival(0, bkpap_module, &root);
        _BK_CHK_RET(ret, "get rank of arrival failed");
    }

    // intranode bcast
    ret = comm->c_coll->coll_bcast(rbuf, count, dtype, root, comm, comm->c_coll->coll_bcast_module);
    _BK_CHK_RET(ret, "singlenode bcast failed");

    return ret;
}

// Intranode reduce, internode-syncstructure, internode, allreduce 
static inline int _bk_multinode_allreduce(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    int64_t arrival_pos = -1;
    int inter_rank = ompi_comm_rank(bkpap_module->inter_comm), intra_rank = ompi_comm_rank(bkpap_module->intra_comm);
    int inter_wsize = ompi_comm_size(bkpap_module->inter_comm);
    int k = mca_coll_bkpap_component.allreduce_k_value;

    ret = bkpap_module->intra_comm->c_coll->coll_reduce(
        sbuf, rbuf, count, dtype, op, 0,
        bkpap_module->intra_comm,
        bkpap_module->intra_comm->c_coll->coll_reduce_module);
    _BK_CHK_RET(ret, "intranode reduce failed");

    if (intra_rank == 0) {
        ret = mca_coll_bkpap_arrive_at_inter(bkpap_module, inter_rank, &arrival_pos);
        _BK_CHK_RET(ret, "arrive at inter falied");
        arrival_pos += 1;
        BKPAP_OUTPUT("rank %d arrive %ld start val: %x", inter_rank, arrival_pos, ((int*)rbuf)[0]);

        int tmp_k = k;
        while (arrival_pos % tmp_k == 0) {
            int num_buffers = (k - 1);
            // BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
            ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, bkpap_module);
            _BK_CHK_RET(ret, "reduce postbuf failed");

            tmp_k *= k;
            if (tmp_k > inter_wsize) break;
        }

        if (arrival_pos == 0) {
            if ((tmp_k / k) < inter_wsize) {  // condition to do final recieve if not power of k
                int num_buffers = 1; // TODO: fix to that it will work for different K values, this only works for k=4
                // BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
                ret = mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, bkpap_module);
                _BK_CHK_RET(ret, "reduce postbuf failed");
            }
        }
        else {
            int send_arrival_pos;
            int send_hrank = -1;
            send_arrival_pos = arrival_pos - (arrival_pos % tmp_k);
            while (-1 == send_hrank) {
                ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, bkpap_module, &send_hrank);
                _BK_CHK_RET(ret, "get rank of arrival failed");
            }

            // BKPAP_OUTPUT("rank %d arrive %ld send to pos %d (rank %d)", global_rank, arrival_pos, send_arrival_pos, send_hrank);
            ret = mca_coll_bkpap_write_parent_postbuf(rbuf, dtype, count, arrival_pos, tmp_k, send_hrank, bkpap_module->inter_comm, bkpap_module);
            _BK_CHK_RET(ret, "write parent postbuf failed");
        }

        // internode bcast
        int root = -1;
        while (-1 == root) {
            ret = mca_coll_bkpap_get_rank_of_arrival(0, bkpap_module, &root);
            _BK_CHK_RET(ret, "get rank of arrival failed");
        }

        ret = bkpap_module->inter_comm->c_coll->coll_bcast(
            rbuf, count, dtype, root,
            bkpap_module->inter_comm,
            bkpap_module->inter_comm->c_coll->coll_bcast_module);
        _BK_CHK_RET(ret, "inter-stage bcast failed");
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
    int ret = OMPI_SUCCESS;
    // int k = mca_coll_bkpap_component.allreduce_k_value;
    // int64_t arrival_pos;

    if (!ompi_op_is_commute(op)) {
        BKPAP_ERROR("Commutative operation, going to fallback");
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
    int inter_wsize = ompi_comm_rank(bkpap_module->inter_comm);

    int is_multinode = intra_wsize < global_wsize;
    struct ompi_communicator_t* ss_comm = (is_multinode) ? bkpap_module->inter_comm : comm;
    int ss_wsize = (is_multinode) ? inter_wsize : global_wsize;

    // if (intra_rank == 0 && !bkpap_module->ucp_is_initialized) {
    if ((is_multinode && intra_rank == 0 && !bkpap_module->ucp_is_initialized)
        || (!is_multinode && !bkpap_module->ucp_is_initialized)) {
        ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        ret = mca_coll_bkpap_wireup_postbuffs(bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }

        ret = mca_coll_bkpap_wireup_syncstructure(bkpap_module, ss_comm);
        if (OMPI_SUCCESS != ret) {
            BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
            goto bkpap_ar_fallback;
        }
        bkpap_module->ucp_is_initialized = 1;
        // ss_comm->c_coll->coll_barrier(ss_comm, ss_comm->c_coll->coll_barrier_module);
    }


    if (is_multinode) {
        ret = _bk_multinode_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module);
        if(ret != OMPI_SUCCESS){
            BKPAP_ERROR("multi-node failed, falling back");
            goto bkpap_ar_fallback;
        }
    }
    else {
        ret = _bk_singlenode_allreduce(sbuf, rbuf, count, dtype, op, bkpap_module);
        if(ret != OMPI_SUCCESS){
            BKPAP_ERROR("single-node failed, falling back");
            goto bkpap_ar_fallback;
        }
    }

    // reset syncstructure
    if (0 == ompi_comm_rank(ss_comm)) {
        int64_t* tmp = (int64_t*)bkpap_module->local_syncstructure->counter_attr.address;
        *tmp = -1;
        tmp = (int64_t*)bkpap_module->local_syncstructure->arrival_arr_attr.address;
        for (int i = 0; i < ss_wsize; i++)
            tmp[i] = -1;
    }

    BKPAP_OUTPUT("rank %d returning first val %x BKPAP ALLREDUCE SUCCESSFULL", global_rank, ((int*)rbuf)[0]);
    return OMPI_SUCCESS;

bkpap_ar_fallback:

    return bkpap_module->fallback_allreduce(
        sbuf, rbuf, count, dtype, op, comm,
        bkpap_module->fallback_allreduce_module);
}