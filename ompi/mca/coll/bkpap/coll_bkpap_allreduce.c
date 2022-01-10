#include "math.h"

#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/op/op.h"

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	int ret = OMPI_SUCCESS;
	int k = mca_coll_bkpap_component.allreduce_k_value;
	int64_t arrival_pos;

	if (!ompi_op_is_commute(op)){
		BKPAP_ERROR("Commutative operation, going to fallback");
		goto bkpap_ar_fallback;
	}

	// if IN_PLACE, rbuf is local contents, and will be used as local buffer 
	// if not IN_PLACE, copy sbuf into rbuf
	if (MPI_IN_PLACE != sbuf){
		ret = ompi_datatype_copy_content_same_ddt(dtype, count, rbuf, (char*)sbuf);
		if(ret != OMPI_SUCCESS){
			BKPAP_ERROR("Not in place memcpy failed, falling back");
			goto bkpap_ar_fallback;
		}
	}

	// global rank 0 has the dstructure
	// rank 0 of each inter-communicator is a leader

	// check if remote memory is setup
	// 	check if rkey/addr exists on module
	// 	pap_datastructre is setup on rank 0
	// 	rkeys for remote leaders are available

	BKPAP_OUTPUT("Allreduce Called rank %d", ompi_comm_rank(comm));

	if (OPAL_UNLIKELY(NULL == bkpap_module->ucp_ep_arr)) {
		ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, comm);
		comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_pbuffs.dbell_addr_arr || NULL == bkpap_module->remote_pbuffs.dbell_addr_arr
		|| NULL == bkpap_module->remote_pbuffs.buffer_addr_arr || NULL == bkpap_module->remote_pbuffs.buffer_addr_arr)) {
		ret = mca_coll_bkpap_wireup_postbuffs(bkpap_module, comm);
		comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_syncstructure_arrival_arr_rkey || NULL == bkpap_module->remote_syncstructure_counter_rkey
		|| 0 == bkpap_module->remote_syncstructure_counter_addr || 0 == bkpap_module->remote_syncstructure_arrival_arr_addr)) {
		ret = mca_coll_bkpap_wireup_syncstructure(bkpap_module, comm);
		comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	// check if intrandode communicator exists

	if (OPAL_UNLIKELY(NULL == bkpap_module->intra_comm || NULL == bkpap_module->inter_comm)) {
		ret = mca_coll_bkpap_wireup_hier_comms(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("inter/intra communicator creation failed");
			goto bkpap_ar_fallback;
		}
	}

	int global_rank = ompi_comm_rank(comm);
	int global_wsize = ompi_comm_size(comm);
	int intra_rank = ompi_comm_rank(bkpap_module->intra_comm);
	// inter_wsize = ompi_comm_size(bkpap_module->inter_comm);
	// inter_rank = ompi_comm_rank(bkpap_module->inter_comm);
	#warning inter_wsize / inter_rank are set to global, fix for multi - node test
		int inter_wsize = global_wsize; // TODO: fix when multi-node 
	int inter_rank = global_rank; // TODO: fix when multi-node


	// bkpap_module->intra_comm->c_coll->coll_allreduce(
	// 		sbuf, rbuf, count, dtype, op, bkpap_module->intra_comm,
	// 		bkpap_module->intra_comm->c_coll->coll_allreduce_module);

	// if (low_rank == 0){

	// }


	ret = mca_coll_bkpap_arrive_at_inter(bkpap_module, comm, &arrival_pos);
	arrival_pos += 1;
	BKPAP_OUTPUT("rank %d arrive %ld start val: %x", inter_rank, arrival_pos, ((int*)rbuf)[0]);
	// comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);

	int tmp_k = k;
	while (arrival_pos % tmp_k == 0) {
		int num_buffers = (k - 1);
		// BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
		mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, comm, bkpap_module);

		tmp_k *= k;
		if (tmp_k > inter_wsize) break;
	}

	if (arrival_pos == 0) {
		if ((tmp_k / k) < inter_wsize) {  // condition to do final recieve if not power of k
			int num_buffers = 1; // TODO: fix to that it will work for different K values, this only works for k=4
			// BKPAP_OUTPUT("rank %d arrive %ld recive with num_buffers %d and tmp_k %d", global_rank, arrival_pos, num_buffers, tmp_k);
			mca_coll_bkpap_reduce_postbufs(rbuf, dtype, count, op, num_buffers, comm, bkpap_module);
		}
	}
	else {
		int send_arrival_pos;
		int send_hrank = -1;
		send_arrival_pos = arrival_pos - (arrival_pos % tmp_k);
		while (-1 == send_hrank) {
			ret = mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, bkpap_module, comm, &send_hrank);
			// TODO: Errorchecking :(
		}

		// BKPAP_OUTPUT("rank %d arrive %ld send to pos %d (rank %d)", global_rank, arrival_pos, send_arrival_pos, send_hrank);
		ret = mca_coll_bkpap_write_parent_postbuf(rbuf, dtype, count, arrival_pos, tmp_k, send_hrank, comm, bkpap_module);
	}

	// internode bcast
	int root = -1;
	while(-1 == root)
		ret = mca_coll_bkpap_get_rank_of_arrival(0, bkpap_module, comm, &root);
		// TODO: Errorchecking :(


	#warning switch from global-comm to inter-comm 
	// ret = bkpap_module->inter_comm->c_coll->coll_bcast(
	// 	rbuf, count, dtype, root, 
	// 	bkpap_module->inter_comm,
	// 	bkpap_module->inter_comm->c_coll->coll_bcast_module
	// );
	ret = comm->c_coll->coll_bcast(
		rbuf, count, dtype, root, 
		comm,
		comm->c_coll->coll_bcast_module
	);
	if(ret != OMPI_SUCCESS){
		BKPAP_ERROR("'inter-stage' bcast failed");
		return ret;
	}
	// intranode bcast
	// bkpap_module->intra_comm->c_coll->coll_bcast(
	// 	rbuf, count, dtype, 0, 
	// 	bkpap_module->intra_comm,
	// 	bkpap_module->intra_comm->c_coll->coll_bcast_module
	// );

	comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
	if (global_rank == 0) {
		int64_t* tmp = (int64_t*)bkpap_module->local_syncstructure->counter_attr.address;
		*tmp = -1;
		tmp = (int64_t*)bkpap_module->local_syncstructure->arrival_arr_attr.address;
		for (int i = 0; i < global_wsize; i++)
			tmp[i] = -1;
	}
	
	BKPAP_OUTPUT("rank %d returning first val %x BKPAP ALLREDUCE SUCCESSFULL", global_rank, ((int*)rbuf)[0]);
	return OMPI_SUCCESS;

bkpap_ar_fallback:

	return bkpap_module->fallback_allreduce(
		sbuf, rbuf, count, dtype, op, comm,
		bkpap_module->fallback_allreduce_module);
}