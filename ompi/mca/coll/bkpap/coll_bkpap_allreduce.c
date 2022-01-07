#include "math.h"

#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"


int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	int ret = OMPI_SUCCESS;
	int k = mca_coll_bkpap_component.allreduce_k_value;
	int64_t arrival_pos;
	// if commutative, goto(fallback)


	// global rank 0 has the dstructure
	// rank 0 of each inter-communicator is a leader

	// check if remote memory is setup
	// 	check if rkey/addr exists on module
	// 	pap_datastructre is setup on rank 0
	// 	rkeys for remote leaders are available

	BKPAP_OUTPUT("Allreduce Called rank %d", ompi_comm_rank(comm));

	if (OPAL_UNLIKELY(NULL == bkpap_module->ucp_ep_arr)) {
		ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_pbuffs.dbell_addr_arr || NULL == bkpap_module->remote_pbuffs.dbell_addr_arr 
		|| NULL == bkpap_module->remote_pbuffs.buffer_addr_arr || NULL == bkpap_module->remote_pbuffs.buffer_addr_arr)) {
		ret = mca_coll_bkpap_wireup_postbuffs(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_syncstructure_arrival_arr_rkey || NULL == bkpap_module->remote_syncstructure_counter_rkey
		|| 0 == bkpap_module->remote_syncstructure_counter_addr || 0 == bkpap_module->remote_syncstructure_arrival_arr_addr)) {
		ret = mca_coll_bkpap_wireup_syncstructure(bkpap_module, comm);
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

	int intra_rank = ompi_comm_rank(bkpap_module->intra_comm);
	// inter_wsize = ompi_comm_size(bkpap_module->inter_comm);
	// inter_rank = ompi_comm_rank(bkpap_module->inter_comm);
	int inter_wsize = ompi_comm_size(comm); // TODO: fix when multi-node
	int inter_rank = ompi_comm_rank(comm); // TODO: fix when multi-node
	int global_rank = ompi_comm_rank(comm);
	int global_wsize = ompi_comm_size(comm);


	// bkpap_module->intra_comm->c_coll->coll_allreduce(
	// 		sbuf, rbuf, count, dtype, op, bkpap_module->intra_comm,
	// 		bkpap_module->intra_comm->c_coll->coll_allreduce_module);

	// if (low_rank == 0){

	// }

	ret = mca_coll_bkpap_arrive_at_inter(bkpap_module, comm, &arrival_pos);
	arrival_pos += 1;
	// BKPAP_OUTPUT("rank %d arrive %ld", inter_rank, arrival_pos);
	// comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
	// if(inter_rank == 0){
	// 	int64_t *_bk_tmp = (int64_t*)bkpap_module->local_syncstructure->arrival_arr_attr.address;
	// 	BKPAP_OUTPUT("ss arr [%ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld]",
	// 		_bk_tmp[0], _bk_tmp[1], _bk_tmp[2], _bk_tmp[3], _bk_tmp[4], _bk_tmp[5], _bk_tmp[6], _bk_tmp[7],
	// 		_bk_tmp[8], _bk_tmp[9], _bk_tmp[10], _bk_tmp[11], _bk_tmp[12], _bk_tmp[13], _bk_tmp[14], _bk_tmp[15]);
	// }

	int tmp_k = k;
	while (arrival_pos % tmp_k == 0) {
		BKPAP_OUTPUT("rank %d arrive %ld recive with tmp_k %d", global_rank, arrival_pos, tmp_k);
		mca_coll_bkpap_reduce_postbufs(comm, bkpap_module);

		tmp_k *= k;
		if (tmp_k > inter_wsize) break;
	}

	if (arrival_pos == 0) {
		if ((tmp_k / k) < global_wsize) {  // condition to do final recieve if not power of k
			BKPAP_OUTPUT("rank %d arrive %ld recive with tmp_k %d", global_rank, arrival_pos, tmp_k);
			mca_coll_bkpap_reduce_postbufs(comm, bkpap_module);
		}
	}
	else {
		int send_arrival_pos;
		int send_hrank = -1;
		send_arrival_pos = arrival_pos - (arrival_pos % tmp_k);
		while(-1 == send_hrank){
			mca_coll_bkpap_get_rank_of_arrival(send_arrival_pos, bkpap_module, comm, &send_hrank);
			if(-1 == send_hrank){
				BKPAP_OUTPUT("rank %d translated pos %d to -1, trying again", inter_rank, send_arrival_pos);
				// sleep(2);
			}

		}
		
		BKPAP_OUTPUT("rank %d arrive %ld send to pos %d (rank %d)", global_rank, arrival_pos, send_arrival_pos, send_hrank);
		ret = mca_coll_bkpap_write_parent_postbuf(sbuf, dtype, count, arrival_pos, send_hrank, comm, bkpap_module);
	}

	// internode bcast
	// bkpap_module->inter_comm->c_coll->coll_bcast(
	// 	, count, dtype, , 
	// 	bkpap_module->inter_comm,
	// 	bkpap_module->inter_comm->c_coll->coll_bcast_module
	// );

	// intranode bcast
	// bkpap_module->intra_comm->c_coll->coll_bcast(
	// 	, count, dtype, 0, 
	// 	bkpap_module->intra_comm,
	// 	bkpap_module->intra_comm->c_coll->coll_bcast_module
	// );


	// internode allreduce (see sm for starters, shift to Yiltan's GPU one eventualy)

	// atomic_fadd into counter of remote structure
	// depending on arrival position, calc parrents
	// 	if not first in section, send to parent (direct alg) 
	// 	if first in section, figure out how to recieve and reduce
	//	repeat to go upp the tree

	comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);
	if (global_rank == 0) {
		int64_t* tmp = (int64_t*)bkpap_module->local_syncstructure->counter_attr.address;
		*tmp = -1;
		tmp = (int64_t*)bkpap_module->local_syncstructure->arrival_arr_attr.address;
		for(int i = 0; i<global_wsize; i++)
			tmp[i] = -1;
	}

bkpap_ar_fallback:
	// return ompi_coll_base_allreduce_intra_basic_linear(
	// 	sbuf, rbuf, count, dtype, op, comm,
	// 	bkpap_module->fallback_allreduce_module);

	return bkpap_module->fallback_allreduce(
		sbuf, rbuf, count, dtype, op, comm,
		bkpap_module->fallback_allreduce_module);
}