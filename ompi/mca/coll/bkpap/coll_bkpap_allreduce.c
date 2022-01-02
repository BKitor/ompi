#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"


int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
							struct ompi_datatype_t* dtype,
							struct ompi_op_t* op,
							struct ompi_communicator_t* comm,
							mca_coll_base_module_t* module){
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	int ret = OMPI_SUCCESS;
	int low_rank, hi_rank;
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

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_postbuff_addr_arr || NULL == bkpap_module->remote_postbuff_rkey_arr)) {
		ret = mca_coll_bkpap_wireup_remote_postbuffs(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_syncstructure_arrival_rkey || NULL == bkpap_module->remote_syncstructure_counter_rkey
		|| 0 == bkpap_module->remote_syncstructure_counter_addr || 0 == bkpap_module->remote_syncstructure_arrival_addr)) {
		ret = mca_coll_bkpap_wireup_syncstructure(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	// check if intrandode communicator exists
	
	if(OPAL_UNLIKELY(NULL == bkpap_module->intra_comm || NULL == bkpap_module->inter_comm)){
		ret = mca_coll_bkpap_wirup_hier_comms(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("inter/intra communicator creation failed");
			goto bkpap_ar_fallback;
		}
	}
	
	low_rank = ompi_comm_rank(bkpap_module->intra_comm);
	hi_rank = ompi_comm_rank(comm);
	
	
	// bkpap_module->intra_comm->c_coll->coll_allreduce(
	// 		sbuf, rbuf, count, dtype, op, bkpap_module->intra_comm,
	// 		bkpap_module->intra_comm->c_coll->coll_allreduce_module);
	
	ret = mca_coll_bkpap_arrive_at_inter(bkpap_module, comm, &arrival_pos);
	BKPAP_OUTPUT("rank %d arrived at position %ld", hi_rank, arrival_pos);
	comm->c_coll->coll_barrier(comm, comm->c_coll->coll_barrier_module);

	if(hi_rank == 0){
		int64_t *tmp = (int64_t*) bkpap_module->local_syncstructure->counter_attr.address;
		*tmp = -1;
	}
	// bkpap_module->inter_comm->c_coll->coll_bcast(
	// 	, count, dtype, , 
	// 	bkpap_module->inter_comm,
	// 	bkpap_module->inter_comm->c_coll->coll_bcast_module
	// );
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


bkpap_ar_fallback:
	// return ompi_coll_base_allreduce_intra_basic_linear(
	// 	sbuf, rbuf, count, dtype, op, comm,
	// 	bkpap_module->fallback_allreduce_module);

	return bkpap_module->fallback_allreduce(
		sbuf, rbuf, count, dtype, op, comm,
		bkpap_module->fallback_allreduce_module);
}