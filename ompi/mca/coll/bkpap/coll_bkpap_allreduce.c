#include "coll_bkpap.h"
#include "ompi/mca/coll/base/coll_base_functions.h"


int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;

	// if commutative, goto(fallback)


	// global rank 0 has the dstructure
	// rank 0 of each inter-communicator is a leader

	// check if intrandode communicator exists

	// check if remote memory is setup
	// 	check if rkey/addr exists on module
	// 	pap_datastructre is setup on rank 0
	// 	rkeys for remote leaders are available

	// internode allreduce (see sm for starters, shift to Yiltan's GPU one eventualy)

	// atomic_fadd into counter of remote structure
	// depending on arrival position, calc parrents
	// 	if not first in section, send to parent (direct alg) 
	// 	if first in section, figure out how to recieve and reduce
	//	repeat to go upp the tree

	BKPAP_OUTPUT("Allreduce Called rank %d", ompi_comm_rank(comm));

	if (OPAL_UNLIKELY(NULL == bkpap_module->ucp_ep_arr)) {
		int ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_postbuff_addr_arr || NULL == bkpap_module->remote_postbuff_rkey_arr)) {
		int ret = mca_coll_bkpap_wireup_remote_postbuffs(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Postbuffer Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}

	if (OPAL_UNLIKELY(NULL == bkpap_module->remote_syncstructure_arrival_rkey || NULL == bkpap_module->remote_syncstructure_counter_rkey
		|| 0 == bkpap_module->remote_syncstructure_counter_addr || 0 == bkpap_module->remote_syncstructure_arrival_addr)) {
		int ret = mca_coll_bkpap_wireup_syncstructure(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
			goto bkpap_ar_fallback;
		}
	}


bkpap_ar_fallback:
	// return ompi_coll_base_allreduce_intra_basic_linear(
	// 	sbuf, rbuf, count, dtype, op, comm,
	// 	bkpap_module->fallback_allreduce_module);

	return bkpap_module->fallback_allreduce(
		sbuf, rbuf, count, dtype, op, comm,
		bkpap_module->fallback_allreduce_module);
}