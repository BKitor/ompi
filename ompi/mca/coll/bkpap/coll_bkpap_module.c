#include "coll_bkpap.h"
#include <string.h>
#include "opal/util/show_help.h"

static void mca_coll_bkpap_module_construct(mca_coll_bkpap_module_t* module) {
	module->fallback_allreduce = NULL;
	module->fallback_allreduce_module = NULL;

	module->wsize = -1;
	module->rank = -1;
	module->ucp_ep_arr = NULL;
	
	module->local_postbuf_h = NULL;
	module->local_postbuf_attrs.address = NULL;
	module->local_postbuf_attrs.field_mask = 0;
	module->local_postbuf_attrs.length = -1;

	module->remote_postbuff_addr_arr = NULL;
	module->remote_postbuff_rkey_arr = NULL;
	
	module->local_syncstructure = NULL;
	module->remote_syncstructure_counter_addr = 0;
	module->remote_syncstructure_counter_rkey = NULL;
	module->remote_syncstructure_arrival_addr = 0;
	module->remote_syncstructure_arrival_rkey = NULL;
}

static void mca_coll_bkpap_module_destruct(mca_coll_bkpap_module_t* module) {
	
	if(NULL != module->local_postbuf_h){
		ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, module->local_postbuf_h);
	}
	free(module->remote_postbuff_addr_arr);
	module->remote_postbuff_addr_arr = NULL;
	for(int32_t i = 0; i < module->wsize; i++){
		if(module->remote_postbuff_rkey_arr == NULL)break;
		if(module->remote_postbuff_rkey_arr[i] == NULL)continue;
		ucp_rkey_destroy(module->remote_postbuff_rkey_arr[i]);
	}
	free(module->remote_postbuff_rkey_arr);
	module->remote_postbuff_addr_arr = NULL;
	
	if(NULL != module->local_syncstructure){
		if(NULL != module->local_syncstructure->counter_mem_h)
			ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, module->local_syncstructure->counter_mem_h);
		if(NULL != module->local_syncstructure->arrival_arr_mem_h)
			ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, module->local_syncstructure->arrival_arr_mem_h);
		free(module->local_syncstructure);
		module->local_syncstructure = NULL;
	}

	if(NULL != module->remote_syncstructure_counter_rkey)
		ucp_rkey_destroy(module->remote_syncstructure_counter_rkey);
	module->remote_syncstructure_counter_rkey = NULL;
	module->remote_syncstructure_counter_addr = 0;
	if(NULL != module->remote_syncstructure_arrival_rkey)
		ucp_rkey_destroy(module->remote_syncstructure_arrival_rkey);
	module->remote_syncstructure_arrival_rkey = NULL;
	module->remote_syncstructure_arrival_addr = 0;
	
	for (int32_t i = 0; i < module->wsize; i++) {
		if(NULL == module->ucp_ep_arr) break;
		if (NULL == module->ucp_ep_arr[i]) continue;
		ucp_ep_destroy(module->ucp_ep_arr[i]);
	}
	free(module->ucp_ep_arr);
	module->ucp_ep_arr = NULL;

	OBJ_RELEASE(module->fallback_allreduce_module);
}

OBJ_CLASS_INSTANCE(mca_coll_bkpap_module_t, mca_coll_base_module_t,
	mca_coll_bkpap_module_construct, mca_coll_bkpap_module_destruct);

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority) {
	mca_coll_bkpap_module_t* bkpap_module;

	bkpap_module = OBJ_NEW(mca_coll_bkpap_module_t);
	if (NULL == bkpap_module) {
		return NULL;
	}

	*priority = mca_coll_bkpap_component.priority;
	bkpap_module->super.coll_module_enable = mca_coll_bkpap_module_enable;
	bkpap_module->super.coll_allreduce = mca_coll_bkpap_allreduce;

	return &(bkpap_module->super);
}

int mca_coll_bkpap_module_enable(mca_coll_base_module_t* module, struct ompi_communicator_t* comm) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;

	// check for allgather/allgaterv, needed for setup when echangaing ucp data
	if (NULL == comm->c_coll->coll_allgather_module || NULL == comm->c_coll->coll_allgatherv_module) {
		opal_show_help("help-mpi-coll-bkpap.txt", "missing collective", true,
			ompi_process_info.nodename,
			mca_coll_bkpap_component.priority, "allreduce");
		return OMPI_ERR_NOT_FOUND;
	}

	// check for allreduce and retain fallback
	if (NULL == comm->c_coll->coll_allreduce_module) {
		opal_show_help("help-mpi-coll-bkpap.txt", "missing collective", true,
			ompi_process_info.nodename,
			mca_coll_bkpap_component.priority, "allreduce");
		return OMPI_ERR_NOT_FOUND;
	}
	else {
		bkpap_module->fallback_allreduce_module = comm->c_coll->coll_allreduce_module;
		bkpap_module->fallback_allreduce = comm->c_coll->coll_allreduce;
		OBJ_RETAIN(bkpap_module->fallback_allreduce_module);
	}

	return OMPI_SUCCESS;
}
