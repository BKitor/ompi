#include "coll_bkpap.h"
#include <string.h>
#include "opal/util/show_help.h"

static void mca_coll_bkpap_module_construct(mca_coll_bkpap_module_t* module) {
}

static void mca_coll_bkpap_module_destruct(mca_coll_bkpap_module_t* module) {
	OBJ_RELEASE(module->fallback_allreduce_module);
}

OBJ_CLASS_INSTANCE(mca_coll_bkpap_module_t, mca_coll_base_module_t,
	mca_coll_bkpap_module_construct, mca_coll_bkpap_module_destruct);

int mca_coll_bkpap_init_query(bool enable_progress_threads, bool enable_mpi_threads) {
	return OMPI_SUCCESS;
}

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority) {
	mca_coll_bkpap_module_t* bkpap_module;

	bkpap_module = OBJ_NEW(mca_coll_bkpap_module_t);
	if (NULL == bkpap_module) {
		return NULL;
	}

	*priority = mca_coll_bkpap_component.priority;
	
	bkpap_module->super.coll_module_enable = mca_coll_bkpap_module_enable;

    bkpap_module->super.coll_allreduce  = mca_coll_bkpap_allreduce;
	
	return &(bkpap_module->super);
}

int mca_coll_bkpap_module_enable(mca_coll_base_module_t *module, struct ompi_communicator_t *comm){
	mca_coll_bkpap_module_t *bkpap_module = (mca_coll_bkpap_module_t*) module;
	
	if (NULL == comm->c_coll->coll_allreduce_module){
		opal_show_help("help-mpi-coll-bkpap.txt", "missing collective", true,
				ompi_process_info.nodename,
				mca_coll_bkpap_component.priority, "allreduce");
		return OMPI_ERR_NOT_FOUND;
	}else{
		bkpap_module->fallback_allreduce_module = comm->c_coll->coll_allreduce_module;
		bkpap_module->fallback_allreduce = comm->c_coll->coll_allreduce;
		OBJ_RETAIN(comm->c_coll->coll_allreduce_module);
	}
	
	return OMPI_SUCCESS;
}