#include "coll_bkpap.h"

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module) {

	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*) module;
	
	BKPAP_OUTPUT("Allreduce Called");

	return bkpap_module->fallback_allreduce(
		sbuf, rbuf, count, dtype, op, comm,
		bkpap_module->fallback_allreduce_module);


}