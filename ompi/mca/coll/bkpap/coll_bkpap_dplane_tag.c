#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

int mca_coll_bkpap_tag_wireup(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	return OPAL_ERR_NOT_IMPLEMENTED;
}

int mca_coll_bkpap_tag_send_postbuf(const void* buf, struct ompi_datatype_t* dtype, int count, int dest, int slot, struct ompi_communicator_t* comm, mca_coll_base_module_t* module) {
	return OPAL_ERR_NOT_IMPLEMENTED;
}

int mca_coll_bkpap_tag_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count, ompi_op_t* op, int num_buffers, ompi_communicator_t* comm, mca_coll_base_module_t* module) {
	return OPAL_ERR_NOT_IMPLEMENTED;
}