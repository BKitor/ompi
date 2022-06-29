#include "coll_bkpap.h"

int bk_inter_bcast(void* buf, int count, struct ompi_datatype_t* dtype, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {
	int ret = OMPI_SUCCESS, rank = ompi_comm_rank(comm);

	// ptrdiff_t lb, extent;
	// ompi_datatype_get_extent(dtype, &lb, &extent);
	// void* bc_tmp =  NULL;
	// ret = bkpap_mempool_alloc(&bc_tmp, (count * extent), BKPAP_POSTBUF_MEMORY_TYPE_HOST, bkpap_module);
	// BKPAP_CHK_MPI_MSG_LBL(ret, "bk_mempool_alloc failed", bkpap_abort_binomial_allreduce);

	// if (bcast_root == inter_rank)
	// 	ompi_datatype_copy_content_same_ddt(dtype, count, bc_tmp, rbuf);
	ret = ompi_coll_base_bcast_intra_scatter_allgather_ring(buf, count, dtype, root, comm, &bkpap_module->super, 0);
	BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_opt_bc);

	// if (bcast_root != inter_rank)
	// 	ompi_datatype_copy_content_same_ddt(dtype, count, rbuf, bc_tmp);
	// ret = bk_mempool_free(bc_tmp, BKPAP_POSTBUF_MEMORY_TYPE_HOST, bkpap_module);
	// BKPAP_CHK_MPI_MSG_LBL(ret, "bk_mempool_free failed", bkpap_abort_binomial_allreduce);

	return OMPI_SUCCESS;
bkpap_abort_opt_bc:
	return ret;
}

int bk_intra_bcast(void* buf, int count, struct ompi_datatype_t* dtype, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {
	int ret = OMPI_SUCCESS;
	ret = ompi_coll_base_bcast_intra_scatter_allgather_ring(buf, count, dtype, root, comm, &bkpap_module->super, 0);
	return ret;
}