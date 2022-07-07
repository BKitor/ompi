#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

int coll_bkpap_rma_send_to_early(void* send_buf, int send_count,
	struct ompi_datatype_t* dtype, int peer_rank,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;

}

int coll_bkpap_rma_send_to_late(void* send_buf, int send_count, struct ompi_datatype_t* dtype,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

int coll_bkpap_rma_recv_from_early(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, int peer_rank, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

int coll_bkpap_rma_recv_from_late(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, int64_t tag, int64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

int coll_bkpap_rma_sendrecv_from_early(void* send_buf, int send_count, void* recv_buf,
	int recv_count, struct ompi_datatype_t* dtype, int peer_rank, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

int coll_bkpap_rma_sendrecv_from_late(void* send_buf, int send_count, void* recv_buf,
	int recv_count, struct ompi_datatype_t* dtype, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

int coll_bkpap_rma_sendrecv(void* sbuf, int send_count, void* rbuf, int recv_count,
	struct ompi_datatype_t* dtype, ompi_op_t* op, int peer_rank, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	return OPAL_ERR_NOT_IMPLEMENTED;
}

static void bk_set_rma_dplane_ftbl(coll_bkpap_dplane_ftbl_t* ftbl) {
	ftbl->send_to_early = coll_bkpap_rma_send_to_early;
	ftbl->send_to_late = coll_bkpap_rma_send_to_late;
	ftbl->recv_from_early = coll_bkpap_rma_recv_from_early;
	ftbl->recv_from_late = coll_bkpap_rma_recv_from_late;
	ftbl->sendrecv_from_early = coll_bkpap_rma_sendrecv_from_early;
	ftbl->sendrecv_from_late = coll_bkpap_rma_sendrecv_from_late;
	ftbl->sendrecv = coll_bkpap_rma_sendrecv;
}

int mca_coll_bkpap_rma_dplane_wireup(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS, mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	ucs_status_t status = UCS_OK;
	ucp_mem_map_params_t mem_map_params;
	void* postbuf_rkey_buffer = NULL, * dbell_rkey_buffer = NULL;
	size_t postbuf_rkey_buffer_size, dbell_rkey_buffer_size, * postbuf_rkey_size_arr = NULL, * dbell_rkey_size_arr = NULL;
	int* agv_displ_arr = NULL, * agv_count_arr = NULL;
	uint8_t* agv_rkey_recv_buf = NULL;
	size_t agv_rkey_recv_buf_size = 0;

	if (BKPAP_ALLREDUCE_ALG_RSA == mca_coll_bkpap_component.allreduce_alg) {
		BKPAP_ERROR("RSA with RMS dataplane is not supported");
		return OMPI_ERR_NOT_SUPPORTED;
	}

	BKPAP_MSETZ(mem_map_params);
	BKPAP_MSETZ(module->dplane.rma.local.dbell_attrs);
	BKPAP_MSETZ(module->dplane.rma.local.postbuf_attrs);

	void* mapped_postbuf = NULL;
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size;

	switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
	case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
		ret = posix_memalign(&mapped_postbuf, sizeof(int64_t), mapped_postbuf_size);
		if (0 != ret || NULL == mapped_postbuf) {
			BKPAP_ERROR("posix_memalign failed, exiting");
			goto bkpap_remotepostbuf_wireup_err;
		}
		ret = OMPI_SUCCESS;
		break;
	case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
		ret = cudaMalloc(&mapped_postbuf, mapped_postbuf_size);
		BKPAP_CHK_CUDA(ret, bkpap_remotepostbuf_wireup_err);
		ret = OMPI_SUCCESS;
		break;
	case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
		ret = cudaMallocManaged(&mapped_postbuf, mapped_postbuf_size, cudaMemAttachGlobal);
		BKPAP_CHK_CUDA(ret, bkpap_remotepostbuf_wireup_err);
		ret = OMPI_SUCCESS;
		break;
	default:
		BKPAP_ERROR("Bad memory type, %d", mca_coll_bkpap_component.bk_postbuf_memory_type);
		return OMPI_ERROR;
		break;
	}

	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
	mem_map_params.address = mapped_postbuf;
	mem_map_params.length = mapped_postbuf_size;
	mem_map_params.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type;

	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->dplane.rma.local.postbuf_h);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	BKPAP_MSETZ(mem_map_params);
	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_FLAGS |
		UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
	mem_map_params.address = NULL;
	mem_map_params.length = sizeof(int64_t);
	mem_map_params.flags = UCP_MEM_MAP_ALLOCATE;
	mem_map_params.memory_type = UCS_MEMORY_TYPE_HOST;
	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->dplane.rma.local.dbell_h);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	module->dplane.rma.local.postbuf_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(module->dplane.rma.local.postbuf_h, &module->dplane.rma.local.postbuf_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	module->dplane.rma.local.dbell_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(module->dplane.rma.local.dbell_h, &module->dplane.rma.local.dbell_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	int64_t* dbells = module->dplane.rma.local.dbell_attrs.address;
	dbells[0] = BKPAP_DBELL_UNSET;

	module->dplane.rma.remote.buffer_addr_arr = calloc(mpi_size, sizeof(*module->dplane.rma.remote.buffer_addr_arr));
	BKPAP_CHK_MALLOC(module->dplane.rma.remote.buffer_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&module->dplane.rma.local.postbuf_attrs.address, 1, MPI_LONG_LONG,
		module->dplane.rma.remote.buffer_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	module->dplane.rma.remote.dbell_addr_arr = calloc(mpi_size, sizeof(*module->dplane.rma.remote.dbell_addr_arr));
	BKPAP_CHK_MALLOC(module->dplane.rma.remote.dbell_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&module->dplane.rma.local.dbell_attrs.address, 1, MPI_LONG_LONG,
		module->dplane.rma.remote.dbell_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->dplane.rma.local.postbuf_h, &postbuf_rkey_buffer, &postbuf_rkey_buffer_size);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->dplane.rma.local.dbell_h, &dbell_rkey_buffer, &dbell_rkey_buffer_size);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	postbuf_rkey_size_arr = calloc(mpi_size, sizeof(*postbuf_rkey_size_arr));
	BKPAP_CHK_MALLOC(postbuf_rkey_size_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&postbuf_rkey_buffer_size, 1, MPI_LONG_LONG,
		postbuf_rkey_size_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	dbell_rkey_size_arr = calloc(mpi_size, sizeof(*dbell_rkey_size_arr));
	BKPAP_CHK_MALLOC(dbell_rkey_size_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&dbell_rkey_buffer_size, 1, MPI_LONG_LONG,
		dbell_rkey_size_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	agv_displ_arr = calloc(mpi_size, sizeof(*agv_displ_arr));
	BKPAP_CHK_MALLOC(agv_displ_arr, bkpap_remotepostbuf_wireup_err);
	agv_count_arr = calloc(mpi_size, sizeof(*agv_count_arr));
	BKPAP_CHK_MALLOC(agv_count_arr, bkpap_remotepostbuf_wireup_err);

	agv_rkey_recv_buf_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = agv_rkey_recv_buf_size;
		agv_count_arr[i] = postbuf_rkey_size_arr[i];
		agv_rkey_recv_buf_size += postbuf_rkey_size_arr[i];
	}
	agv_rkey_recv_buf = malloc(agv_rkey_recv_buf_size);
	BKPAP_CHK_MALLOC(agv_rkey_recv_buf, bkpap_remotepostbuf_wireup_err);
	memset(agv_rkey_recv_buf, 0, agv_rkey_recv_buf_size);
	ret = comm->c_coll->coll_allgatherv(postbuf_rkey_buffer, postbuf_rkey_buffer_size, MPI_BYTE,
		agv_rkey_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);
	module->dplane.rma.remote.buffer_rkey_arr = calloc(mpi_size, sizeof(*module->dplane.rma.remote.buffer_rkey_arr));
	BKPAP_CHK_MALLOC(module->dplane.rma.remote.buffer_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->dplane.rma.remote.buffer_rkey_arr[i]);
		BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	}

	agv_rkey_recv_buf_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = agv_rkey_recv_buf_size;
		agv_count_arr[i] = dbell_rkey_size_arr[i];
		agv_rkey_recv_buf_size += dbell_rkey_size_arr[i];
	}
	free(agv_rkey_recv_buf);
	agv_rkey_recv_buf = malloc(agv_rkey_recv_buf_size);
	BKPAP_CHK_MALLOC(agv_rkey_recv_buf, bkpap_remotepostbuf_wireup_err);
	memset(agv_rkey_recv_buf, 0, agv_rkey_recv_buf_size);
	ret = comm->c_coll->coll_allgatherv(dbell_rkey_buffer, dbell_rkey_buffer_size, MPI_BYTE,
		agv_rkey_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);
	module->dplane.rma.remote.dbell_rkey_arr = calloc(mpi_size, sizeof(*module->dplane.rma.remote.dbell_rkey_arr));
	BKPAP_CHK_MALLOC(module->dplane.rma.remote.dbell_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->dplane.rma.remote.dbell_rkey_arr[i]);
		BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	}

	ucp_rkey_buffer_release(postbuf_rkey_buffer);

	bk_set_rma_dplane_ftbl(&module->dplane_ftbl);

	BKPAP_OUTPUT("ucp postbuf wireup SUCCESS");
bkpap_remotepostbuf_wireup_err:

	free(postbuf_rkey_size_arr);
	free(dbell_rkey_size_arr);
	free(agv_displ_arr);
	free(agv_count_arr);
	free(agv_rkey_recv_buf);
	return ret;
}

void mca_coll_bkpap_rma_dplane_destroy(mca_coll_bkpap_rma_dplane_t* rma_dplane, mca_coll_bkpap_module_t* bkpap_module) {
	for (int i = 0; i < bkpap_module->wsize; i++) {
		if (NULL == bkpap_module->dplane.rma.remote.buffer_rkey_arr)break;
		if (NULL == bkpap_module->dplane.rma.remote.buffer_rkey_arr[i])continue;
		ucp_rkey_destroy(bkpap_module->dplane.rma.remote.buffer_rkey_arr[i]);
	}
	free(bkpap_module->dplane.rma.remote.buffer_rkey_arr);
	bkpap_module->dplane.rma.remote.buffer_rkey_arr = NULL;
	free(bkpap_module->dplane.rma.remote.buffer_addr_arr);
	bkpap_module->dplane.rma.remote.buffer_addr_arr = NULL;
	for (int i = 0; i < bkpap_module->wsize; i++) {
		if (NULL == bkpap_module->dplane.rma.remote.dbell_rkey_arr)break;
		if (NULL == bkpap_module->dplane.rma.remote.dbell_rkey_arr[i]) continue;
		ucp_rkey_destroy(bkpap_module->dplane.rma.remote.dbell_rkey_arr[i]);
	}
	free(bkpap_module->dplane.rma.remote.dbell_rkey_arr);
	bkpap_module->dplane.rma.remote.dbell_rkey_arr = NULL;
	free(bkpap_module->dplane.rma.remote.dbell_addr_arr);
	bkpap_module->dplane.rma.remote.dbell_addr_arr = NULL;

	if (NULL != bkpap_module->dplane.rma.local.postbuf_h) {
		ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, bkpap_module->dplane.rma.local.postbuf_h);
	}
	bkpap_module->dplane.rma.local.postbuf_h = NULL;
	bkpap_module->dplane.rma.local.postbuf_attrs.address = NULL;
	void* free_local_pbuff = bkpap_module->dplane.rma.local.postbuf_attrs.address;
	mca_coll_bkpap_postbuf_memory_t mem_t = mca_coll_bkpap_component.bk_postbuf_memory_type;
	if (BKPAP_POSTBUF_MEMORY_TYPE_CUDA == mem_t || BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED == mem_t) {
		cudaFree(free_local_pbuff);
	}
	else {
		free(free_local_pbuff);
	}
	if (NULL != bkpap_module->dplane.rma.local.dbell_h) {
		ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, bkpap_module->dplane.rma.local.dbell_h);
	}
	bkpap_module->dplane.rma.local.dbell_h = NULL;
	bkpap_module->dplane.rma.local.dbell_attrs.address = NULL;

}