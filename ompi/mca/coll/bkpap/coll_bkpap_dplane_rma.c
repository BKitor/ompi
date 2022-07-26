#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

int coll_bkpap_rma_send_to_late(void* send_buf, int send_count, struct ompi_datatype_t* dtype,
	uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module) {


	return coll_bkpap_tag_send_to_late(send_buf, send_count, dtype, tag, tag_mask, comm, bkpap_module);
}

int coll_bkpap_rma_recv_from_early(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag,
	uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	return coll_bkpap_tag_recv_from_early(recv_buf, recv_count, dtype, peer_rank, tag, tag_mask, comm, bkpap_module);
}

// ucp_atomic_cswap if(dbell == DBELL_EMPTY ) -> dbell <= DBELL_INUSE
// ucp_put_nbx data
// ucp_flush_ep
// ucp_atomic_swap dbell <= DBELL_FULL
int coll_bkpap_rma_send_to_early(void* send_buf, int send_count,
	struct ompi_datatype_t* dtype, int peer_rank,
	uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module) {

	ucs_status_ptr_t req_ptr;
	ucs_status_t stat;
	ucp_ep_h peer_ep = bkpap_module->ucp_ep_arr[peer_rank];

	uint64_t pbuf_addr = bkpap_module->dplane.rma.remote.buffer_addr_arr[peer_rank];
	ucp_rkey_h pbuf_rkey = bkpap_module->dplane.rma.remote.buffer_rkey_arr[peer_rank];
	uint64_t dbell_addr = bkpap_module->dplane.rma.remote.dbell_addr_arr[peer_rank];
	ucp_rkey_h dbell_rkey = bkpap_module->dplane.rma.remote.dbell_rkey_arr[peer_rank];

	ptrdiff_t extent;
	ompi_datatype_type_extent(dtype, &extent);
	size_t send_buf_size = extent * send_count;

	uint64_t amo_reply_buf = BKPAP_DBELL_INUSE, amo_buf = BKPAP_DBELL_EMPTY;
	ucp_request_param_t put_params, amo_params;
	bk_fill_put_params(&put_params, bkpap_module);
	bk_fill_amo_params(&amo_params, &amo_reply_buf);

	while (BKPAP_DBELL_EMPTY != amo_reply_buf) {
		amo_reply_buf = BKPAP_DBELL_INUSE;
		req_ptr = ucp_atomic_op_nbx(peer_ep, UCP_ATOMIC_OP_CSWAP, &amo_buf, 1, dbell_addr, dbell_rkey, &amo_params);
		stat = bk_poll_ucs_completion(req_ptr);
		if (OPAL_UNLIKELY(UCS_OK != stat)) {
			BKPAP_ERROR("cswap_failed poll_all_completion");
			return OMPI_ERROR;
		}
		// add a usleep here? for like 20us or sometihng...?
	}

	req_ptr = ucp_put_nbx(peer_ep, send_buf, send_buf_size, pbuf_addr, pbuf_rkey, &put_params);
	stat = bk_poll_ucs_completion(req_ptr);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("data put_nbx poll_all_completion");
		return OMPI_ERROR;
	}

	put_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
	req_ptr = ucp_ep_flush_nbx(peer_ep, &put_params);
	stat = bk_poll_ucs_completion(req_ptr);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("flush poll_all_completion");
		return OMPI_ERROR;
	}

	amo_params.memory_type = UCS_MEMORY_TYPE_HOST;
	amo_buf = BKPAP_DBELL_FULL;
	req_ptr = ucp_atomic_op_nbx(peer_ep, UCP_ATOMIC_OP_SWAP, &amo_buf, 1, dbell_addr, dbell_rkey, &amo_params);
	stat = bk_poll_ucs_completion(req_ptr);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("swap poll_all_completion");
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

// poll dbell until completion
int coll_bkpap_rma_recv_from_late(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	volatile int64_t* dbell = bkpap_module->dplane.rma.local.dbell_attrs.address;

	while (BKPAP_DBELL_FULL != *dbell) ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
	// while (BKPAP_DBELL_SET != *dbell) ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);

	if (OPAL_UNLIKELY(recv_buf != bkpap_module->dplane.rma.local.postbuf_attrs.address)) {
		BKPAP_ERROR("recv_buf != local.pbuff");
		return OMPI_ERROR;
	}

	// *dbell = BKPAP_DBELL_UNSET;

	return OMPI_SUCCESS;
}

int coll_bkpap_rma_sendrecv_from_early(void* send_buf, int send_count, void* recv_buf,
	int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag,
	uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	return coll_bkpap_tag_sendrecv_from_early(send_buf, send_count, recv_buf, recv_count, dtype, peer_rank, tag, tag_mask, comm, bkpap_module);
}

int coll_bkpap_rma_sendrecv_from_late(void* send_buf, int send_count, void* recv_buf,
	int recv_count, struct ompi_datatype_t* dtype, uint64_t tag,
	uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	return coll_bkpap_tag_sendrecv_from_late(send_buf, send_count, recv_buf, recv_count, dtype, tag, tag_mask, comm, bkpap_module);
}

int coll_bkpap_rma_sendrecv(void* sbuf, int send_count, void* rbuf, int recv_count,
	struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag,
	uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	return coll_bkpap_tag_sendrecv(sbuf, send_count, rbuf, recv_count, dtype, peer_rank, tag, tag_mask, comm, bkpap_module);
}

int coll_bkpap_rma_reset_late_recv_buf(mca_coll_bkpap_module_t* bkpap_module) {
	int64_t* dbell = bkpap_module->dplane.rma.local.dbell_attrs.address;
	dbell[0] = BKPAP_DBELL_EMPTY;
	return OMPI_SUCCESS;
}

static void bk_set_rma_dplane_ftbl(coll_bkpap_dplane_ftbl_t* ftbl) {
	ftbl->send_to_early = coll_bkpap_rma_send_to_early;
	ftbl->send_to_late = coll_bkpap_rma_send_to_late;
	ftbl->recv_from_early = coll_bkpap_rma_recv_from_early;
	ftbl->recv_from_late = coll_bkpap_rma_recv_from_late;
	ftbl->sendrecv_from_early = coll_bkpap_rma_sendrecv_from_early;
	ftbl->sendrecv_from_late = coll_bkpap_rma_sendrecv_from_late;
	ftbl->sendrecv = coll_bkpap_rma_sendrecv;
	ftbl->reset_late_recv_buf = coll_bkpap_rma_reset_late_recv_buf;
}

int mca_coll_bkpap_rma_dplane_wireup(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm) {
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
	BKPAP_MSETZ(bkpap_module->dplane.rma.local.dbell_attrs);
	BKPAP_MSETZ(bkpap_module->dplane.rma.local.postbuf_attrs);

	void* mapped_postbuf = NULL;
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size;

	switch (bkpap_module->dplane_mem_t) {
	case BKPAP_DPLANE_MEM_TYPE_HOST:
		ret = posix_memalign(&mapped_postbuf, sizeof(int64_t), mapped_postbuf_size);
		if (0 != ret || NULL == mapped_postbuf) {
			BKPAP_ERROR("posix_memalign failed, exiting");
			goto bkpap_remotepostbuf_wireup_err;
		}
		ret = OMPI_SUCCESS;
		break;
	case BKPAP_DPLANE_MEM_TYPE_CUDA:
		ret = cudaMalloc(&mapped_postbuf, mapped_postbuf_size);
		BKPAP_CHK_CUDA(ret, bkpap_remotepostbuf_wireup_err);
		ret = OMPI_SUCCESS;
		break;
	case BKPAP_DPLANE_MEM_TYPE_CUDA_MANAGED:
		ret = cudaMallocManaged(&mapped_postbuf, mapped_postbuf_size, cudaMemAttachGlobal);
		BKPAP_CHK_CUDA(ret, bkpap_remotepostbuf_wireup_err);
		ret = OMPI_SUCCESS;
		break;
	default:
		BKPAP_ERROR("Bad memory type, %d", bkpap_module->dplane_mem_t);
		return OMPI_ERROR;
		break;
	}

	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
	mem_map_params.address = mapped_postbuf;
	mem_map_params.length = mapped_postbuf_size;
	mem_map_params.memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);

	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &bkpap_module->dplane.rma.local.postbuf_h);
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
	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &bkpap_module->dplane.rma.local.dbell_h);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	bkpap_module->dplane.rma.local.postbuf_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(bkpap_module->dplane.rma.local.postbuf_h, &bkpap_module->dplane.rma.local.postbuf_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	bkpap_module->dplane.rma.local.dbell_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(bkpap_module->dplane.rma.local.dbell_h, &bkpap_module->dplane.rma.local.dbell_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	int64_t* dbells = bkpap_module->dplane.rma.local.dbell_attrs.address;
	dbells[0] = BKPAP_DBELL_EMPTY;

	bkpap_module->dplane.rma.remote.buffer_addr_arr = calloc(mpi_size, sizeof(*bkpap_module->dplane.rma.remote.buffer_addr_arr));
	BKPAP_CHK_MALLOC(bkpap_module->dplane.rma.remote.buffer_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&bkpap_module->dplane.rma.local.postbuf_attrs.address, 1, MPI_LONG_LONG,
		bkpap_module->dplane.rma.remote.buffer_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	bkpap_module->dplane.rma.remote.dbell_addr_arr = calloc(mpi_size, sizeof(*bkpap_module->dplane.rma.remote.dbell_addr_arr));
	BKPAP_CHK_MALLOC(bkpap_module->dplane.rma.remote.dbell_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&bkpap_module->dplane.rma.local.dbell_attrs.address, 1, MPI_LONG_LONG,
		bkpap_module->dplane.rma.remote.dbell_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, bkpap_module->dplane.rma.local.postbuf_h, &postbuf_rkey_buffer, &postbuf_rkey_buffer_size);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, bkpap_module->dplane.rma.local.dbell_h, &dbell_rkey_buffer, &dbell_rkey_buffer_size);
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
	bkpap_module->dplane.rma.remote.buffer_rkey_arr = calloc(mpi_size, sizeof(*bkpap_module->dplane.rma.remote.buffer_rkey_arr));
	BKPAP_CHK_MALLOC(bkpap_module->dplane.rma.remote.buffer_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			bkpap_module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&bkpap_module->dplane.rma.remote.buffer_rkey_arr[i]);
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
	bkpap_module->dplane.rma.remote.dbell_rkey_arr = calloc(mpi_size, sizeof(*bkpap_module->dplane.rma.remote.dbell_rkey_arr));
	BKPAP_CHK_MALLOC(bkpap_module->dplane.rma.remote.dbell_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			bkpap_module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&bkpap_module->dplane.rma.remote.dbell_rkey_arr[i]);
		BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	}

	ucp_rkey_buffer_release(postbuf_rkey_buffer);

	bk_set_rma_dplane_ftbl(&bkpap_module->dplane_ftbl);

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
	bkpap_dplane_mem_t mem_t = bkpap_module->dplane_mem_t;
	if (BKPAP_DPLANE_MEM_TYPE_CUDA == mem_t || BKPAP_DPLANE_MEM_TYPE_CUDA_MANAGED == mem_t) {
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