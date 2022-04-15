#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop


int mca_coll_bkpap_rma_wireup(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS, mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	ucs_status_t status = UCS_OK;
	ucp_mem_map_params_t mem_map_params;
	void* postbuf_rkey_buffer = NULL, * dbell_rkey_buffer = NULL;
	size_t postbuf_rkey_buffer_size, dbell_rkey_buffer_size, * postbuf_rkey_size_arr = NULL, * dbell_rkey_size_arr = NULL;
	int* agv_displ_arr = NULL, * agv_count_arr = NULL;
	uint8_t* agv_rkey_recv_buf = NULL;
	size_t agv_rkey_recv_buf_size = 0;

	BKPAP_MSETZ(mem_map_params);
	BKPAP_MSETZ(module->local_pbuffs.rma.dbell_attrs);
	BKPAP_MSETZ(module->local_pbuffs.rma.postbuf_attrs);
	module->local_pbuffs.rma.num_buffs = num_bufs;

	void* mapped_postbuf = NULL;
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size * (num_bufs);

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

	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_pbuffs.rma.postbuf_h);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	BKPAP_MSETZ(mem_map_params);
	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_FLAGS |
		UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
	mem_map_params.address = NULL;
	mem_map_params.length = sizeof(int64_t) * (num_bufs);
	mem_map_params.flags = UCP_MEM_MAP_ALLOCATE;
	mem_map_params.memory_type = UCS_MEMORY_TYPE_HOST;
	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_pbuffs.rma.dbell_h);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	module->local_pbuffs.rma.postbuf_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(module->local_pbuffs.rma.postbuf_h, &module->local_pbuffs.rma.postbuf_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	module->local_pbuffs.rma.dbell_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH; //| UCP_MEM_ATTR_FIELD_MEM_TYPE;
	status = ucp_mem_query(module->local_pbuffs.rma.dbell_h, &module->local_pbuffs.rma.dbell_attrs);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	int64_t* dbells = module->local_pbuffs.rma.dbell_attrs.address;
	for (int i = 0; i < (num_bufs); i++)
		dbells[i] = BKPAP_DBELL_UNSET;
	dbells = NULL;

	module->remote_pbuffs.buffer_addr_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.buffer_addr_arr));
	BKPAP_CHK_MALLOC(module->remote_pbuffs.buffer_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&module->local_pbuffs.rma.postbuf_attrs.address, 1, MPI_LONG_LONG,
		module->remote_pbuffs.buffer_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	module->remote_pbuffs.dbell_addr_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.dbell_addr_arr));
	BKPAP_CHK_MALLOC(module->remote_pbuffs.dbell_addr_arr, bkpap_remotepostbuf_wireup_err);
	ret = comm->c_coll->coll_allgather(
		&module->local_pbuffs.rma.dbell_attrs.address, 1, MPI_LONG_LONG,
		module->remote_pbuffs.dbell_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	BKPAP_CHK_MPI(ret, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_pbuffs.rma.postbuf_h, &postbuf_rkey_buffer, &postbuf_rkey_buffer_size);
	BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_pbuffs.rma.dbell_h, &dbell_rkey_buffer, &dbell_rkey_buffer_size);
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
	module->remote_pbuffs.buffer_rkey_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.buffer_rkey_arr));
	BKPAP_CHK_MALLOC(module->remote_pbuffs.buffer_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->remote_pbuffs.buffer_rkey_arr[i]);
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
	module->remote_pbuffs.dbell_rkey_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.dbell_rkey_arr));
	BKPAP_CHK_MALLOC(module->remote_pbuffs.dbell_rkey_arr, bkpap_remotepostbuf_wireup_err);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->remote_pbuffs.dbell_rkey_arr[i]);
		BKPAP_CHK_UCP(status, bkpap_remotepostbuf_wireup_err);
	}

	ucp_rkey_buffer_release(postbuf_rkey_buffer);
	BKPAP_OUTPUT("ucp postbuf wireup SUCCESS");
bkpap_remotepostbuf_wireup_err:

	free(postbuf_rkey_size_arr);
	free(dbell_rkey_size_arr);
	free(agv_displ_arr);
	free(agv_count_arr);
	free(agv_rkey_recv_buf);
	return ret;
}

int mca_coll_bkpap_rma_send_postbuf(const void* buf,
	struct ompi_datatype_t* dtype, int count, int dest, int slot,
	struct ompi_communicator_t* comm, mca_coll_base_module_t* module) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	ucs_status_t status = UCS_OK;
	ucs_status_ptr_t status_ptr = UCS_OK;
	int ret = OMPI_SUCCESS;
	int64_t dbell_put_buf = BKPAP_DBELL_SET;
	uint64_t postbuf_addr;
	size_t dtype_size, buf_size;

	ucp_request_param_t req_attr = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE | UCP_OP_ATTR_FIELD_CALLBACK,
		.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type,
		.user_data = NULL,
		.cb.send = _bk_send_cb
	};
	postbuf_addr = (bkpap_module->remote_pbuffs.buffer_addr_arr[dest]) + (slot * mca_coll_bkpap_component.postbuff_size);
	ompi_datatype_type_size(dtype, &dtype_size);
	buf_size = dtype_size * (ptrdiff_t)count;

	status_ptr = ucp_put_nbx(
		bkpap_module->ucp_ep_arr[dest], buf, buf_size,
		postbuf_addr,
		bkpap_module->remote_pbuffs.buffer_rkey_arr[dest],
		&req_attr);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("rank %d, write rank %d postbuf returned error %d (%s)", ompi_comm_rank(comm), dest, UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr)) {
		ucp_request_free(status_ptr);
	}

	status = ucp_worker_fence(mca_coll_bkpap_component.ucp_worker);
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker fence failed");
		return OMPI_ERROR;
	}

	uint64_t dbell_addr = (bkpap_module->remote_pbuffs.dbell_addr_arr[dest]) + (slot * sizeof(uint64_t));
	req_attr.memory_type = UCS_MEMORY_TYPE_HOST;
	status_ptr = ucp_put_nbx(
		bkpap_module->ucp_ep_arr[dest], &dbell_put_buf, sizeof(dbell_put_buf),
		dbell_addr,
		bkpap_module->remote_pbuffs.dbell_rkey_arr[dest],
		&req_attr);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("rank %d write rank %d debll returned error %d (%s)", ompi_comm_rank(comm), dest, UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr)) {
		ucp_request_free(status_ptr);
	}

	status = _bk_flush_worker();
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker Flush Failed");
		return OMPI_ERROR;
	}

	return ret;
}

int mca_coll_bkpap_rma_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count,
	ompi_op_t* op, int num_buffers, ompi_communicator_t* comm, mca_coll_base_module_t* module) {
	int ret = OMPI_SUCCESS;
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	volatile int64_t* dbells = bkpap_module->local_pbuffs.rma.dbell_attrs.address;
	uint8_t* pbuffs = bkpap_module->local_pbuffs.rma.postbuf_attrs.address;

	for (int i = 0; i < num_buffers; i++) {
		while (BKPAP_DBELL_UNSET == dbells[i]) ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
		void* recived_buffer = pbuffs + (i * mca_coll_bkpap_component.postbuff_size);

		switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
		case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
		case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
			bk_gpu_op_reduce(op, recived_buffer, local_buf, count, dtype);
			break;
		case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
			ompi_op_reduce(op, recived_buffer,
				local_buf, count, dtype);
			break;
		default:
			BKPAP_ERROR("Bad memory type, %d", mca_coll_bkpap_component.bk_postbuf_memory_type);
			return OMPI_ERROR;
			break;
		}

		dbells[i] = BKPAP_DBELL_UNSET;
		BKPAP_OUTPUT("FINISH_LOCAL_REDUCE rank: %d, i: %d, [%ld %ld %ld], memtype %d", ompi_comm_rank(comm), i, dbells[0], dbells[1], dbells[2], mca_coll_bkpap_component.bk_postbuf_memory_type);
	}

	return ret;
}