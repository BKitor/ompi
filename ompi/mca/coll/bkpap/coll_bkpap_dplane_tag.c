#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

static void _bk_recv_cb(void *request, ucs_status_t status, const ucp_tag_recv_info_t* tag_info, void* user_data){
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}


int mca_coll_bkpap_tag_wireup(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	mca_coll_bkpap_local_tag_postbuf_t* tag_postbuf = &module->local_pbuffs.tag;
	void* mapped_postbuf = NULL;
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size * num_bufs;
	int ret = posix_memalign(&mapped_postbuf, sizeof(int64_t), mapped_postbuf_size);
	if (0 != ret || NULL == mapped_postbuf) {
		BKPAP_ERROR("posix_memaligh failed");
		goto bkpap_abort_tag_wireup;
	}
	tag_postbuf->buff_arr = mapped_postbuf;
	tag_postbuf->num_buffs = num_bufs;
	tag_postbuf->buff_size = mca_coll_bkpap_component.postbuff_size;
	return OMPI_SUCCESS;

bkpap_abort_tag_wireup:
	BKPAP_ERROR("Tag Wireup Error");
	return OMPI_ERROR;
}


// ucp_send_tag_nbx buf to dest, poll completion
int mca_coll_bkpap_tag_send_postbuf(const void* buf, struct ompi_datatype_t* dtype,
	int count, int dest, int slot, struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module) {
	size_t dtype_size = 0, buf_size = 0;
	ucs_status_t status = UCS_OK;
	ucs_status_ptr_t status_ptr = NULL;
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	ucp_request_param_t req_params = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
						| UCP_OP_ATTR_FIELD_USER_DATA
						| UCP_OP_ATTR_FIELD_MEMORY_TYPE,
		.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type,
		.user_data = NULL,
		.cb.send = _bk_send_cb
	};

	ompi_datatype_type_size(dtype, &dtype_size);
	buf_size = dtype_size * (ptrdiff_t)count;

	ucp_ep_h dest_ep = bkpap_module->ucp_ep_arr[dest];
	ucp_tag_t ucp_tag = slot;

	status_ptr = ucp_tag_send_nbx(dest_ep, buf, buf_size, ucp_tag, &req_params);
	if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
		status = UCS_PTR_STATUS(status_ptr);
		BKPAP_ERROR("rank: %d, dest: %d, send error %d(%s)", ompi_comm_rank(comm), dest, status, ucs_status_string(status));
		return OMPI_ERROR;
	}
	status = _bk_poll_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("_poll_completoin of tag_send failed");
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

int mca_coll_bkpap_tag_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype,
	int count, ompi_op_t* op, int num_buffers,
	ompi_communicator_t* comm, mca_coll_base_module_t* module) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t  status = UCS_OK;
	size_t dtype_size = 0, buf_size = 0;
	void* ucp_recv_buf = bkpap_module->local_pbuffs.tag.buff_arr;

	ompi_datatype_type_size(dtype, &dtype_size);
	buf_size = dtype_size * (ptrdiff_t)count;

	for (int i = 0; i < num_buffers; i++) {
		ucp_request_param_t req_params = {
			.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
							| UCP_OP_ATTR_FLAG_NO_IMM_CMPL
							| UCP_OP_ATTR_FIELD_MEMORY_TYPE
							| UCP_OP_ATTR_FIELD_USER_DATA,
			.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type,
			.cb.recv = _bk_recv_cb,
			.user_data = NULL
		};

		status_ptr = ucp_tag_recv_nbx(mca_coll_bkpap_component.ucp_worker, ucp_recv_buf, buf_size, i, 0, &req_params);
		if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
			status = UCS_PTR_STATUS(status_ptr);
			BKPAP_ERROR("rank: %d, i: %d, recv error %d(%s)", ompi_comm_rank(comm), i, status, ucs_status_string(status));
			return OMPI_ERROR;
		}
		status = _bk_poll_completion(status_ptr);
		if(OPAL_UNLIKELY(UCS_OK != status)){
			BKPAP_ERROR("poll completion failed");
			return OMPI_ERROR;
		}

		switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
		case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
		case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
			bk_gpu_op_reduce(op, ucp_recv_buf, local_buf, count, dtype);
			break;
		case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
			ompi_op_reduce(op, ucp_recv_buf, local_buf, count, dtype);
			break;
		default:
			BKPAP_ERROR("Bad memory type, %d", mca_coll_bkpap_component.bk_postbuf_memory_type);
			return OMPI_ERROR;
			break;
		}


	}

	return OMPI_SUCCESS;
}