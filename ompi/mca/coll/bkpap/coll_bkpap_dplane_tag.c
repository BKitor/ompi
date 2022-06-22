#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"

#include "opal/cuda/common_cuda.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

static void _bk_recv_cb(void* request, ucs_status_t status, const ucp_tag_recv_info_t* tag_info, void* user_data) {
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}

static inline void bk_fill_tag_recv_params(ucp_request_param_t* req_params) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.recv = _bk_recv_cb;
	req_params->memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type;
	req_params->user_data = NULL;
	req_params->datatype = ucp_dt_make_contig(1);
}

static inline void bk_fill_tag_send_params(ucp_request_param_t* req_params) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.send = _bk_send_cb;
	req_params->user_data = NULL;
	req_params->memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type;
	req_params->datatype = ucp_dt_make_contig(1);
}


int mca_coll_bkpap_tag_wireup(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS, alg = mca_coll_bkpap_component.allreduce_alg, mem_type = mca_coll_bkpap_component.bk_postbuf_memory_type;
	mca_coll_bkpap_local_tag_postbuf_t* tag_postbuf = &module->local_pbuffs.tag;

	int num_postbufs = (BKPAP_ALLREDUCE_ALG_RSA == alg || BKPAP_ALLREDUCE_ALG_BINOMIAL == alg) ? 1 : (mca_coll_bkpap_component.allreduce_k_value - 1); // should depend on component.alg
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size * num_postbufs;
	void* mapped_postbuf = NULL;

	bk_alloc_pbufft(&mapped_postbuf, mapped_postbuf_size, mca_coll_bkpap_component.bk_postbuf_memory_type);

	tag_postbuf->buff_arr = mapped_postbuf;
	tag_postbuf->num_buffs = num_postbufs;
	tag_postbuf->buff_size = mca_coll_bkpap_component.postbuff_size;
	tag_postbuf->mem_type = mem_type;
	return ret;

	BKPAP_ERROR("Tag Wireup Error");
	return OMPI_ERROR;
}


// ucp_send_tag_nbx buf to dest, poll completion
int mca_coll_bkpap_tag_send_postbuf(const void* buf, struct ompi_datatype_t* dtype,
	int count, int dest, int slot, struct ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* module) {
	size_t dtype_size = 0, buf_size = 0;
	ucs_status_t status = UCS_OK;
	ucs_status_ptr_t status_ptr = NULL;
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

	ucp_ep_h dest_ep = module->ucp_ep_arr[dest];
	ucp_tag_t ucp_tag = slot;

	status_ptr = ucp_tag_send_nbx(dest_ep, buf, buf_size, ucp_tag, &req_params);
	if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
		status = UCS_PTR_STATUS(status_ptr);
		BKPAP_ERROR("rank: %d, dest: %d, send error %d(%s)", ompi_comm_rank(comm), dest, status, ucs_status_string(status));
		return OMPI_ERROR;
	}
	status = bk_poll_ucs_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("_poll_completoin of tag_send failed");
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

int mca_coll_bkpap_tag_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype,
	int count, ompi_op_t* op, int num_buffers,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t  status = UCS_OK;
	size_t dtype_size = 0, buf_size = 0;
	void* ucp_recv_buf = NULL;

	ompi_datatype_type_size(dtype, &dtype_size); // should probably be extend, but we don't care about derrived data types
	buf_size = dtype_size * (ptrdiff_t)count;

	for (int i = 0; i < num_buffers; i++) {
		ucp_request_param_t req_params = {
			.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
							| UCP_OP_ATTR_FIELD_MEMORY_TYPE
							| UCP_OP_ATTR_FIELD_USER_DATA,
			.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type,
			.cb.recv = _bk_recv_cb,
			.user_data = NULL
		};

		ptrdiff_t pbuf_recv_offset = i * mca_coll_bkpap_component.postbuff_size;
		ucp_recv_buf = ((int8_t*)module->local_pbuffs.tag.buff_arr) + pbuf_recv_offset;
		ucp_tag_t ucp_tag = i;

		status_ptr = ucp_tag_recv_nbx(mca_coll_bkpap_component.ucp_worker, ucp_recv_buf, buf_size, ucp_tag, 0, &req_params);
		if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
			status = UCS_PTR_STATUS(status_ptr);
			BKPAP_ERROR("rank: %d, i: %d, recv error %d(%s)", ompi_comm_rank(comm), i, status, ucs_status_string(status));
			return OMPI_ERROR;
		}
		status = bk_poll_ucs_completion(status_ptr);
		if (OPAL_UNLIKELY(UCS_OK != status)) {
			BKPAP_ERROR("poll completion failed");
			return OMPI_ERROR;
		}
		mca_coll_bkpap_reduce_local(op, ucp_recv_buf, local_buf, count, dtype);
	}

	return OMPI_SUCCESS;
}

int mca_coll_bkpap_sendrecv(void* sbuf, int send_count, void* rbuf, int recv_count, struct ompi_datatype_t* dtype, ompi_op_t* op, int peer_rank, int64_t tag, int64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t status_array[2];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_size = (send_count * extent), recv_size = (recv_count * extent);
	ucp_request_param_t send_params, recv_params;
	bk_fill_tag_send_params(&send_params);
	bk_fill_tag_recv_params(&recv_params);
	ucp_ep_h peer_ep = module->ucp_ep_arr[peer_rank];
	ucp_tag_t tag_tmp = BK_RSA_SET_TAG_TYPE_DATA(tag);

	status_array[0] = ucp_tag_send_nbx(peer_ep, sbuf, send_size, tag_tmp, &send_params); // send data
	status_array[1] = ucp_tag_recv_nbx(w, rbuf, recv_size, tag_tmp, tag_mask, &recv_params); // recv data
	
	stat = bk_poll_all_ucs_completion(status_array, 2);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in sendrecv_poll_all: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}

// pair with recv_from_late, recv data
int mca_coll_bkpap_dplane_send_to_early(void* send_buf, int send_count, struct ompi_datatype_t* dtype, int peer_rank, int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucp_ep_h ep = module->ucp_ep_arr[peer_rank];
	ucs_status_ptr_t data_send_ptr;
	ucs_status_t stat;
	ptrdiff_t extent, lb;
	ucp_request_param_t send_params;

	bk_fill_tag_send_params(&send_params);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_byte_count = (send_count * extent);

	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	data_send_ptr = ucp_tag_send_nbx(ep, send_buf, send_byte_count, tag, &send_params);
	stat = bk_poll_ucs_completion(data_send_ptr);
	if(OPAL_UNLIKELY(UCS_OK != stat)){
		BKPAP_ERROR("tag_send_to_early data send failed: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

// pair with recv_from_early, recv rank->send data
int mca_coll_bkpap_dplane_send_to_late(void* send_buf, int send_count, struct ompi_datatype_t* dtype, int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h ep;
	ucs_status_ptr_t rank_recv_req, data_send_req;
	ucs_status_t stat;
	ucp_request_param_t send_params, recv_params;
	ptrdiff_t extent, lb;

	bk_fill_tag_send_params(&send_params);
	bk_fill_tag_recv_params(&recv_params);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_byte_count = (send_count * extent);

	recv_params.memory_type = UCS_MEMORY_TYPE_HOST;
	int64_t rank_recv_buf;
	tag = BK_BINOMIAL_TAG_SET_RANK(tag);
	rank_recv_req = ucp_tag_recv_nbx(w, &rank_recv_buf, sizeof(rank_recv_buf), tag, tag_mask, &recv_params);
	
	if(OPAL_UNLIKELY(UCS_PTR_IS_ERR(rank_recv_req))){
		stat =  UCS_PTR_STATUS(rank_recv_buf);
		return OMPI_ERROR;
	}
	
	stat = bk_poll_ucs_completion(rank_recv_req);
	
	if(OPAL_UNLIKELY(UCS_OK  != stat)){
		return OMPI_ERROR;
	}

	ep = module->ucp_ep_arr[rank_recv_buf];
	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	data_send_req = ucp_tag_send_nbx(ep, send_buf, send_byte_count, tag, &send_params);
	stat = bk_poll_ucs_completion(data_send_req);
	if(OPAL_UNLIKELY(UCS_OK != stat)){
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}

// pair with send_to_late, send rank->recv data
int mca_coll_bkpap_dplane_recv_from_early(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucs_status_ptr_t stat_ptr_arr[2];
	ucs_status_t stat;
	ucp_request_param_t send_params, recv_params;
	ucp_ep_h ep = module->ucp_ep_arr[peer_rank];
	ptrdiff_t extent, lb;

	bk_fill_tag_send_params(&send_params);
	bk_fill_tag_recv_params(&recv_params);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent);

	tag = BK_BINOMIAL_TAG_SET_RANK(tag);
	int64_t send_rank_buf = ompi_comm_rank(comm);
	send_params.memory_type = UCS_MEMORY_TYPE_HOST;
	stat_ptr_arr[0] = ucp_tag_send_nbx(ep, &send_rank_buf, sizeof(send_rank_buf), tag, &send_params);

	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	stat_ptr_arr[1] = ucp_tag_recv_nbx(w, recv_buf, recv_byte_count, tag, tag_mask, &recv_params);
	stat = bk_poll_all_ucs_completion(stat_ptr_arr, 2);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("tag_recv_from_early poll all completion failed: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

// pair with send_to_early, recv data
int mca_coll_bkpap_dplane_recv_from_late(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucs_status_ptr_t stat_ptr;
	ucs_status_t stat;
	ucp_request_param_t recv_params;
	ptrdiff_t extent, lb;

	bk_fill_tag_recv_params(&recv_params);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent);

	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	stat_ptr = ucp_tag_recv_nbx(w, recv_buf, recv_byte_count, tag, tag_mask, &recv_params);
	stat = bk_poll_ucs_completion(stat_ptr);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("tag_recv_from_late tag recv failed: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

int mca_coll_bkpap_dplane_sendrecv_from_early(void* send_buf, int send_count,
	void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t status_req_arr[3];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h peer_ep = module->ucp_ep_arr[peer_rank];
	ucp_request_param_t send_params, recv_params;

	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent), send_byte_count = (send_count * extent);
	bk_fill_tag_send_params(&send_params);
	bk_fill_tag_recv_params(&recv_params);
	ucp_tag_t rank_send_tag = BK_RSA_SET_TAG_TYPE_RANK(tag);
	ucp_tag_t data_send_tag = BK_RSA_SET_TAG_TYPE_DATA(tag), data_recv_tag = BK_RSA_SET_TAG_TYPE_DATA(tag);

	int64_t rank_send_buf = ompi_comm_rank(comm);
	send_params.memory_type = UCS_MEMORY_TYPE_HOST;
	status_req_arr[0] = ucp_tag_send_nbx(peer_ep, &rank_send_buf, sizeof(rank_send_buf), rank_send_tag, &send_params);
	send_params.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type;
	status_req_arr[1] = ucp_tag_send_nbx(peer_ep, send_buf, send_byte_count, data_send_tag, &send_params);
	status_req_arr[2] = ucp_tag_recv_nbx(w, recv_buf, recv_byte_count, data_recv_tag, tag_mask, &recv_params);

	stat = bk_poll_all_ucs_completion(status_req_arr, 3);
	BKPAP_PROFILE("dplane_tag_late_p2p_wait_end", ompi_comm_rank(comm));
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in sendrecv_poll_all: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}

int mca_coll_bkpap_dplane_sendrecv_from_late(void* send_buf, int send_count,
	void* recv_buf, int recv_count, struct ompi_datatype_t* dtype,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {

	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t rank_recv_req, data_req_arr[2];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h peer_ep;
	ucp_request_param_t send_params, recv_params;

	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent), send_byte_count = (send_count * extent);
	bk_fill_tag_send_params(&send_params);
	bk_fill_tag_recv_params(&recv_params);
	ucp_tag_t rank_recv_tag = BK_RSA_SET_TAG_TYPE_RANK(tag);
	ucp_tag_t data_send_tag = BK_RSA_SET_TAG_TYPE_DATA(tag), data_recv_tag = BK_RSA_SET_TAG_TYPE_DATA(tag);

	int64_t rank_recv_buf;
	recv_params.memory_type = UCS_MEMORY_TYPE_HOST;
	rank_recv_req = ucp_tag_recv_nbx(w, &rank_recv_buf, sizeof(rank_recv_buf), rank_recv_tag, tag_mask, &recv_params);
	recv_params.memory_type = mca_coll_bkpap_component.ucs_postbuf_memory_type;
	data_req_arr[0] = ucp_tag_recv_nbx(w, recv_buf, recv_byte_count, data_recv_tag, tag_mask, &recv_params);

	stat = bk_poll_ucs_completion(rank_recv_req);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in early p2p poll_completion: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	peer_ep = module->ucp_ep_arr[rank_recv_buf];
	data_req_arr[1] = ucp_tag_send_nbx(peer_ep, send_buf, send_byte_count, data_send_tag, &send_params);

	stat = bk_poll_all_ucs_completion(data_req_arr, 2);
	BKPAP_PROFILE("dplane_tag_early_p2p_wait_end", ompi_comm_rank(comm));
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in early p2p poll_all_completion: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}