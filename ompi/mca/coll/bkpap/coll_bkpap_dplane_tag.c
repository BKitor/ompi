#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"

#include "opal/cuda/common_cuda.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop


int coll_bkpap_tag_sendrecv(void* sbuf, int send_count, void* rbuf,
	int recv_count, struct ompi_datatype_t* dtype,
	int peer_rank, int64_t tag, int64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t status_array[2];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_size = (send_count * extent), recv_size = (recv_count * extent);
	ucp_request_param_t send_params, recv_params;
	bk_fill_tag_send_params(&send_params, bkpap_module);
	bk_fill_tag_recv_params(&recv_params, bkpap_module);
	ucp_ep_h peer_ep = bkpap_module->ucp_ep_arr[peer_rank];
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
int coll_bkpap_tag_send_to_early(void* send_buf, int send_count,
	struct ompi_datatype_t* dtype, int peer_rank, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module) {

	ucp_ep_h ep = bkpap_module->ucp_ep_arr[peer_rank];
	ucs_status_ptr_t data_send_ptr;
	ucs_status_t stat;
	ptrdiff_t extent, lb;
	ucp_request_param_t send_params;

	bk_fill_tag_send_params(&send_params, bkpap_module);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_byte_count = (send_count * extent);

	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	data_send_ptr = ucp_tag_send_nbx(ep, send_buf, send_byte_count, tag, &send_params);
	stat = bk_poll_ucs_completion(data_send_ptr);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("tag_send_to_early data send failed: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}

// pair with recv_from_early, recv rank->send data
int coll_bkpap_tag_send_to_late(void* send_buf, int send_count,
	struct ompi_datatype_t* dtype, int64_t tag, int64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h ep;
	ucs_status_ptr_t rank_recv_req, data_send_req;
	ucs_status_t stat;
	ucp_request_param_t send_params, recv_params;
	ptrdiff_t extent, lb;

	bk_fill_tag_send_params(&send_params, bkpap_module);
	bk_fill_tag_recv_params(&recv_params, bkpap_module);
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t send_byte_count = (send_count * extent);

	recv_params.memory_type = UCS_MEMORY_TYPE_HOST;
	int64_t rank_recv_buf;
	tag = BK_BINOMIAL_TAG_SET_RANK(tag);
	rank_recv_req = ucp_tag_recv_nbx(w, &rank_recv_buf, sizeof(rank_recv_buf), tag, tag_mask, &recv_params);

	if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(rank_recv_req))) {
		stat = UCS_PTR_STATUS(rank_recv_buf);
		return OMPI_ERROR;
	}

	stat = bk_poll_ucs_completion(rank_recv_req);

	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		return OMPI_ERROR;
	}

	ep = bkpap_module->ucp_ep_arr[rank_recv_buf];
	tag = BK_BINOMIAL_TAG_SET_DATA(tag);
	data_send_req = ucp_tag_send_nbx(ep, send_buf, send_byte_count, tag, &send_params);
	stat = bk_poll_ucs_completion(data_send_req);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}

// pair with send_to_late, send rank->recv data
int coll_bkpap_tag_recv_from_early(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, int peer_rank, int64_t tag,
	int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucs_status_ptr_t stat_ptr_arr[2];
	ucs_status_t stat;
	ucp_request_param_t send_params, recv_params;
	ucp_ep_h ep = bkpap_module->ucp_ep_arr[peer_rank];
	ptrdiff_t extent, lb;

	bk_fill_tag_send_params(&send_params, bkpap_module);
	bk_fill_tag_recv_params(&recv_params, bkpap_module);
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
int coll_bkpap_tag_recv_from_late(void* recv_buf, int recv_count,
	struct ompi_datatype_t* dtype, int64_t tag, int64_t tag_mask,
	ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucs_status_ptr_t stat_ptr;
	ucs_status_t stat;
	ucp_request_param_t recv_params;
	ptrdiff_t extent, lb;

	bk_fill_tag_recv_params(&recv_params, bkpap_module);
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

int coll_bkpap_tag_sendrecv_from_early(void* send_buf, int send_count,
	void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t status_req_arr[3];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h peer_ep = bkpap_module->ucp_ep_arr[peer_rank];
	ucp_request_param_t send_params, recv_params;

	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent), send_byte_count = (send_count * extent);
	bk_fill_tag_send_params(&send_params, bkpap_module);
	bk_fill_tag_recv_params(&recv_params, bkpap_module);
	ucp_tag_t rank_send_tag = BK_RSA_SET_TAG_TYPE_RANK(tag);
	ucp_tag_t data_send_tag = BK_RSA_SET_TAG_TYPE_DATA(tag), data_recv_tag = BK_RSA_SET_TAG_TYPE_DATA(tag);

	int64_t rank_send_buf = ompi_comm_rank(comm);
	send_params.memory_type = UCS_MEMORY_TYPE_HOST;
	status_req_arr[0] = ucp_tag_send_nbx(peer_ep, &rank_send_buf, sizeof(rank_send_buf), rank_send_tag, &send_params);
	send_params.memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
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

int coll_bkpap_tag_sendrecv_from_late(void* send_buf, int send_count,
	void* recv_buf, int recv_count, struct ompi_datatype_t* dtype,
	int64_t tag, int64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {

	ucs_status_t stat = UCS_OK;
	ucs_status_ptr_t rank_recv_req, data_req_arr[2];
	ucp_worker_h w = mca_coll_bkpap_component.ucp_worker;
	ucp_ep_h peer_ep;
	ucp_request_param_t send_params, recv_params;

	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);
	size_t recv_byte_count = (recv_count * extent), send_byte_count = (send_count * extent);
	bk_fill_tag_send_params(&send_params, bkpap_module);
	bk_fill_tag_recv_params(&recv_params, bkpap_module);
	ucp_tag_t rank_recv_tag = BK_RSA_SET_TAG_TYPE_RANK(tag);
	ucp_tag_t data_send_tag = BK_RSA_SET_TAG_TYPE_DATA(tag), data_recv_tag = BK_RSA_SET_TAG_TYPE_DATA(tag);

	int64_t rank_recv_buf;
	recv_params.memory_type = UCS_MEMORY_TYPE_HOST;
	rank_recv_req = ucp_tag_recv_nbx(w, &rank_recv_buf, sizeof(rank_recv_buf), rank_recv_tag, tag_mask, &recv_params);
	recv_params.memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
	data_req_arr[0] = ucp_tag_recv_nbx(w, recv_buf, recv_byte_count, data_recv_tag, tag_mask, &recv_params);

	stat = bk_poll_ucs_completion(rank_recv_req);
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in early p2p poll_completion: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	peer_ep = bkpap_module->ucp_ep_arr[rank_recv_buf];
	data_req_arr[1] = ucp_tag_send_nbx(peer_ep, send_buf, send_byte_count, data_send_tag, &send_params);

	stat = bk_poll_all_ucs_completion(data_req_arr, 2);
	BKPAP_PROFILE("dplane_tag_early_p2p_wait_end", ompi_comm_rank(comm));
	if (OPAL_UNLIKELY(UCS_OK != stat)) {
		BKPAP_ERROR("error in early p2p poll_all_completion: %d (%s)", stat, ucs_status_string(stat));
		return OMPI_ERROR;
	}
	return OMPI_SUCCESS;
}

int coll_bkpap_tag_reset_late_recv_buf(mca_coll_bkpap_module_t* bkpap_module) {
	return OMPI_SUCCESS;
}

int mca_coll_bkpap_tag_dplane_wireup(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm) {

	int ret = OMPI_SUCCESS, alg = mca_coll_bkpap_component.allreduce_alg, mem_type = bkpap_module->dplane_mem_t;
	mca_coll_bkpap_tag_dplane_t* tag_dplane = &bkpap_module->dplane.tag;

	int num_postbufs = (alg == BKPAP_ALLREDUCE_ALG_BASE_RSA_GPU) ? 0 : 1;
	size_t mapped_postbuf_size = mca_coll_bkpap_component.postbuff_size * num_postbufs;
	void* mapped_postbuf = NULL;

	ret = bk_alloc_dplane_mem_t(&mapped_postbuf, mapped_postbuf_size, bkpap_module->dplane_mem_t);
	BKPAP_CHK_MPI_MSG_LBL(ret, "Alloc bk_pbufft failed", bk_abort_tag_wireup);

	tag_dplane->buff_arr = mapped_postbuf;
	tag_dplane->buff_size = mca_coll_bkpap_component.postbuff_size;
	tag_dplane->mem_type = mem_type;

	bkpap_module->dplane_ftbl.send_to_early = coll_bkpap_tag_send_to_early;
	bkpap_module->dplane_ftbl.send_to_late = coll_bkpap_tag_send_to_late;
	bkpap_module->dplane_ftbl.recv_from_early = coll_bkpap_tag_recv_from_early;
	bkpap_module->dplane_ftbl.recv_from_late = coll_bkpap_tag_recv_from_late;
	bkpap_module->dplane_ftbl.sendrecv_from_early = coll_bkpap_tag_sendrecv_from_early;
	bkpap_module->dplane_ftbl.sendrecv_from_late = coll_bkpap_tag_sendrecv_from_late;
	bkpap_module->dplane_ftbl.sendrecv = coll_bkpap_tag_sendrecv;
	bkpap_module->dplane_ftbl.reset_late_recv_buf = coll_bkpap_tag_reset_late_recv_buf;

	return ret;

bk_abort_tag_wireup:
	BKPAP_ERROR("Tag Wireup Error");
	return OMPI_ERROR;
}

void mca_coll_bkpap_tag_dplane_destroy(mca_coll_bkpap_tag_dplane_t* tag_dplane) {
	// if (NULL != tag_dplane->buff_arr)
	// 	bk_free_pbufft(tag_dplane->buff_arr, tag_dplane->mem_type);
}