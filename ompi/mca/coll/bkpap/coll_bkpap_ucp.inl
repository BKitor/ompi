#include "coll_bkpap.h"

static void _bk_send_cb(void* request, ucs_status_t status, void* args) {
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}

static void _bk_send_cb_noparams(void* request, ucs_status_t status) {
	_bk_send_cb(request, status, NULL);
}

static inline int bk_gpu_op_reduce(ompi_op_t* op, void* source, void* target, size_t full_count, ompi_datatype_t* dtype) {
	if (OPAL_LIKELY(MPI_FLOAT == dtype && MPI_SUM == op)) { // is sum float
		vec_add_float(source, target, full_count);
	}
	else {
		BKPAP_ERROR("Falling back to ompi impl");
		// FULL SEND TO A SEGV !!!
		ompi_op_reduce(op, source, target, full_count, dtype);
	}
	return OMPI_SUCCESS;
}

static inline int mca_coll_bkpap_reduce_local(ompi_op_t* op, void* source, void* target, size_t count, ompi_datatype_t* dtype) {
	// BKPAP_OUTPUT("DBG_REDUCE_IN: source[0]: %.2f, target[0]: %.2f", ((float*)source)[0], ((float*)target)[0]);
	switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
	case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
	case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
		bk_gpu_op_reduce(op, source, target, count, dtype);
		break;
	case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
		ompi_op_reduce(op, source, target, count, dtype);
		break;
	default:
		BKPAP_ERROR("Bad memory type, %d", mca_coll_bkpap_component.bk_postbuf_memory_type);
		return OMPI_ERROR;
		break;
	}

	// BKPAP_OUTPUT("DBG_REDUCE_OUT: source[0]: %.2f, target[0]: %.2f", ((float*)source)[0], ((float*)target)[0]);
	return OMPI_SUCCESS;
}


static inline int mca_coll_bkpap_reduce_dataplane(void* local_buf, struct ompi_datatype_t* dtype, int count, ompi_op_t* op, int num_buffers, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	int dplane_type = mca_coll_bkpap_component.dataplane_type;

	switch (dplane_type) {
	case BKPAP_DATAPLANE_RMA:
		return mca_coll_bkpap_rma_reduce_postbufs(local_buf, dtype, count, op, num_buffers, comm, module);
		break;
	case BKPAP_DATAPLANE_TAG:
		BKPAP_OUTPUT("TAG dataplane reduce, rank: %d, count: %d", ompi_comm_rank(comm), count);
		return mca_coll_bkpap_tag_reduce_postbufs(local_buf, dtype, count, op, num_buffers, comm, module);
		break;
	default:
		BKPAP_ERROR("Bad dataplance type %d, possibilities are {0:RMA, 1:TAG}", dplane_type);
		return OMPI_ERROR;
	}
	return OMPI_ERROR;
}

static inline int mca_coll_bkpap_send_dataplane(const void* buf, struct ompi_datatype_t* dtype, int count, int dest, int slot, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	int dplane_type = mca_coll_bkpap_component.dataplane_type;
	switch (dplane_type) {
	case BKPAP_DATAPLANE_RMA:
		BKPAP_OUTPUT("RMA dataplane send, rank: %d, count: %d, dest: %d, slot: %d", ompi_comm_rank(comm), count, dest, slot);
		return mca_coll_bkpap_rma_send_postbuf(buf, dtype, count, dest, slot, comm, module);
		break;
	case BKPAP_DATAPLANE_TAG:
		BKPAP_OUTPUT("TAG dataplane send, rank: %d, count: %d, dest: %d, slot: %d", ompi_comm_rank(comm), count, dest, slot);
		return mca_coll_bkpap_tag_send_postbuf(buf, dtype, count, dest, slot, comm, module);
		break;
		break;

	default:
		BKPAP_ERROR("Bad dataplance type %d, possibilities are {0:RMA, 1:TAG}", dplane_type);
		return OMPI_ERROR;
		break;
	}

}


static inline ucs_status_t _bk_poll_completion(ucs_status_ptr_t status_ptr) {
	if (UCS_OK == status_ptr) {
		return UCS_OK;
	}

	if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
		BKPAP_ERROR("poll completion returing error %d (%s)", UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		ucp_request_free(status_ptr);
		return UCS_PTR_STATUS(status_ptr);
	}
	ucs_status_t status = UCS_OK;
	mca_coll_bkpap_req_t* req = status_ptr;

	while (UCS_INPROGRESS != UCS_PTR_STATUS(status_ptr) || !req->complete) {
		ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
	}

	status = req->ucs_status;
	req->complete = 0;
	req->ucs_status = UCS_INPROGRESS;
	ucp_request_free(status_ptr);
	return status;
}

static inline ucs_status_t _bk_poll_all_completion(ucs_status_ptr_t* status_ptr_array, int len) {
	ucs_status_t s = UCS_OK;
	for (int i = 0; i < len; i++) {
		if (UCS_OK != (s = _bk_poll_completion(status_ptr_array[i]))) {
			BKPAP_ERROR("UCS error in poll_all_completion: %d (%s)", s, ucs_status_string(s));
			return s;
		}
	}
	return s;
}

static inline ucs_status_t _bk_flush_worker(void) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status;
	status_ptr = ucp_worker_flush_nb(mca_coll_bkpap_component.ucp_worker, 0, _bk_send_cb_noparams);
	status = _bk_poll_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("UCS error in poll_completion: %d (%s)", status, ucs_status_string(status));
	}
	return status;
}
