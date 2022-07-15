#include "coll_bkpap.h"


static void _bk_send_cb(void* request, ucs_status_t status, void* args) {
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}

static void _bk_recv_cb(void* request, ucs_status_t status, const ucp_tag_recv_info_t* tag_info, void* user_data) {
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}

static inline ucs_status_t bk_poll_ucs_completion(ucs_status_ptr_t status_ptr) {
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

static inline ucs_status_t bk_poll_all_ucs_completion(ucs_status_ptr_t* status_ptr_array, int len) {
	ucs_status_t s = UCS_OK;
	for (int i = 0; i < len; i++) {
		if(UCS_OK == UCS_PTR_STATUS(status_ptr_array[i])) continue;
		if (UCS_OK != (s = bk_poll_ucs_completion(status_ptr_array[i]))) {
			BKPAP_ERROR("UCS error in poll_all_completion: %d (%s)", s, ucs_status_string(s));
			return s;
		}
	}
	return s;
}

static inline int bk_ompi_request_wait_all(ompi_request_t** request_arr, int req_arr_len) {
    int tmp_is_completed;
    ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    while (!tmp_is_completed) {
        ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
        ompi_request_test_all(req_arr_len, request_arr, &tmp_is_completed, MPI_STATUSES_IGNORE);
    }
    return OMPI_SUCCESS;
}

static inline ucs_memory_type_t bk_convert_memt_bk2ucs(bkpap_dplane_mem_t bk_memt) {
    switch (bk_memt) {
    case BKPAP_DPLANE_MEM_TYPE_CUDA:
        return UCS_MEMORY_TYPE_CUDA;
        break;
    case BKPAP_DPLANE_MEM_TYPE_CUDA_MANAGED:
        return UCS_MEMORY_TYPE_CUDA_MANAGED;
        break;
    case BKPAP_DPLANE_MEM_TYPE_HOST:
        return UCS_MEMORY_TYPE_HOST;
        break;
    default:
        BKPAP_ERROR("bad memory type %d", bk_memt);
        break;
    }
    return -1;
}

static inline void bk_fill_amo_params(ucp_request_param_t* req_params, void* repl_buf_addr) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE
		| UCP_OP_ATTR_FIELD_REPLY_BUFFER;
	req_params->cb.send = _bk_send_cb;
	req_params->user_data = NULL;
	req_params->memory_type = UCS_MEMORY_TYPE_HOST;
	req_params->datatype = ucp_dt_make_contig(8); // 64bit 
	req_params->reply_buffer = repl_buf_addr;
}

static inline void bk_fill_put_params(ucp_request_param_t* req_params, mca_coll_bkpap_module_t* bkpap_module) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.send = _bk_send_cb;
	req_params->memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
	req_params->user_data = NULL;
	req_params->datatype = ucp_dt_make_contig(1);
}

static inline void bk_fill_tag_recv_params(ucp_request_param_t* req_params, mca_coll_bkpap_module_t* bkpap_module) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.recv = _bk_recv_cb;
	req_params->memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
	req_params->user_data = NULL;
	req_params->datatype = ucp_dt_make_contig(1);
}

static inline void bk_fill_tag_send_params(ucp_request_param_t* req_params, mca_coll_bkpap_module_t* bkpap_module) {
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.send = _bk_send_cb;
	req_params->user_data = NULL;
	req_params->memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
	req_params->datatype = ucp_dt_make_contig(1);
}

static inline void bk_fill_get_params(ucp_request_param_t* req_params, mca_coll_bkpap_module_t* bkpap_module){
	req_params->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK
		| UCP_OP_ATTR_FIELD_USER_DATA
		| UCP_OP_ATTR_FIELD_MEMORY_TYPE
		| UCP_OP_ATTR_FIELD_DATATYPE;
	req_params->cb.send = _bk_send_cb;
	req_params->user_data = NULL;
	req_params->memory_type = bk_convert_memt_bk2ucs(bkpap_module->dplane_mem_t);
	req_params->datatype = ucp_dt_make_contig(1);
	
}