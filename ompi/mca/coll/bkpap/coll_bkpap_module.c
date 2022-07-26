#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"
#include "opal/util/show_help.h"

static void mca_coll_bkpap_module_construct(mca_coll_bkpap_module_t* bkpap_module) {
	memset(&(bkpap_module->endof_super), 0, sizeof(*bkpap_module) - sizeof(bkpap_module->super));
}

// TODO: Fix issue of hanging on ucp_ep_destroy()
// started after transitioning postbuf size from (postbuf_size * k) to (postbuf_size * (k-1)) 
static void mca_coll_bkpap_module_destruct(mca_coll_bkpap_module_t* bkpap_module) {
	cudaStreamDestroy(bkpap_module->bk_cs[0]);
	cudaStreamDestroy(bkpap_module->bk_cs[1]);
	cudaFreeHost(bkpap_module->host_pinned_buf);

	bkpap_finalize_mempool(bkpap_module);

	if (NULL != bkpap_module->remote_syncstructure) {
		for (int i = 0; i < bkpap_module->num_syncstructures; i++) {
			mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(bkpap_module->remote_syncstructure[i]);
			free(remote_ss_tmp->ss_arrival_arr_offsets);
			remote_ss_tmp->ss_arrival_arr_offsets = NULL;
			if (NULL != remote_ss_tmp->arrival_arr_rkey)
				ucp_rkey_destroy(remote_ss_tmp->arrival_arr_rkey);
			remote_ss_tmp->arrival_arr_rkey = NULL;
			remote_ss_tmp->arrival_arr_addr = 0;
			if (NULL != remote_ss_tmp->counter_rkey)
				ucp_rkey_destroy(remote_ss_tmp->counter_rkey);
			remote_ss_tmp->counter_rkey = NULL;
			remote_ss_tmp->counter_addr = 0;
		}
		free(bkpap_module->remote_syncstructure);
		bkpap_module->remote_syncstructure = NULL;
	}

	if (bkpap_module->rank == 0) {
		for (int i = 0; i < bkpap_module->num_syncstructures; i++) {
			mca_coll_bkpap_local_syncstruct_t* local_ss_tmp = &(bkpap_module->local_syncstructure[i]);
			if (NULL != local_ss_tmp->counter_mem_h)
				ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, local_ss_tmp->counter_mem_h);
			if (NULL != local_ss_tmp->arrival_arr_mem_h)
				ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, local_ss_tmp->arrival_arr_mem_h);
		}
		free(bkpap_module->local_syncstructure);
		bkpap_module->local_syncstructure = NULL;
	}

	if (BKPAP_DPLANE_RMA == bkpap_module->dplane_t){
		mca_coll_bkpap_rma_dplane_destroy(&bkpap_module->dplane.rma, bkpap_module);
	}

	else if (BKPAP_DPLANE_TAG == mca_coll_bkpap_component.dplane_t) {
		mca_coll_bkpap_tag_dplane_destroy(&bkpap_module->dplane.tag);
	}

	for (int32_t i = 0; i < bkpap_module->wsize; i++) {
		if (NULL == bkpap_module->ucp_ep_arr) break;
		if (NULL == bkpap_module->ucp_ep_arr[i]) continue;
		ucp_ep_destroy(bkpap_module->ucp_ep_arr[i]);
	}
	free(bkpap_module->ucp_ep_arr);
	bkpap_module->ucp_ep_arr = NULL;
	bkpap_module->ucp_is_initialized = 0;

	if (NULL != bkpap_module->intra_comm) {
		ompi_comm_free(&(bkpap_module->intra_comm));
		bkpap_module->intra_comm = NULL;
	}
	if (NULL != bkpap_module->inter_comm) {
		ompi_comm_free(&(bkpap_module->inter_comm));
		bkpap_module->inter_comm = NULL;
	}

	OBJ_RELEASE(bkpap_module->fallback_allreduce_module);
	bkpap_module->fallback_allreduce_module = NULL;
	bkpap_module->fallback_allreduce = NULL;
}

OBJ_CLASS_INSTANCE(mca_coll_bkpap_module_t, mca_coll_base_module_t,
	mca_coll_bkpap_module_construct, mca_coll_bkpap_module_destruct);

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority) {
	mca_coll_bkpap_module_t* bkpap_module;

	bkpap_module = OBJ_NEW(mca_coll_bkpap_module_t);
	if (NULL == bkpap_module) {
		return NULL;
	}

	*priority = mca_coll_bkpap_component.priority;

	bkpap_module->super.coll_module_enable = mca_coll_bkpap_module_enable;
	bkpap_module->super.coll_allreduce = mca_coll_bkpap_allreduce;

	return &(bkpap_module->super);
}

int mca_coll_bkpap_module_enable(mca_coll_base_module_t* module, struct ompi_communicator_t* comm) {
	mca_coll_bkpap_module_t* bkpap_module = (mca_coll_bkpap_module_t*)module;
	mca_coll_base_comm_t* data = NULL;

	// check for allgather/allgaterv, needed for setup when echangaing ucp data
	if (NULL == comm->c_coll->coll_allgather_module || NULL == comm->c_coll->coll_allgatherv_module) {
		opal_show_help("help-mpi-coll-bkpap.txt", "missing collective", true,
			ompi_process_info.nodename,
			mca_coll_bkpap_component.priority, "allreduce");
		return OMPI_ERR_NOT_FOUND;
	}

	// check for allreduce and retain fallback
	if (NULL == comm->c_coll->coll_allreduce_module) {
		opal_show_help("help-mpi-coll-bkpap.txt", "missing collective", true,
			ompi_process_info.nodename,
			mca_coll_bkpap_component.priority, "allreduce");
		return OMPI_ERR_NOT_FOUND;
	}
	else {
		bkpap_module->fallback_allreduce_module = comm->c_coll->coll_allreduce_module;
		bkpap_module->fallback_allreduce = comm->c_coll->coll_allreduce;
		OBJ_RETAIN(bkpap_module->fallback_allreduce_module);
	}

	if (NULL == bkpap_module->super.base_data) {
		data = OBJ_NEW(mca_coll_base_comm_t);
		BKPAP_CHK_MALLOC(data, bkpap_abort_module_enable);
		data->cached_ntree = NULL;
		data->cached_bintree = NULL;
		data->cached_bmtree = NULL;
		data->cached_in_order_bmtree = NULL;
		data->cached_kmtree = NULL;
		data->cached_chain = NULL;
		data->cached_pipeline = NULL;
		data->cached_in_order_bintree = NULL;
		bkpap_module->super.base_data = data;
	}
	
	bkpap_module->dplane_t = mca_coll_bkpap_component.dplane_t;
	bkpap_module->dplane_mem_t = mca_coll_bkpap_component.dplane_mem_t;

	return OMPI_SUCCESS;

bkpap_abort_module_enable:
	return OMPI_ERROR;
}

int mca_coll_bkpap_wireup_hier_comms(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS;
	opal_info_t comm_info;
	OBJ_CONSTRUCT(&comm_info, opal_info_t);
	int w_rank = ompi_comm_rank(comm);

	mca_coll_base_module_t* tmp_ar_m = comm->c_coll->coll_allreduce_module;
	mca_coll_base_module_allreduce_fn_t tmp_ar_f = comm->c_coll->coll_allreduce;
	comm->c_coll->coll_allreduce = bkpap_module->fallback_allreduce;
	comm->c_coll->coll_allreduce_module = bkpap_module->fallback_allreduce_module;

	ret = opal_info_set(&comm_info, "ompi_comm_coll_preference", "tuned,^bkpap");
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	ret = ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
		&comm_info, &(bkpap_module->intra_comm));
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	int low_rank = ompi_comm_rank(bkpap_module->intra_comm);

	ret = opal_info_set(&comm_info, "ompi_comm_coll_preference", "tuned,^bkpap");
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	ret = ompi_comm_split_with_info(comm, low_rank, w_rank, &comm_info, &(bkpap_module->inter_comm), false);
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);

	OBJ_DESTRUCT(&comm_info);

bkpap_wireup_hier_comms_err:
	comm->c_coll->coll_allreduce = tmp_ar_f;
	comm->c_coll->coll_allreduce_module = tmp_ar_m;
	return ret;
}

int mca_coll_bkpap_lazy_init_module_ucx(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm, int alg) {
	int ret = OMPI_SUCCESS;

	if (OPAL_UNLIKELY(NULL == mca_coll_bkpap_component.ucp_context)) {
		ret = mca_coll_bkpap_init_ucx(mca_coll_bkpap_component.enable_threads);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("UCX Initialization Failed");
			return ret;
		}
		BKPAP_OUTPUT("UCX Initialization SUCCESS");
	}

	ret = mca_coll_bkpap_wireup_endpoints(bkpap_module, comm);
	if (OMPI_SUCCESS != ret) {
		BKPAP_ERROR("Endpoint Wireup Failed, fallingback");
		return ret;
	}

	switch (bkpap_module->dplane_t) {
	case BKPAP_DPLANE_RMA:
		ret = mca_coll_bkpap_rma_dplane_wireup(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("RMA Wireup Failed, fallingback");
			return ret;
		}
		break;

	case BKPAP_DPLANE_TAG:
		ret = mca_coll_bkpap_tag_dplane_wireup(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("TAG Wireup Failed, fallingback");
			return ret;
		}
		break;
	default:
		BKPAP_ERROR("BAD DATAPLANE TYPE SELECTED %d, options are {0:RMA, 1:TAG}", bkpap_module->dplane_t);
		return OMPI_ERROR;
		break;
	}

	if (BKPAP_DPLANE_MEM_TYPE_HOST != bkpap_module->dplane_mem_t) {
		cudaStreamCreate(&bkpap_module->bk_cs[0]);
		cudaStreamCreate(&bkpap_module->bk_cs[1]);
		cudaMallocHost(&bkpap_module->host_pinned_buf, mca_coll_bkpap_component.postbuff_size);
	}

	int num_syncstructures = 1;
	size_t counter_arr_len = 0; // log_k(wsize);
	size_t arrival_arr_len = 0; // wsize + wsize/k + wsize/k^2 + wsize/k^3 ...
	int64_t* arrival_arr_offsets_tmp = NULL;
	switch (alg) {
	case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
	case BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE:
	case BKPAP_ALLREDUCE_ALG_KTREE:
		BKPAP_ERROR("KTREE algorithms were removed");
		return OMPI_ERR_NOT_SUPPORTED;
		break;
	case BKPAP_ALLREDUCE_ALG_RSA:
	case BKPAP_ALLREDUCE_ALG_BINOMIAL:
	case BKPAP_ALLREDUCE_ALG_CHAIN:
		num_syncstructures = 1;
		counter_arr_len = 1;
		arrival_arr_len = ompi_comm_size(comm);
		arrival_arr_offsets_tmp = NULL;
		break;
	case BKPAP_ALLREDUCE_ALG_BASE_RSA_GPU:
		num_syncstructures = 0;
		counter_arr_len = 0;
		arrival_arr_len = 0;
		arrival_arr_offsets_tmp = NULL;
		break;
	default:
		BKPAP_ERROR("Bad algorithm %d specified, failed to setup syncstructure", alg);
		return OMPI_ERROR;
		break;
	}

	ret = mca_coll_bkpap_wireup_syncstructure(counter_arr_len, arrival_arr_len, num_syncstructures, bkpap_module, comm);
	if (OMPI_SUCCESS != ret) {
		BKPAP_ERROR("Syncstructure Wireup Failed, fallingback");
		return ret;
	}
	for (int i = 0; i < num_syncstructures; i++) {
		bkpap_module->remote_syncstructure[i].ss_counter_len = counter_arr_len;
		bkpap_module->remote_syncstructure[i].ss_arrival_arr_len = arrival_arr_len;
		bkpap_module->remote_syncstructure[i].ss_arrival_arr_offsets = arrival_arr_offsets_tmp;
	}
	bkpap_module->num_syncstructures = num_syncstructures;
	arrival_arr_offsets_tmp = NULL;
	
	if(BKPAP_ALLREDUCE_ALG_CHAIN == alg){
		bk_launch_background_thread();
	}

	return ret;
}

int bkpap_init_mempool(mca_coll_bkpap_module_t* bkpap_module) {
	for (int i = 0; i < BKPAP_DPLANE_MEM_TYPE_COUNT; i++) {
		bkpap_mempool_t* m = &bkpap_module->mempool[i];
		m->head = NULL;
		m->memtype = i;
	}
	return OMPI_SUCCESS;
}

int bkpap_finalize_mempool(mca_coll_bkpap_module_t* bkpap_module) {
	for (int i = 0; i < BKPAP_DPLANE_MEM_TYPE_COUNT; i++) {
		bkpap_mempool_t* m = &bkpap_module->mempool[i];
		bkpap_mempool_buf_t* b = m->head;
		while (NULL != b) {
			bkpap_mempool_buf_t* bn = b->next;
			if (b->allocated)
				BKPAP_ERROR("Freeing mempoolbuf marked allocated, migh havem mem-leak");
			bkpap_mempool_destroy_buf(b, m);
			b = bn;
		}

	}
	return OMPI_SUCCESS;
}

// returns error if runs out of space
int bk_fill_array_str_ld(size_t arr_len, int64_t* arr, size_t str_limit, char* out_str) {
	if (str_limit < 3) return OMPI_ERROR;
	char tmp[16] = { "\0" };
	*out_str = '\0';
	strcat(out_str, "[");
	for (size_t i = 0; i < arr_len; i++) {
		if (i == 0)
			sprintf(tmp, " %ld", arr[i]);
		else
			sprintf(tmp, ", %ld", arr[i]);

		if (strlen(tmp) > (str_limit - strlen(out_str)))
			return OMPI_ERROR;

		strcat(out_str, tmp);
	}

	if (strlen(out_str) > (str_limit + 1))
		return OMPI_ERROR;
	strcat(out_str, " ]");
	return OMPI_SUCCESS;
}