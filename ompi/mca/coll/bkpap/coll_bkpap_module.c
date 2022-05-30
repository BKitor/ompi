#include "coll_bkpap.h"
#include "opal/util/show_help.h"

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

static void mca_coll_bkpap_module_construct(mca_coll_bkpap_module_t* module) {
	memset(&(module->endof_super), 0, sizeof(*module) - sizeof(module->super));
}

// TODO: Fix issue of hanging on ucp_ep_destroy()
// started after transitioning postbuf size from (postbuf_size * k) to (postbuf_size * (k-1)) 
static void mca_coll_bkpap_module_destruct(mca_coll_bkpap_module_t* module) {

	if (NULL != module->remote_syncstructure) {
		for (int i = 0; i < module->num_syncstructures; i++) {
			mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(module->remote_syncstructure[i]);
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
		free(module->remote_syncstructure);
		module->remote_syncstructure = NULL;
	}

	if (module->rank == 0) {
		for (int i = 0; i < module->num_syncstructures; i++) {
			mca_coll_bkpap_local_syncstruct_t* local_ss_tmp = &(module->local_syncstructure[i]);
			if (NULL != local_ss_tmp->counter_mem_h)
				ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, local_ss_tmp->counter_mem_h);
			if (NULL != local_ss_tmp->arrival_arr_mem_h)
				ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, local_ss_tmp->arrival_arr_mem_h);
		}
		free(module->local_syncstructure);
		module->local_syncstructure = NULL;
	}

	for (int i = 0; i < module->wsize; i++) {
		if (NULL == module->remote_pbuffs.buffer_rkey_arr)break;
		if (NULL == module->remote_pbuffs.buffer_rkey_arr[i])continue;
		ucp_rkey_destroy(module->remote_pbuffs.buffer_rkey_arr[i]);
	}
	free(module->remote_pbuffs.buffer_rkey_arr);
	module->remote_pbuffs.buffer_rkey_arr = NULL;
	free(module->remote_pbuffs.buffer_addr_arr);
	module->remote_pbuffs.buffer_addr_arr = NULL;
	for (int i = 0; i < module->wsize; i++) {
		if (NULL == module->remote_pbuffs.dbell_rkey_arr)break;
		if (NULL == module->remote_pbuffs.dbell_rkey_arr[i]) continue;
		ucp_rkey_destroy(module->remote_pbuffs.dbell_rkey_arr[i]);
	}
	free(module->remote_pbuffs.dbell_rkey_arr);
	module->remote_pbuffs.dbell_rkey_arr = NULL;
	free(module->remote_pbuffs.dbell_addr_arr);
	module->remote_pbuffs.dbell_addr_arr = NULL;

	if (BKPAP_DATAPLANE_RMA == mca_coll_bkpap_component.dataplane_type) {
		if (NULL != module->local_pbuffs.rma.postbuf_h) {
			ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, module->local_pbuffs.rma.postbuf_h);
		}
		module->local_pbuffs.rma.postbuf_h = NULL;
		module->local_pbuffs.rma.postbuf_attrs.address = NULL;
		void* free_local_pbuff = module->local_pbuffs.rma.postbuf_attrs.address;
		if (BKPAP_POSTBUF_MEMORY_TYPE_CUDA == mca_coll_bkpap_component.bk_postbuf_memory_type
			|| BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED == mca_coll_bkpap_component.bk_postbuf_memory_type
			) {
			cudaFree(free_local_pbuff);
		}
		else {
			free(free_local_pbuff);
		}
		if (NULL != module->local_pbuffs.rma.dbell_h) {
			ucp_mem_unmap(mca_coll_bkpap_component.ucp_context, module->local_pbuffs.rma.dbell_h);
		}
		module->local_pbuffs.rma.dbell_h = NULL;
		module->local_pbuffs.rma.dbell_attrs.address = NULL;
	}
	else if (BKPAP_DATAPLANE_TAG == mca_coll_bkpap_component.dataplane_type) {
		if (BKPAP_POSTBUF_MEMORY_TYPE_CUDA == mca_coll_bkpap_component.bk_postbuf_memory_type
			|| BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED == mca_coll_bkpap_component.bk_postbuf_memory_type) {
			cudaFree(module->local_pbuffs.tag.buff_arr);
		}
		else {
			free(module->local_pbuffs.tag.buff_arr);
		}
		module->local_pbuffs.tag.buff_arr = NULL;
	}

	for (int32_t i = 0; i < module->wsize; i++) {
		if (NULL == module->ucp_ep_arr) break;
		if (NULL == module->ucp_ep_arr[i]) continue;
		ucp_ep_destroy(module->ucp_ep_arr[i]);
	}
	free(module->ucp_ep_arr);
	module->ucp_ep_arr = NULL;
	module->ucp_is_initialized = 0;

	if (NULL != module->intra_comm) {
		ompi_comm_free(&(module->intra_comm));
		module->intra_comm = NULL;
	}
	if (NULL != module->inter_comm) {
		ompi_comm_free(&(module->inter_comm));
		module->inter_comm = NULL;
	}

	OBJ_RELEASE(module->fallback_allreduce_module);
	module->fallback_allreduce_module = NULL;
	module->fallback_allreduce = NULL;
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


	return OMPI_SUCCESS;

bkpap_abort_module_enable:
	return OMPI_ERROR;
}

int mca_coll_bkpap_wireup_hier_comms(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS;
	opal_info_t comm_info;
	OBJ_CONSTRUCT(&comm_info, opal_info_t);
	int w_rank = ompi_comm_rank(comm);

	mca_coll_base_module_t* tmp_ar_m = comm->c_coll->coll_allreduce_module;
	mca_coll_base_module_allreduce_fn_t tmp_ar_f = comm->c_coll->coll_allreduce;
	comm->c_coll->coll_allreduce = module->fallback_allreduce;
	comm->c_coll->coll_allreduce_module = module->fallback_allreduce_module;

	ret = opal_info_set(&comm_info, "ompi_comm_coll_preference", "tuned,^bkpap");
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	ret = ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
		&comm_info, &(module->intra_comm));
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	int low_rank = ompi_comm_rank(module->intra_comm);

	ret = opal_info_set(&comm_info, "ompi_comm_coll_preference", "tuned,^bkpap");
	BKPAP_CHK_MPI(ret, bkpap_wireup_hier_comms_err);
	ret = ompi_comm_split_with_info(comm, low_rank, w_rank, &comm_info, &(module->inter_comm), false);
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

	switch (mca_coll_bkpap_component.dataplane_type) {
	case BKPAP_DATAPLANE_RMA:
		ret = mca_coll_bkpap_rma_wireup(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("RMA Wireup Failed, fallingback");
			return ret;
		}
		break;

	case BKPAP_DATAPLANE_TAG:
		ret = mca_coll_bkpap_tag_wireup(bkpap_module, comm);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("TAG Wireup Failed, fallingback");
			return ret;
		}
		break;
	default:
		BKPAP_ERROR("BAD DATAPLANE TYPE SELECTED %d, options are {0:RMA, 1:TAG}", mca_coll_bkpap_component.dataplane_type);
		return OMPI_ERROR;
		break;
	}

	// TODO: refactor into 'bkpap_precalc_ktree/ktree_pipeline/rsa'
	int k = mca_coll_bkpap_component.allreduce_k_value;
	int num_syncstructures = 1;
	size_t counter_arr_len = 0; // log_k(wsize);
	size_t arrival_arr_len = 0; // wsize + wsize/k + wsize/k^2 + wsize/k^3 ...
	int64_t* arrival_arr_offsets_tmp = NULL;
	switch (alg) {
	case BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE:
	case BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE:
		num_syncstructures = 2;
		counter_arr_len = 1;
		arrival_arr_len = ompi_comm_size(comm);
		arrival_arr_offsets_tmp = NULL;
		break;
	case BKPAP_ALLREDUCE_ALG_KTREE:
		for (int i = 1; i < ompi_comm_size(comm); i *= k)
			counter_arr_len++;
		arrival_arr_offsets_tmp = calloc(counter_arr_len, sizeof(*arrival_arr_offsets_tmp));

		for (size_t i = 0; i < counter_arr_len; i++) {
			int k_pow_i = 1;
			for (size_t j = 0; j < i; j++)
				k_pow_i *= k;
			arrival_arr_len += (ompi_comm_size(comm) / k_pow_i);
			if ((i + 1) != counter_arr_len)
				arrival_arr_offsets_tmp[i + 1] = arrival_arr_offsets_tmp[i] + (ompi_comm_size(comm) / k_pow_i);
		}
		break;
	case BKPAP_ALLREDUCE_ALG_RSA:
		num_syncstructures = 1;
		counter_arr_len = 1;
		arrival_arr_len = ompi_comm_size(comm);
		arrival_arr_offsets_tmp = NULL;
		break;
	case BKPAP_ALLREDUCE_BASE_RSA_GPU:
		num_syncstructures = 0;
		counter_arr_len = 0;
		arrival_arr_len = 0;
		arrival_arr_offsets_tmp = NULL;
		break;
	default:
		BKPAP_ERROR("Bad algorithms specified, failed to setup syncstructure");
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

	return ret;
}