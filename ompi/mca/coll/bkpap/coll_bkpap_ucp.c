#include "coll_bkpap.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/pml/pml.h"

void mca_coll_bkpap_req_init(void* request) {
	mca_coll_bkpap_req_t* r = request;
	r->ucs_status = UCS_INPROGRESS;
	r->complete = 0;
}

static void _bk_send_cb(void* request, ucs_status_t status, void* args) {
	mca_coll_bkpap_req_t* req = request;
	req->ucs_status = status;
	req->complete = 1;
}

static void _bk_send_cb_noparams(void* request, ucs_status_t status) {
	_bk_send_cb(request, status, NULL);
}

static inline ucs_status_t _bk_poll_completion(ucs_status_ptr_t status_ptr) {
	if (status_ptr == UCS_OK) {
		return UCS_OK;
	}

	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("poll completion returing error %d (%s)", UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
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

static inline ucs_status_t _bk_flush_worker(void) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status;
	status_ptr = ucp_worker_flush_nb(mca_coll_bkpap_component.ucp_worker, 0, _bk_send_cb_noparams);
	status = _bk_poll_completion(status_ptr);
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker flush failed");
	}
	return status;
}

int mca_coll_bkpap_init_ucx(int enable_mpi_threads) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_init_ucp_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in endpoint wireup failed"); ret = OMPI_ERROR; goto bkpap_init_ucp_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in endpoint wireup failed"); goto bkpap_init_ucp_err;}
	int ret = OMPI_SUCCESS;
	ucp_params_t ucp_params;
	ucp_worker_params_t worker_params;
	ucp_config_t* config;
	ucs_status_t status;

	BKPAP_MSETZ(ucp_params);
	BKPAP_MSETZ(worker_params);

	status = ucp_config_read("MPI", NULL, &config);
	if (UCS_OK != status) {
		return OMPI_ERROR;
	}

	ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
		UCP_PARAM_FIELD_REQUEST_SIZE |
		UCP_PARAM_FIELD_REQUEST_INIT |
		UCP_PARAM_FIELD_MT_WORKERS_SHARED |
		UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
	ucp_params.features = UCP_FEATURE_AMO64 | UCP_FEATURE_RMA;
	ucp_params.request_size = sizeof(mca_coll_bkpap_req_t);
	ucp_params.request_init = mca_coll_bkpap_req_init;
	ucp_params.mt_workers_shared = 0; /* we do not need mt support for context
									 since it will be protected by worker */
	ucp_params.estimated_num_eps = ompi_proc_world_size();

#if HAVE_DECL_UCP_PARAM_FIELD_ESTIMATED_NUM_PPN
	ucp_params.estimated_num_ppn = opal_process_info.num_local_peers + 1;
	ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
#endif

	status = ucp_init(&ucp_params, config, &mca_coll_bkpap_component.ucp_context);
	ucp_config_release(config);
	_BKPAP_CHK_UCP(status);

	worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = (enable_mpi_threads == MPI_THREAD_SINGLE) ? UCS_THREAD_MODE_SINGLE : UCS_THREAD_MODE_MULTI;
	status = ucp_worker_create(mca_coll_bkpap_component.ucp_context, &worker_params, &mca_coll_bkpap_component.ucp_worker);
	_BKPAP_CHK_UCP(status);

	status = ucp_worker_get_address(
		mca_coll_bkpap_component.ucp_worker,
		&mca_coll_bkpap_component.ucp_worker_addr,
		&mca_coll_bkpap_component.ucp_worker_addr_len
	);
	_BKPAP_CHK_UCP(status);

bkpap_init_ucp_err:
	return ret;
#undef _BKPAP_CHK_MALLOC
#undef _BKPAP_CHK_UCP
#undef _BKPAP_CHK_MPI
}

// might want to make static inline, and move to header
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_ep_wireup_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in endpoint wireup failed"); ret = OMPI_ERROR; goto bkpap_ep_wireup_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in endpoint wireup failed"); goto bkpap_ep_wireup_err;}
	ucs_status_t status = UCS_OK;
	ucp_ep_params_t ep_params;
	int ret = OMPI_SUCCESS;
	int mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	int* agv_count_arr = NULL, * agv_displ_arr = NULL;
	size_t* remote_addr_len_buf = NULL;
	uint8_t* agv_remote_addr_recv_buf = NULL;

	BKPAP_MSETZ(ep_params);

	agv_count_arr = calloc(mpi_size, sizeof(*agv_count_arr));
	_BKPAP_CHK_MALLOC(agv_count_arr);
	agv_displ_arr = calloc(mpi_size, sizeof(*agv_displ_arr));
	_BKPAP_CHK_MALLOC(agv_displ_arr);
	remote_addr_len_buf = calloc(mpi_size, sizeof(*remote_addr_len_buf));
	_BKPAP_CHK_MALLOC(remote_addr_len_buf);

	// gather address lengths
	ret = comm->c_coll->coll_allgather(
		&mca_coll_bkpap_component.ucp_worker_addr_len, 1, MPI_LONG_LONG,
		remote_addr_len_buf, 1, MPI_LONG_LONG, comm,
		comm->c_coll->coll_allgather_module
	);
	_BKPAP_CHK_MPI(ret);

	// setup allgatherv count/displs
	size_t total_addr_buff_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = total_addr_buff_size;
		agv_count_arr[i] = remote_addr_len_buf[i];
		total_addr_buff_size += remote_addr_len_buf[i];
	}
	agv_remote_addr_recv_buf = malloc(total_addr_buff_size);
	_BKPAP_CHK_MALLOC(agv_remote_addr_recv_buf);
	memset(agv_remote_addr_recv_buf, 0, total_addr_buff_size);

	// allgatherv the ucp_addr_t
	ret = comm->c_coll->coll_allgatherv(
		mca_coll_bkpap_component.ucp_worker_addr, mca_coll_bkpap_component.ucp_worker_addr_len, MPI_BYTE,
		agv_remote_addr_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module
	);
	_BKPAP_CHK_MPI(ret);

	// for loop to populate ep_arr
	// module->ucp_ep_arr = calloc(mpi_size, sizeof(*module->ucp_ep_arr));
	module->ucp_ep_arr = calloc(mpi_size, sizeof(ucp_ep_h));
	_BKPAP_CHK_MALLOC(module->ucp_ep_arr);
	ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
	module->wsize = mpi_size;
	module->rank = mpi_rank;
	for (int i = 0; i < mpi_size; i++) {
		// if (i == mpi_rank)continue;
		ep_params.address = (void*)(agv_remote_addr_recv_buf + agv_displ_arr[i]);
		status = ucp_ep_create(mca_coll_bkpap_component.ucp_worker, &ep_params, &module->ucp_ep_arr[i]);
		_BKPAP_CHK_UCP(status);
	}


	BKPAP_OUTPUT("ucp endpoints wireup SUCCESS");
bkpap_ep_wireup_err:
	free(agv_remote_addr_recv_buf);
	free(agv_count_arr);
	free(agv_displ_arr);
	free(remote_addr_len_buf);

	return ret;
#undef _BKPAP_CHK_MALLOC
#undef _BKPAP_CHK_UCP
#undef _BKPAP_CHK_MPI
}

// TODO: Number of allocated postsbufs should be determined by the alg
int mca_coll_bkpap_wireup_postbuffs(int alg, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_remotepostbuf_wireup_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in postbuf wireup failed"); ret = OMPI_ERROR; goto bkpap_remotepostbuf_wireup_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in postbuf wireup failed"); goto bkpap_remotepostbuf_wireup_err;}
	int ret = OMPI_SUCCESS, mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	ucs_status_t status = UCS_OK;
	ucp_mem_map_params_t mem_map_params;
	void* postbuf_rkey_buffer = NULL, * dbell_rkey_buffer = NULL;
	size_t postbuf_rkey_buffer_size, dbell_rkey_buffer_size, * postbuf_rkey_size_arr = NULL, * dbell_rkey_size_arr = NULL;
	int* agv_displ_arr = NULL, * agv_count_arr = NULL;
	uint8_t* agv_rkey_recv_buf = NULL;
	size_t agv_rkey_recv_buf_size = 0;

	BKPAP_MSETZ(mem_map_params);
	BKPAP_MSETZ(module->local_postbuf_attrs);
	BKPAP_MSETZ(module->local_dbell_attrs);

	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_FLAGS;
	mem_map_params.address = NULL;
	mem_map_params.length = mca_coll_bkpap_component.postbuff_size * (mca_coll_bkpap_component.allreduce_k_value - 1);
	mem_map_params.flags = UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_NONBLOCK;

	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_postbuf_h);
	_BKPAP_CHK_UCP(status);

	mem_map_params.length = sizeof(int64_t) * (mca_coll_bkpap_component.allreduce_k_value - 1);
	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_dbell_h);
	_BKPAP_CHK_UCP(status);

	module->local_postbuf_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
	status = ucp_mem_query(module->local_postbuf_h, &module->local_postbuf_attrs);
	_BKPAP_CHK_UCP(status);

	module->local_dbell_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
	status = ucp_mem_query(module->local_dbell_h, &module->local_dbell_attrs);
	_BKPAP_CHK_UCP(status);
	int64_t* dbells = module->local_dbell_attrs.address;
	for (int i = 0; i < (mca_coll_bkpap_component.allreduce_k_value - 1); i++)
		dbells[i] = BKPAP_DBELL_UNSET;
	dbells = NULL;


	module->remote_pbuffs.buffer_addr_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.buffer_addr_arr));
	_BKPAP_CHK_MALLOC(module->remote_pbuffs.buffer_addr_arr);
	ret = comm->c_coll->coll_allgather(
		&module->local_postbuf_attrs.address, 1, MPI_LONG_LONG,
		module->remote_pbuffs.buffer_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	module->remote_pbuffs.dbell_addr_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.dbell_addr_arr));
	_BKPAP_CHK_MALLOC(module->remote_pbuffs.dbell_addr_arr);
	ret = comm->c_coll->coll_allgather(
		&module->local_dbell_attrs.address, 1, MPI_LONG_LONG,
		module->remote_pbuffs.dbell_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_postbuf_h, &postbuf_rkey_buffer, &postbuf_rkey_buffer_size);
	_BKPAP_CHK_UCP(status);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_dbell_h, &dbell_rkey_buffer, &dbell_rkey_buffer_size);
	_BKPAP_CHK_UCP(status);

	postbuf_rkey_size_arr = calloc(mpi_size, sizeof(*postbuf_rkey_size_arr));
	_BKPAP_CHK_MALLOC(postbuf_rkey_size_arr);
	ret = comm->c_coll->coll_allgather(
		&postbuf_rkey_buffer_size, 1, MPI_LONG_LONG,
		postbuf_rkey_size_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	dbell_rkey_size_arr = calloc(mpi_size, sizeof(*dbell_rkey_size_arr));
	_BKPAP_CHK_MALLOC(dbell_rkey_size_arr);
	ret = comm->c_coll->coll_allgather(
		&dbell_rkey_buffer_size, 1, MPI_LONG_LONG,
		dbell_rkey_size_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	agv_displ_arr = calloc(mpi_size, sizeof(*agv_displ_arr));
	_BKPAP_CHK_MALLOC(agv_displ_arr);
	agv_count_arr = calloc(mpi_size, sizeof(*agv_count_arr));
	_BKPAP_CHK_MALLOC(agv_count_arr);

	agv_rkey_recv_buf_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = agv_rkey_recv_buf_size;
		agv_count_arr[i] = postbuf_rkey_size_arr[i];
		agv_rkey_recv_buf_size += postbuf_rkey_size_arr[i];
	}
	agv_rkey_recv_buf = malloc(agv_rkey_recv_buf_size);
	_BKPAP_CHK_MALLOC(agv_rkey_recv_buf);
	memset(agv_rkey_recv_buf, 0, agv_rkey_recv_buf_size);
	ret = comm->c_coll->coll_allgatherv(postbuf_rkey_buffer, postbuf_rkey_buffer_size, MPI_BYTE,
		agv_rkey_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module);
	_BKPAP_CHK_MPI(ret);
	module->remote_pbuffs.buffer_rkey_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.buffer_rkey_arr));
	_BKPAP_CHK_MALLOC(module->remote_pbuffs.buffer_rkey_arr);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->remote_pbuffs.buffer_rkey_arr[i]);
		_BKPAP_CHK_UCP(status);
	}

	agv_rkey_recv_buf_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = agv_rkey_recv_buf_size;
		agv_count_arr[i] = dbell_rkey_size_arr[i];
		agv_rkey_recv_buf_size += dbell_rkey_size_arr[i];
	}
	free(agv_rkey_recv_buf);
	agv_rkey_recv_buf = malloc(agv_rkey_recv_buf_size);
	_BKPAP_CHK_MALLOC(agv_rkey_recv_buf);
	memset(agv_rkey_recv_buf, 0, agv_rkey_recv_buf_size);
	ret = comm->c_coll->coll_allgatherv(dbell_rkey_buffer, dbell_rkey_buffer_size, MPI_BYTE,
		agv_rkey_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module);
	_BKPAP_CHK_MPI(ret);
	module->remote_pbuffs.dbell_rkey_arr = calloc(mpi_size, sizeof(*module->remote_pbuffs.dbell_rkey_arr));
	_BKPAP_CHK_MALLOC(module->remote_pbuffs.dbell_rkey_arr);
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_rkey_recv_buf + agv_displ_arr[i],
			&module->remote_pbuffs.dbell_rkey_arr[i]);
		_BKPAP_CHK_UCP(status);
	}

	ucp_rkey_buffer_release(postbuf_rkey_buffer);
	// BKPAP_OUTPUT("rank %d, rkeys [%lx %lx %lx %lx]", mpi_rank, 
	// 	module->remote_pbuffs.buffer_rkey_arr[0],
	// 	module->remote_pbuffs.buffer_rkey_arr[1],
	// 	module->remote_pbuffs.buffer_rkey_arr[2],
	// 	module->remote_pbuffs.buffer_rkey_arr[3]
	// );
	BKPAP_OUTPUT("ucp postbuf wireup SUCCESS");
bkpap_remotepostbuf_wireup_err:

	free(postbuf_rkey_size_arr);
	free(dbell_rkey_size_arr);
	free(agv_displ_arr);
	free(agv_count_arr);
	free(agv_rkey_recv_buf);
	return ret;
#undef _BKPAP_CHK_MALLOC
#undef _BKPAP_CHK_UCP
#undef _BKPAP_CHK_MPI
}

int mca_coll_bkpap_wireup_syncstructure(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_syncstructure_wireup_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in syncstructure wireup failed"); ret = OMPI_ERROR; goto bkpap_syncstructure_wireup_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in syncstructure wireup failed"); goto bkpap_syncstructure_wireup_err;}
	int ret = OMPI_SUCCESS, mpi_rank = ompi_comm_rank(comm), mpi_size = ompi_comm_size(comm);
	ucp_mem_map_params_t mem_map_params;
	ucs_status_t status = UCS_OK;
	void* counter_rkey_buffer = NULL, * arrival_arr_rkey_buffer = NULL;
	size_t counter_rkey_buffer_size, arrival_arr_rkey_buffer_size;
	int64_t* mapped_mem_tmp = NULL;


	if (mpi_rank == 0) {
		module->local_syncstructure = calloc(1, sizeof(*module->local_syncstructure));

		BKPAP_MSETZ(mem_map_params);
		mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
			UCP_MEM_MAP_PARAM_FIELD_LENGTH |
			UCP_MEM_MAP_PARAM_FIELD_FLAGS;

		mem_map_params.address = NULL;
		mem_map_params.length = sizeof(int64_t);
		mem_map_params.flags = UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_NONBLOCK;
		status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_syncstructure->counter_mem_h);
		_BKPAP_CHK_UCP(status);

		BKPAP_MSETZ(module->local_syncstructure->counter_attr);
		module->local_syncstructure->counter_attr.field_mask = UCP_MEM_ATTR_FIELD_LENGTH | UCP_MEM_ATTR_FIELD_ADDRESS;
		status = ucp_mem_query(module->local_syncstructure->counter_mem_h, &module->local_syncstructure->counter_attr);
		_BKPAP_CHK_UCP(status);
		mapped_mem_tmp = (int64_t*)module->local_syncstructure->counter_attr.address;
		*mapped_mem_tmp = -1;
		mapped_mem_tmp = NULL;

		status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_syncstructure->counter_mem_h,
			&counter_rkey_buffer, &counter_rkey_buffer_size);
		_BKPAP_CHK_UCP(status);
		ret = comm->c_coll->coll_bcast(&module->local_syncstructure->counter_attr.address, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(&counter_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(counter_rkey_buffer, counter_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);

		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], counter_rkey_buffer, &module->remote_syncstructure_counter_rkey);
		_BKPAP_CHK_UCP(status);
		module->remote_syncstructure_counter_addr = (uint64_t)module->local_syncstructure->counter_attr.address;


		mem_map_params.address = NULL;
		mem_map_params.length = sizeof(int64_t) * mpi_size;
		mem_map_params.flags = UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_NONBLOCK;
		status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_syncstructure->arrival_arr_mem_h);
		_BKPAP_CHK_UCP(status);

		BKPAP_MSETZ(module->local_syncstructure->arrival_arr_attr);
		module->local_syncstructure->arrival_arr_attr.field_mask = UCP_MEM_ATTR_FIELD_LENGTH | UCP_MEM_ATTR_FIELD_ADDRESS;
		status = ucp_mem_query(module->local_syncstructure->arrival_arr_mem_h, &module->local_syncstructure->arrival_arr_attr);
		_BKPAP_CHK_UCP(status);
		mapped_mem_tmp = (int64_t*)module->local_syncstructure->arrival_arr_attr.address;
		for (int i = 0; i < mpi_size; i++)
			mapped_mem_tmp[i] = -1;
		mapped_mem_tmp = NULL;

		status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_syncstructure->arrival_arr_mem_h,
			&arrival_arr_rkey_buffer, &arrival_arr_rkey_buffer_size);
		_BKPAP_CHK_UCP(status);
		ret = comm->c_coll->coll_bcast(&module->local_syncstructure->arrival_arr_attr.address, 1, MPI_LONG_LONG, 0,
			comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(&arrival_arr_rkey_buffer_size, 1, MPI_LONG_LONG, 0,
			comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(arrival_arr_rkey_buffer, arrival_arr_rkey_buffer_size, MPI_BYTE, 0,
			comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);

		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &module->remote_syncstructure_arrival_arr_rkey);
		_BKPAP_CHK_UCP(status);
		module->remote_syncstructure_arrival_arr_addr = (uint64_t)module->local_syncstructure->arrival_arr_attr.address;

		ucp_rkey_buffer_release(counter_rkey_buffer);
		counter_rkey_buffer = NULL;
		ucp_rkey_buffer_release(arrival_arr_rkey_buffer);
		arrival_arr_rkey_buffer = NULL;
	}
	else {
		ret = comm->c_coll->coll_bcast(&module->remote_syncstructure_counter_addr, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(&counter_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		counter_rkey_buffer = calloc(1, counter_rkey_buffer_size);
		_BKPAP_CHK_MALLOC(counter_rkey_buffer);
		ret = comm->c_coll->coll_bcast(counter_rkey_buffer, counter_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], counter_rkey_buffer, &module->remote_syncstructure_counter_rkey);
		_BKPAP_CHK_UCP(status);

		ret = comm->c_coll->coll_bcast(&module->remote_syncstructure_arrival_arr_addr, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(&arrival_arr_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		arrival_arr_rkey_buffer = calloc(1, arrival_arr_rkey_buffer_size);
		_BKPAP_CHK_MALLOC(arrival_arr_rkey_buffer);
		ret = comm->c_coll->coll_bcast(arrival_arr_rkey_buffer, arrival_arr_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &module->remote_syncstructure_arrival_arr_rkey);
		_BKPAP_CHK_UCP(status);
	}


	BKPAP_OUTPUT("ucp syncstructure wireup SUCCESS");
bkpap_syncstructure_wireup_err:
	free(counter_rkey_buffer);
	free(arrival_arr_rkey_buffer);
	return ret;
#undef _BKPAP_CHK_MALLOC
#undef _BKPAP_CHK_UCP
#undef _BKPAP_CHK_MPI
}


int mca_coll_bkpap_arrive_at_inter(mca_coll_bkpap_module_t* module, int64_t ss_rank, int64_t* ret_pos) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status = OMPI_SUCCESS;
	int64_t reply_buf = -1, put_buf = ss_rank;

	// TODO: Could try to be hardcore and move the arrival_arr put into the counter_fadd callback 

	status_ptr = ucp_atomic_fetch_nb(
		module->ucp_ep_arr[0], UCP_ATOMIC_FETCH_OP_FADD, 1, &reply_buf, sizeof(reply_buf),
		module->remote_syncstructure_counter_addr, module->remote_syncstructure_counter_rkey,
		_bk_send_cb_noparams);
	status = _bk_poll_completion(status_ptr);
	if (UCS_OK != status) {
		BKPAP_ERROR("counter increment failed");
		return OMPI_ERROR;
	}

	uint64_t put_addr = ((reply_buf + 1) * sizeof(int64_t)) + (module->remote_syncstructure_arrival_arr_addr);
	status_ptr = ucp_put_nb(
		module->ucp_ep_arr[0], &put_buf, sizeof(put_buf),
		put_addr, module->remote_syncstructure_arrival_arr_rkey,
		_bk_send_cb_noparams);

	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("inter UCS check failed: %d (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr))
		ucp_request_free(status_ptr);

	status = _bk_flush_worker();
	if (UCS_OK != status) {
		BKPAP_ERROR("worker flush failed");
		return OMPI_ERROR;
	}

	*ret_pos = reply_buf;
	return OMPI_SUCCESS;
}

int mca_coll_bkpap_leave_inter(mca_coll_bkpap_module_t* module, int arrival) {
	ucs_status_t status = UCS_OK;
	ucs_status_ptr_t status_ptr = NULL;
	int64_t put_buf = -1;

	uint64_t put_addr = (arrival * sizeof(int64_t)) + (module->remote_syncstructure_arrival_arr_addr);
	status_ptr = ucp_put_nb(
		module->ucp_ep_arr[0], &put_buf, sizeof(put_buf),
		put_addr, module->remote_syncstructure_arrival_arr_rkey,
		_bk_send_cb_noparams);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("put arrival failed");
		return OMPI_ERROR;
	}

	status = ucp_atomic_post(
		module->ucp_ep_arr[0], UCP_ATOMIC_POST_OP_ADD, -1, sizeof(int64_t),
		module->remote_syncstructure_counter_addr, module->remote_syncstructure_counter_rkey);
	if (status != UCS_OK) {
		BKPAP_ERROR("UCP post nb failed: %d (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}

	status = _bk_flush_worker();
	if (status != UCS_OK) {
		BKPAP_ERROR("bk flush worker: %d (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}
 
    // if (0 == ompi_comm_rank(ss_comm) && intra_rank == 0) {
    //     int64_t* tmp = (int64_t*)module->local_syncstructure->counter_attr.address;
    // //     while (-1 != *tmp) ucp_worker_progress(mca_coll_bkpap_component.ucp_worker);
    //     tmp = (int64_t*)module->local_syncstructure->arrival_arr_attr.address;
    //     for (int i = 0; i < ss_wsize; i++)
    //         tmp[i] = -1;
    // }

	return OMPI_SUCCESS;
}

int mca_coll_bkpap_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count,
	ompi_op_t* op, int num_buffers, mca_coll_bkpap_module_t* module) {
	int ret = OMPI_SUCCESS;
	volatile int64_t* dbells = module->local_dbell_attrs.address;
	uint8_t* pbuffs = module->local_postbuf_attrs.address;
	size_t dtype_size;
	ompi_datatype_type_size(dtype, &dtype_size);

	BKPAP_OUTPUT("rank %d reducing, dbells [ %ld %ld %ld ]", ompi_comm_rank(module->intra_comm),
		dbells[0],
		dbells[1],
		dbells[2]);

	for (int i = 0; i < num_buffers; i++) {
		while (BKPAP_DBELL_UNSET == dbells[i]);
		void* recived_buffer = pbuffs + (i * mca_coll_bkpap_component.postbuff_size);
		ompi_op_reduce(op, recived_buffer,
			local_buf, count, dtype);
		dbells[i] = BKPAP_DBELL_UNSET;
	}

	return ret;
}

int mca_coll_bkpap_get_rank_of_arrival(int arrival, mca_coll_bkpap_module_t* module, int* rank) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status = UCS_OK;
	int ret = OMPI_SUCCESS;
	int64_t get_buf;

	status_ptr = ucp_get_nb(
		module->ucp_ep_arr[0],
		&get_buf,
		sizeof(get_buf),
		module->remote_syncstructure_arrival_arr_addr + (arrival * sizeof(get_buf)),
		module->remote_syncstructure_arrival_arr_rkey,
		_bk_send_cb_noparams);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("Rank translation failed with error %d (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr))
		ucp_request_free(status_ptr);
	status = _bk_flush_worker();
	if (UCS_OK != status) {
		BKPAP_ERROR("bk flush failed");
		return OMPI_ERROR;
	}

	*rank = get_buf;
	return ret;
}

int mca_coll_bkpap_write_parent_postbuf(const void* buf,
	struct ompi_datatype_t* dtype, int count, int64_t arrival, int radix, int send_rank,
	struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucs_status_t status;
	ucs_status_ptr_t status_ptr;
	int ret = OMPI_SUCCESS;
	int64_t dbell_put_buf = BKPAP_DBELL_SET;
	uint64_t postbuf_addr;
	size_t dtype_size, buf_size;
	int k = mca_coll_bkpap_component.allreduce_k_value;

	int slot = ((arrival / (radix / k)) % k) - 1;

	postbuf_addr = (module->remote_pbuffs.buffer_addr_arr[send_rank]) + (slot * mca_coll_bkpap_component.postbuff_size);
	ompi_datatype_type_size(dtype, &dtype_size);
	buf_size = dtype_size * (ptrdiff_t)count;
	BKPAP_OUTPUT("arrival: %ld (rank %d), dest rank: %d, slot %d, addr: %lx, rkey:%lx", arrival, ompi_comm_rank(comm), send_rank, slot, postbuf_addr, (intptr_t)module->remote_pbuffs.buffer_rkey_arr[send_rank]);
	status_ptr = ucp_put_nb(
		module->ucp_ep_arr[send_rank], buf, buf_size,
		postbuf_addr,
		module->remote_pbuffs.buffer_rkey_arr[send_rank],
		_bk_send_cb_noparams);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("rank %d (arrive %ld) Write patent postbuf returned error %d (%s)", ompi_comm_rank(comm), arrival, UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr))
		ucp_request_free(status_ptr);

	status = ucp_worker_fence(mca_coll_bkpap_component.ucp_worker);
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker fence failed");
		return OMPI_ERROR;
	}

	uint64_t dbell_addr = (module->remote_pbuffs.dbell_addr_arr[send_rank]) + (slot * sizeof(uint64_t));
	status_ptr = ucp_put_nb(
		module->ucp_ep_arr[send_rank], &dbell_put_buf, sizeof(dbell_put_buf),
		dbell_addr,
		module->remote_pbuffs.dbell_rkey_arr[send_rank],
		_bk_send_cb_noparams);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("rank %d (arrive %ld) Write patent debll returned error %d (%s)", ompi_comm_rank(comm), arrival, UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr))
		ucp_request_free(status_ptr);

	status = _bk_flush_worker();
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker flush failed");
		return OMPI_ERROR;
	}

	return ret;
}

int mca_coll_bkpap_reduce_postbufs_p2p(void* local_buf, struct ompi_datatype_t* dtype, int count,
	ompi_op_t* op, int num_buffers, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	int ret = OMPI_SUCCESS;
	volatile int64_t* dbells = module->local_dbell_attrs.address;
	size_t dtype_size;
	ompi_datatype_type_size(dtype, &dtype_size);

	void* tmp_recived_buffer = calloc(count, dtype_size);

	BKPAP_OUTPUT("rank %d p2p reduce, dbells [ %ld %ld %ld ]", ompi_comm_rank(comm),
		dbells[0], dbells[1], dbells[2]);

	for (int i = 0; i < num_buffers; i++) {
		while (BKPAP_DBELL_UNSET == dbells[i]);

		ret = MCA_PML_CALL(recv(tmp_recived_buffer, count, dtype, dbells[i], MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE));
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("reduce p2p recieve failed");
			return ret;
		}
		ompi_op_reduce(op, tmp_recived_buffer, local_buf, count, dtype);

		dbells[i] = BKPAP_DBELL_UNSET;
	}

	BKPAP_OUTPUT("WE REDUCED WITH WITH P2P");

	return ret;
}

int mca_coll_bkpap_write_parent_postbuf_p2p(const void* buf,
	struct ompi_datatype_t* dtype, int count, int64_t arrival, int radix, int send_rank,
	struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module) {
	ucs_status_t status;
	ucs_status_ptr_t status_ptr;
	int ret = OMPI_SUCCESS;
	int64_t dbell_put_buf = ompi_comm_rank(comm);
	uint64_t postbuf_addr;
	size_t dtype_size;
	int k = mca_coll_bkpap_component.allreduce_k_value;

	int slot = ((arrival / (radix / k)) % k) - 1;

	postbuf_addr = (module->remote_pbuffs.buffer_addr_arr[send_rank]) + (slot * mca_coll_bkpap_component.postbuff_size);
	ompi_datatype_type_size(dtype, &dtype_size);
	BKPAP_OUTPUT("arrival: %ld (rank %d), dest rank: %d, slot %d, addr: %lx, rkey:%lx", arrival, ompi_comm_rank(comm), send_rank, slot, postbuf_addr, (intptr_t)module->remote_pbuffs.buffer_rkey_arr[send_rank]);

	uint64_t dbell_addr = (module->remote_pbuffs.dbell_addr_arr[send_rank]) + (slot * sizeof(uint64_t));
	status_ptr = ucp_put_nb(
		module->ucp_ep_arr[send_rank], &dbell_put_buf, sizeof(dbell_put_buf),
		dbell_addr,
		module->remote_pbuffs.dbell_rkey_arr[send_rank],
		_bk_send_cb_noparams);
	if (UCS_PTR_IS_ERR(status_ptr)) {
		BKPAP_ERROR("rank %d (arrive %ld) Write patent debll returned error %d (%s)", ompi_comm_rank(comm), arrival, UCS_PTR_STATUS(status_ptr), ucs_status_string(UCS_PTR_STATUS(status_ptr)));
		return OMPI_ERROR;
	}
	if (UCS_PTR_IS_PTR(status_ptr))
		ucp_request_free(status_ptr);

	status = _bk_flush_worker();
	if (UCS_OK != status) {
		BKPAP_ERROR("Worker flush failed");
		return OMPI_ERROR;
	}
	
	ret = MCA_PML_CALL(send(buf, count, dtype, send_rank, MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
	if(OMPI_SUCCESS != ret){
		BKPAP_ERROR("write_parent_p2p failed");
		return ret;
	}
	BKPAP_OUTPUT("WE POSTED WITH WITH P2P");

	return ret;
}
