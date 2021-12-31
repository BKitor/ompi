#include "coll_bkpap.h"

void mca_coll_bkpap_amoreq_init(void* request) {
	mca_coll_bkpap_amoreq_t* r = request;
	r->ucs_status = UCS_OK;
	r->complete = 0;
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
	ucp_params.features = UCP_FEATURE_AMO64;
	ucp_params.request_size = sizeof(mca_coll_bkpap_amoreq_t);
	ucp_params.request_init = mca_coll_bkpap_amoreq_init;
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
	void* agv_remote_addr_recv_buf = NULL;

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
	memset(agv_remote_addr_recv_buf , 0, total_addr_buff_size);

	// allgatherv the ucp_addr_t
	ret = comm->c_coll->coll_allgatherv(
		mca_coll_bkpap_component.ucp_worker_addr, mca_coll_bkpap_component.ucp_worker_addr_len, MPI_BYTE,
		agv_remote_addr_recv_buf, agv_count_arr , agv_displ_arr, MPI_BYTE,
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
		ep_params.address = agv_remote_addr_recv_buf + agv_displ_arr[i];
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

int mca_coll_bkpap_wireup_remote_postbuffs(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_remotepostbuf_wireup_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in postbuf wireup failed"); ret = OMPI_ERROR; goto bkpap_remotepostbuf_wireup_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in postbuf wireup failed"); goto bkpap_remotepostbuf_wireup_err;}
	int ret = OMPI_SUCCESS, mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	ucs_status_t status = UCS_OK;
	ucp_mem_map_params_t mem_map_params;
	void* rkey_buffer = NULL;
	size_t rkey_buffer_size, * remote_rkey_size_arr = NULL;
	int* agv_displ_arr = NULL, * agv_count_arr = NULL;
	void* agv_remote_rkey_recv_buf = NULL;
	
	BKPAP_MSETZ(mem_map_params);
	BKPAP_MSETZ(module->local_postbuf_attrs);

	mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
		UCP_MEM_MAP_PARAM_FIELD_LENGTH |
		UCP_MEM_MAP_PARAM_FIELD_FLAGS;
	mem_map_params.address = NULL;
	mem_map_params.length = mca_coll_bkpap_component.postbuff_size;
	mem_map_params.flags = UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_NONBLOCK;

	status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_postbuf_h);
	_BKPAP_CHK_UCP(status);

	module->local_postbuf_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
	status = ucp_mem_query(module->local_postbuf_h, &module->local_postbuf_attrs);
	_BKPAP_CHK_UCP(status);

	module->remote_postbuff_addr_arr = calloc(mpi_size, sizeof(*module->remote_postbuff_addr_arr));
	_BKPAP_CHK_MALLOC(module->remote_postbuff_addr_arr);
	ret = comm->c_coll->coll_allgather(
		&module->local_postbuf_attrs.address, 1, MPI_LONG_LONG,
		module->remote_postbuff_addr_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, module->local_postbuf_h, &rkey_buffer, &rkey_buffer_size);
	_BKPAP_CHK_UCP(status);

	remote_rkey_size_arr = calloc(mpi_size, sizeof(*remote_rkey_size_arr));
	_BKPAP_CHK_MALLOC(remote_rkey_size_arr);
	ret = comm->c_coll->coll_allgather(
		&rkey_buffer_size, 1, MPI_LONG_LONG,
		remote_rkey_size_arr, 1, MPI_LONG_LONG,
		comm, comm->c_coll->coll_allgather_module);
	_BKPAP_CHK_MPI(ret);

	agv_displ_arr = calloc(mpi_size, sizeof(*agv_displ_arr));
	_BKPAP_CHK_MALLOC(agv_displ_arr);
	agv_count_arr = calloc(mpi_size, sizeof(*agv_count_arr));
	_BKPAP_CHK_MALLOC(agv_count_arr);
	size_t agv_remote_rkey_recv_buf_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = agv_remote_rkey_recv_buf_size;
		agv_count_arr[i] = remote_rkey_size_arr[i];
		agv_remote_rkey_recv_buf_size += remote_rkey_size_arr[i];
	}
	agv_remote_rkey_recv_buf = malloc(agv_remote_rkey_recv_buf_size);
	_BKPAP_CHK_MALLOC(agv_remote_rkey_recv_buf);
	memset(agv_remote_rkey_recv_buf, 0, agv_remote_rkey_recv_buf_size);

	ret = comm->c_coll->coll_allgatherv(rkey_buffer, rkey_buffer_size, MPI_BYTE,
			agv_remote_rkey_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE, 
			comm, comm->c_coll->coll_allgatherv_module);
	_BKPAP_CHK_MPI(ret);
	
	module->remote_postbuff_rkey_arr = calloc(mpi_size, sizeof(*module->remote_postbuff_rkey_arr));
	_BKPAP_CHK_MALLOC(module->remote_postbuff_rkey_arr);
	for(int i = 0; i<mpi_size; i++){
		if(i == mpi_rank)continue;
		status = ucp_ep_rkey_unpack(
			module->ucp_ep_arr[i],
			agv_remote_rkey_recv_buf + agv_displ_arr[i],
			&module->remote_postbuff_rkey_arr[i]);
		_BKPAP_CHK_UCP(status);
	}

	ucp_rkey_buffer_release(rkey_buffer);
	BKPAP_OUTPUT("ucp postbuf wireup SUCCESS");
bkpap_remotepostbuf_wireup_err:

	free(remote_rkey_size_arr);
	free(agv_displ_arr);
	free(agv_count_arr);
	free(agv_remote_rkey_recv_buf);
	return ret;
#undef _BKPAP_CHK_MALLOC
#undef _BKPAP_CHK_UCP
#undef _BKPAP_CHK_MPI
}

int mca_coll_bkpap_wireup_syncstructure(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm){
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_syncstructure_wireup_err;}
#define _BKPAP_CHK_UCP(_status) if(UCS_OK != _status){BKPAP_ERROR("UCP op in syncstructure wireup failed"); ret = OMPI_ERROR; goto bkpap_syncstructure_wireup_err;}
#define _BKPAP_CHK_MPI(_ret) if(OMPI_SUCCESS != _ret){BKPAP_ERROR("MPI op in syncstructure wireup failed"); goto bkpap_syncstructure_wireup_err;}
	int ret = OMPI_SUCCESS, mpi_rank = ompi_comm_rank(comm), mpi_size = ompi_comm_size(comm);
	ucp_mem_map_params_t mem_map_params;
	ucs_status_t status = UCS_OK;
	void *counter_rkey_buffer = NULL, *arrival_arr_rkey_buffer = NULL;
	size_t counter_rkey_buffer_size, arrival_arr_rkey_buffer_size;

	
	if (mpi_rank == 0){
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
		module->remote_syncstructure_counter_addr = module->local_syncstructure->counter_attr.address;


		mem_map_params.address = NULL;
		mem_map_params.length = sizeof(int64_t) * mpi_size;
		mem_map_params.flags = UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_NONBLOCK;
		status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &module->local_syncstructure->arrival_arr_mem_h);
		_BKPAP_CHK_UCP(status);
		
		BKPAP_MSETZ(module->local_syncstructure->arrival_arr_attr);
		module->local_syncstructure->arrival_arr_attr.field_mask = UCP_MEM_ATTR_FIELD_LENGTH | UCP_MEM_ATTR_FIELD_ADDRESS; 
		status = ucp_mem_query(module->local_syncstructure->arrival_arr_mem_h, &module->local_syncstructure->arrival_arr_attr);
		_BKPAP_CHK_UCP(status);
		
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
		
		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &module->remote_syncstructure_arrival_rkey);
		_BKPAP_CHK_UCP(status);
		module->remote_syncstructure_arrival_addr = module->local_syncstructure->arrival_arr_attr.address;

		ucp_rkey_buffer_release(counter_rkey_buffer);
		counter_rkey_buffer = NULL;
		ucp_rkey_buffer_release(arrival_arr_rkey_buffer);
		arrival_arr_rkey_buffer = NULL;
	}else{
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
		
		ret = comm->c_coll->coll_bcast(&module->remote_syncstructure_arrival_addr, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		ret = comm->c_coll->coll_bcast(&arrival_arr_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		arrival_arr_rkey_buffer = calloc(1, arrival_arr_rkey_buffer_size);
		_BKPAP_CHK_MALLOC(arrival_arr_rkey_buffer);
		ret = comm->c_coll->coll_bcast(arrival_arr_rkey_buffer, arrival_arr_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
		_BKPAP_CHK_MPI(ret);
		status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &module->remote_syncstructure_arrival_rkey);
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