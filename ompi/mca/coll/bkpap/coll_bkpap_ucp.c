#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "bkpap_kernel.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/pml/pml.h"

void mca_coll_bkpap_req_init(void* request) {
	mca_coll_bkpap_req_t* r = request;
	r->ucs_status = UCS_INPROGRESS;
	r->complete = 0;
}

int mca_coll_bkpap_init_ucx(int enable_mpi_threads) {
	int ret = OMPI_SUCCESS;
	ucp_params_t ucp_params;
	ucp_worker_params_t worker_params;
	ucp_config_t* config;
	ucs_status_t status;

	BKPAP_MSETZ(ucp_params);
	BKPAP_MSETZ(worker_params);

	// status = ucp_config_read("MPI", NULL, &config);
	status = ucp_config_read(NULL, NULL, &config);
	if (UCS_OK != status) {
		return OMPI_ERROR;
	}

	ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
		UCP_PARAM_FIELD_REQUEST_SIZE |
		UCP_PARAM_FIELD_REQUEST_INIT |
		UCP_PARAM_FIELD_MT_WORKERS_SHARED |
		UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
	ucp_params.features = UCP_FEATURE_AMO64 | UCP_FEATURE_RMA;
	ucp_params.features |= (mca_coll_bkpap_component.dataplane_type == BKPAP_DATAPLANE_TAG) ? UCP_FEATURE_TAG : 0;
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
	BKPAP_CHK_UCP(status, bkpap_init_ucp_err);

	worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = (enable_mpi_threads == MPI_THREAD_SINGLE) ? UCS_THREAD_MODE_SINGLE : UCS_THREAD_MODE_MULTI;
	status = ucp_worker_create(mca_coll_bkpap_component.ucp_context, &worker_params, &mca_coll_bkpap_component.ucp_worker);
	BKPAP_CHK_UCP(status, bkpap_init_ucp_err);

	status = ucp_worker_get_address(
		mca_coll_bkpap_component.ucp_worker,
		&mca_coll_bkpap_component.ucp_worker_addr,
		&mca_coll_bkpap_component.ucp_worker_addr_len
	);
	BKPAP_CHK_UCP(status, bkpap_init_ucp_err);

bkpap_init_ucp_err:
	return ret;
}

// might want to make static inline, and move to header
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	ucs_status_t status = UCS_OK;
	ucp_ep_params_t ep_params;
	int ret = OMPI_SUCCESS;
	int mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	int* agv_count_arr = NULL, * agv_displ_arr = NULL;
	size_t* remote_addr_len_buf = NULL;
	uint8_t* agv_remote_addr_recv_buf = NULL;

	BKPAP_MSETZ(ep_params);

	agv_count_arr = calloc(mpi_size, sizeof(*agv_count_arr));
	BKPAP_CHK_MALLOC(agv_count_arr, bkpap_ep_wireup_err);
	agv_displ_arr = calloc(mpi_size, sizeof(*agv_displ_arr));
	BKPAP_CHK_MALLOC(agv_displ_arr, bkpap_ep_wireup_err);
	remote_addr_len_buf = calloc(mpi_size, sizeof(*remote_addr_len_buf));
	BKPAP_CHK_MALLOC(remote_addr_len_buf, bkpap_ep_wireup_err);

	// gather address lengths
	ret = comm->c_coll->coll_allgather(
		&mca_coll_bkpap_component.ucp_worker_addr_len, 1, MPI_LONG_LONG,
		remote_addr_len_buf, 1, MPI_LONG_LONG, comm,
		comm->c_coll->coll_allgather_module
	);
	BKPAP_CHK_MPI(ret, bkpap_ep_wireup_err);

	// setup allgatherv count/displs
	size_t total_addr_buff_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = total_addr_buff_size;
		agv_count_arr[i] = remote_addr_len_buf[i];
		total_addr_buff_size += remote_addr_len_buf[i];
	}
	agv_remote_addr_recv_buf = malloc(total_addr_buff_size);
	BKPAP_CHK_MALLOC(agv_remote_addr_recv_buf, bkpap_ep_wireup_err);
	memset(agv_remote_addr_recv_buf, 0, total_addr_buff_size);

	// allgatherv the ucp_addr_t
	ret = comm->c_coll->coll_allgatherv(
		mca_coll_bkpap_component.ucp_worker_addr, mca_coll_bkpap_component.ucp_worker_addr_len, MPI_BYTE,
		agv_remote_addr_recv_buf, agv_count_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module
	);
	BKPAP_CHK_MPI(ret, bkpap_ep_wireup_err);

	// for loop to populate ep_arr
	// module->ucp_ep_arr = calloc(mpi_size, sizeof(*module->ucp_ep_arr));
	module->ucp_ep_arr = calloc(mpi_size, sizeof(ucp_ep_h));
	BKPAP_CHK_MALLOC(module->ucp_ep_arr, bkpap_ep_wireup_err);
	ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
	module->wsize = mpi_size;
	module->rank = mpi_rank;
	for (int i = 0; i < mpi_size; i++) {
		// if (i == mpi_rank)continue;
		ep_params.address = (void*)(agv_remote_addr_recv_buf + agv_displ_arr[i]);
		status = ucp_ep_create(mca_coll_bkpap_component.ucp_worker, &ep_params, &module->ucp_ep_arr[i]);
		BKPAP_CHK_UCP(status, bkpap_ep_wireup_err);
	}


	BKPAP_OUTPUT("ucp endpoints wireup SUCCESS");
bkpap_ep_wireup_err:
	free(agv_remote_addr_recv_buf);
	free(agv_count_arr);
	free(agv_displ_arr);
	free(remote_addr_len_buf);

	return ret;
}


int mca_coll_bkpap_wireup_syncstructure(int num_counters, int num_arrival_slots, int num_structures, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	int ret = OMPI_SUCCESS, mpi_rank = ompi_comm_rank(comm);
	ucp_mem_map_params_t mem_map_params;
	ucs_status_t status = UCS_OK;
	void* counter_rkey_buffer = NULL, * arrival_arr_rkey_buffer = NULL;
	size_t counter_rkey_buffer_size, arrival_arr_rkey_buffer_size;
	int64_t* mapped_mem_tmp = NULL;

	module->remote_syncstructure = calloc(num_structures, sizeof(*module->remote_syncstructure));
	BKPAP_CHK_MALLOC(module->remote_syncstructure, bkpap_syncstructure_wireup_err);

	if (mpi_rank == 0) {
		module->local_syncstructure = calloc(num_structures, sizeof(*module->local_syncstructure));
		BKPAP_CHK_MALLOC(module->local_syncstructure, bkpap_syncstructure_wireup_err);
	}

	module->num_syncstructures = num_structures;

	for (int ss_alloc_iter = 0; ss_alloc_iter < num_structures; ss_alloc_iter++) {
		mca_coll_bkpap_remote_syncstruct_t* remote_ss_tmp = &(module->remote_syncstructure[ss_alloc_iter]);

		if (mpi_rank == 0) {
			mca_coll_bkpap_local_syncstruct_t* local_ss_tmp = &(module->local_syncstructure[ss_alloc_iter]);

			BKPAP_MSETZ(mem_map_params);
			mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
				UCP_MEM_MAP_PARAM_FIELD_LENGTH |
				UCP_MEM_MAP_PARAM_FIELD_FLAGS;

			mem_map_params.address = NULL;
			mem_map_params.length = num_counters * sizeof(int64_t);
			mem_map_params.flags = UCP_MEM_MAP_ALLOCATE;
			status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &(local_ss_tmp->counter_mem_h));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);

			BKPAP_MSETZ(local_ss_tmp->counter_attr);
			local_ss_tmp->counter_attr.field_mask = UCP_MEM_ATTR_FIELD_LENGTH | UCP_MEM_ATTR_FIELD_ADDRESS;
			status = ucp_mem_query(local_ss_tmp->counter_mem_h, &(local_ss_tmp->counter_attr));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			mapped_mem_tmp = (int64_t*)local_ss_tmp->counter_attr.address;
			for (int j = 0; j < num_counters; j++)
				mapped_mem_tmp[j] = -1;
			mapped_mem_tmp = NULL;

			status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, local_ss_tmp->counter_mem_h,
				&counter_rkey_buffer, &counter_rkey_buffer_size);
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&(local_ss_tmp->counter_attr.address), 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&counter_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(counter_rkey_buffer, counter_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);

			status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], counter_rkey_buffer, &(remote_ss_tmp->counter_rkey));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			remote_ss_tmp->counter_addr = (uint64_t)local_ss_tmp->counter_attr.address;

			mem_map_params.address = NULL;
			mem_map_params.length = num_arrival_slots * sizeof(int64_t);
			mem_map_params.flags = UCP_MEM_MAP_ALLOCATE;
			status = ucp_mem_map(mca_coll_bkpap_component.ucp_context, &mem_map_params, &(local_ss_tmp->arrival_arr_mem_h));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);

			BKPAP_MSETZ(local_ss_tmp->arrival_arr_attr);
			local_ss_tmp->arrival_arr_attr.field_mask = UCP_MEM_ATTR_FIELD_LENGTH | UCP_MEM_ATTR_FIELD_ADDRESS;
			status = ucp_mem_query(local_ss_tmp->arrival_arr_mem_h, &(local_ss_tmp->arrival_arr_attr));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			mapped_mem_tmp = (int64_t*)local_ss_tmp->arrival_arr_attr.address;
			for (int j = 0; j < num_arrival_slots; j++)
				mapped_mem_tmp[j] = -1;
			mapped_mem_tmp = NULL;

			status = ucp_rkey_pack(mca_coll_bkpap_component.ucp_context, local_ss_tmp->arrival_arr_mem_h,
				&arrival_arr_rkey_buffer, &arrival_arr_rkey_buffer_size);
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&(local_ss_tmp->arrival_arr_attr.address), 1, MPI_LONG_LONG, 0,
				comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&arrival_arr_rkey_buffer_size, 1, MPI_LONG_LONG, 0,
				comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(arrival_arr_rkey_buffer, arrival_arr_rkey_buffer_size, MPI_BYTE, 0,
				comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);

			status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &remote_ss_tmp->arrival_arr_rkey);
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
			remote_ss_tmp->arrival_arr_addr = (uint64_t)local_ss_tmp->arrival_arr_attr.address;

			ucp_rkey_buffer_release(counter_rkey_buffer);
			counter_rkey_buffer = NULL;
			ucp_rkey_buffer_release(arrival_arr_rkey_buffer);
			arrival_arr_rkey_buffer = NULL;
		}
		else {
			ret = comm->c_coll->coll_bcast(&(remote_ss_tmp->counter_addr), 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&counter_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			counter_rkey_buffer = calloc(1, counter_rkey_buffer_size);
			BKPAP_CHK_MALLOC(counter_rkey_buffer, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(counter_rkey_buffer, counter_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], counter_rkey_buffer, &(remote_ss_tmp->counter_rkey));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);

			ret = comm->c_coll->coll_bcast(&(remote_ss_tmp->arrival_arr_addr), 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(&arrival_arr_rkey_buffer_size, 1, MPI_LONG_LONG, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			arrival_arr_rkey_buffer = calloc(1, arrival_arr_rkey_buffer_size);
			BKPAP_CHK_MALLOC(arrival_arr_rkey_buffer, bkpap_syncstructure_wireup_err);
			ret = comm->c_coll->coll_bcast(arrival_arr_rkey_buffer, arrival_arr_rkey_buffer_size, MPI_BYTE, 0, comm, comm->c_coll->coll_bcast_module);
			BKPAP_CHK_MPI(ret, bkpap_syncstructure_wireup_err);
			status = ucp_ep_rkey_unpack(module->ucp_ep_arr[0], arrival_arr_rkey_buffer, &(remote_ss_tmp->arrival_arr_rkey));
			BKPAP_CHK_UCP(status, bkpap_syncstructure_wireup_err);
		}
	}

	BKPAP_OUTPUT("ucp syncstructure wireup SUCCESS");
bkpap_syncstructure_wireup_err:
	free(counter_rkey_buffer);
	free(arrival_arr_rkey_buffer);
	return ret;
}

// TODO: Add a check that arrival_pos < comm_size, and stall if it isn't  
int mca_coll_bkpap_arrive_ss(int64_t ss_rank, uint64_t counter_offset, uint64_t arrival_arr_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss,
	mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm, int64_t* ret_pos) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status = UCS_OK;
	int64_t reply_buf = -1, op_buf = 1;
	uint64_t put_buf = ss_rank;

	uint64_t counter_addr = (remote_ss->counter_addr) + (counter_offset * sizeof(int64_t));
	uint64_t arrival_arr_addr = (remote_ss->arrival_arr_addr) + (arrival_arr_offset * sizeof(int64_t));

	ucp_request_param_t req_params = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_REPLY_BUFFER |
			UCP_OP_ATTR_FIELD_CALLBACK |
			UCP_OP_ATTR_FIELD_DATATYPE,
		.cb.send = _bk_send_cb,
		.user_data = NULL,
		.datatype = ucp_dt_make_contig(sizeof(reply_buf)),
		.reply_buffer = &reply_buf,
	};

	// TODO: Could try to be hardcore and move the arrival_arr put into the counter_fadd callback 
	status_ptr = ucp_atomic_op_nbx(
		module->ucp_ep_arr[0], UCP_ATOMIC_OP_ADD, &op_buf, 1,
		counter_addr, remote_ss->counter_rkey, &req_params);
	if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(status_ptr))) {
		status = UCS_PTR_STATUS(status_ptr);
		BKPAP_ERROR("atomic_op_nbx failed code %d (%s)", status, ucs_status_string(status));
		ucp_request_free(status_ptr);
		return OMPI_ERROR;
	}
	status = _bk_poll_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("_bk_poll_completion failed code: %d, (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}

	uint64_t put_addr = arrival_arr_addr + ((reply_buf + 1) * sizeof(int64_t));
	req_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
	status_ptr = ucp_put_nbx(
		module->ucp_ep_arr[0], &put_buf, sizeof(put_buf),
		put_addr, remote_ss->arrival_arr_rkey, &req_params);

	status = _bk_poll_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("_bk_poll_completion failed code: %d, (%s)", status, ucs_status_string(status));
		return OMPI_ERROR;
	}

	*ret_pos = reply_buf;
	return OMPI_SUCCESS;
}

int mca_coll_bkpap_leave_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status = UCS_OK;
	ucp_request_param_t req_params = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
			UCP_OP_ATTR_FIELD_USER_DATA |
			UCP_OP_ATTR_FIELD_DATATYPE,
		.cb.send = _bk_send_cb,
		.user_data = NULL,
		.datatype = ucp_dt_make_contig(sizeof(int64_t)),
	};

	int64_t op_buffer = -1;
	status_ptr = ucp_atomic_op_nbx(module->ucp_ep_arr[0], UCP_ATOMIC_OP_ADD, &op_buffer, 1, remote_ss->counter_addr, remote_ss->counter_rkey, &req_params);
	status = _bk_poll_completion(status_ptr);
	if (OPAL_UNLIKELY(UCS_OK != status)) {
		BKPAP_ERROR("_poll_completoin of leave atomic failed");
		return OMPI_ERROR;
	}

	return OMPI_SUCCESS;
}


int mca_coll_bkpap_get_rank_of_arrival(int arrival, uint64_t arrival_round_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, int* rank) {
	ucs_status_ptr_t status_ptr = NULL;
	ucs_status_t status = UCS_OK;
	int ret = OMPI_SUCCESS;
	int64_t get_buf;

	ucp_request_param_t req_params = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
		.cb.send = _bk_send_cb,
		.user_data = NULL
	};

	uint64_t arrival_arr_addr = remote_ss->arrival_arr_addr + (arrival_round_offset * sizeof(int64_t));
	uint64_t arrival_offset = (arrival * sizeof(int64_t));

	status_ptr = ucp_get_nbx(
		module->ucp_ep_arr[0],
		&get_buf,
		sizeof(get_buf),
		(arrival_arr_addr + arrival_offset),
		remote_ss->arrival_arr_rkey,
		&req_params);

	status = _bk_poll_completion(status_ptr);
	if (UCS_OK != status) {
		BKPAP_ERROR("poll completion on get_rank_of_arrival failed");
		return OMPI_ERROR;
	}

	*rank = get_buf;
	return ret;
}


int mca_coll_bkpap_reset_remote_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, struct ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* module) {
	ucs_status_t status = UCS_OK;
	ucs_status_ptr_t status_ptr[2] = { UCS_OK, UCS_OK };
	int ret = OMPI_SUCCESS;

	ucp_request_param_t req_params = {
		.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
		.cb.send = _bk_send_cb,
		.user_data = NULL,
	};

	size_t put_buf_size = remote_ss->ss_arrival_arr_len * sizeof(int64_t);
	int64_t* put_buffer = malloc(put_buf_size);
	for (int i = 0; i < remote_ss->ss_arrival_arr_len; i++) {
		put_buffer[i] = -1;
	}

	status_ptr[0] = ucp_put_nbx(module->ucp_ep_arr[0], put_buffer, put_buf_size,
		remote_ss->arrival_arr_addr, remote_ss->arrival_arr_rkey,
		&req_params);

	status_ptr[1] = ucp_put_nbx(module->ucp_ep_arr[0], put_buffer,
		(remote_ss->ss_counter_len * sizeof(int64_t)), remote_ss->counter_addr, remote_ss->counter_rkey,
		&req_params);

	status = _bk_poll_all_completion(status_ptr, 2);
	if (UCS_OK != status) {
		BKPAP_ERROR("Flush failed");
		return OMPI_ERROR;
	}

	free(put_buffer);
	return ret;
}
