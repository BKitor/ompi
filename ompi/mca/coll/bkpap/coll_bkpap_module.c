#include "coll_bkpap.h"
#include <string.h>
#include "opal/util/show_help.h"

static void mca_coll_bkpap_module_construct(mca_coll_bkpap_module_t* module) {
	module->fallback_allreduce = NULL;
	module->fallback_allreduce_module = NULL;
	module->ucp_ep_arr = NULL;
	module->wsize = -1;
	module->rank = -1;
}

static void mca_coll_bkpap_module_destruct(mca_coll_bkpap_module_t* module) {
	for (uint32_t i = 0; i < module->wsize; i++) {
		if(NULL == module->ucp_ep_arr) break;
		if (NULL == module->ucp_ep_arr[i]) continue;
		ucp_ep_destroy(module->ucp_ep_arr[i]);
	}
	free(module->ucp_ep_arr);

	OBJ_RELEASE(module->fallback_allreduce_module);
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

	return OMPI_SUCCESS;
}


// might want to make static inline, and move to header
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm) {
#define _BKPAP_CHK_MALLOC(_buf) if(NULL == _buf){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto bkpap_ep_wireup_err;}
	ucs_status_t status;
	ucp_ep_params_t ep_params;
	int ret = OMPI_SUCCESS;
	int mpi_size = ompi_comm_size(comm), mpi_rank = ompi_comm_rank(comm);
	int* agv_recv_arr = NULL, * agv_displ_arr = NULL;
	size_t* remote_addr_len_buf = NULL;
	void* remote_addr_buf = NULL;

	BKPAP_MSETZ(ep_params);

	agv_recv_arr = calloc(mpi_size, sizeof(*agv_recv_arr));
	_BKPAP_CHK_MALLOC(agv_recv_arr);
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
	if (OMPI_SUCCESS != ret) {
		BKPAP_ERROR("Remote addr len allgather failed");
		return ret;
	}

	// setup allgatherv count/displs
	size_t total_addr_buff_size = 0;
	for (int i = 0; i < mpi_size; i++) {
		agv_displ_arr[i] = total_addr_buff_size;
		agv_recv_arr[i] = remote_addr_len_buf[i];
		total_addr_buff_size += remote_addr_len_buf[i];
	}
	remote_addr_buf = malloc(total_addr_buff_size);
	memset(remote_addr_buf, 0, total_addr_buff_size);

	// allgatherv the ucp_addr_t
	comm->c_coll->coll_allgatherv(
		mca_coll_bkpap_component.ucp_worker_addr, mca_coll_bkpap_component.ucp_worker_addr_len, MPI_BYTE,
		remote_addr_buf, agv_recv_arr, agv_displ_arr, MPI_BYTE,
		comm, comm->c_coll->coll_allgatherv_module
	);
	if (OMPI_SUCCESS != ret) {
		BKPAP_ERROR("Remote addr len allgather failed");
		return ret;
	}

	// for loop to populate ep_arr
	// module->ucp_ep_arr = calloc(mpi_size, sizeof(*module->ucp_ep_arr));
	module->ucp_ep_arr = calloc(mpi_size, sizeof(ucp_ep_h));
	_BKPAP_CHK_MALLOC(module->ucp_ep_arr);
	ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
	module->wsize = mpi_size;
	module->rank = mpi_rank;
	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_rank)continue;
		ep_params.address = remote_addr_buf + agv_displ_arr[i];
		status = ucp_ep_create(mca_coll_bkpap_component.ucp_worker, &ep_params, &module->ucp_ep_arr[i]);
		if (UCS_OK != status) {
			BKPAP_ERROR("Establishing endpoint %d failed", i);
			goto bkpap_ep_wireup_err;
		}
	}

	free(agv_recv_arr);
	free(agv_displ_arr);
	free(remote_addr_len_buf);

	BKPAP_OUTPUT("ucp endpoints wiredup");
	return OMPI_SUCCESS;

bkpap_ep_wireup_err:
	free(agv_recv_arr);
	free(agv_displ_arr);
	free(remote_addr_len_buf);

	return OMPI_ERROR;
}
int mca_coll_bkpap_wireup_recievebuffers(void) { return OPAL_ERR_NOT_IMPLEMENTED; }
int mca_coll_bkpap_wireup_syncstructure(void) { return OPAL_ERR_NOT_IMPLEMENTED; }