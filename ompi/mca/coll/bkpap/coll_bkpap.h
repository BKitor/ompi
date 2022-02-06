#ifndef MCA_COLL_BKPAP_EXPORT_H
#define MCA_COLL_BKPAP_EXPORT_H

#include "ompi_config.h"
#include "mpi.h"

#include "opal/class/opal_object.h"

#include "ompi/communicator/communicator.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/op/op.h"

#include <ucp/api/ucp.h>

BEGIN_C_DECLS

#define BKPAP_MSETZ(_obj) memset(&_obj, 0, sizeof(_obj)) 
#define BKPAP_OUTPUT(_str,...) OPAL_OUTPUT_VERBOSE((9, ompi_coll_base_framework.framework_output,"%s line %d: "_str, __FILE__, __LINE__, ##__VA_ARGS__))
#define BKPAP_PROFILE(_str,...) OPAL_OUTPUT_VERBOSE((5, ompi_coll_base_framework.framework_output," BKPAP_PROFILE: %.8f "_str, MPI_Wtime(), ##__VA_ARGS__))
#define BKPAP_ERROR(_str,...) BKPAP_OUTPUT("ERROR "_str, ##__VA_ARGS__)
#define BKPAP_POSTBUF_SIZE (1<<26)
#define BKPAP_SEGMENT_SIZE (1<<22)
#define BKPAP_OUTPUT_VARS(...) // would be cool if I had a function that takes a list of local vars, generates a string, and calls BKPAP_OUPUT

enum mca_coll_bkpap_dbell_state {
	BKPAP_DBELL_UNSET = -1,
	BKPAP_DBELL_SET = 1,
};

int mca_coll_bkpap_init_query(bool enable_progress_threads,
	bool enable_mpi_threads);

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority);

int mca_coll_bkpap_module_enable(mca_coll_base_module_t* moduel, struct ompi_communicator_t* comm);

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module);


// for programming/ucp_mem_map sanity, this is just
typedef struct mca_coll_bkpap_local_syncstruct_t {
	ucp_mem_h counter_mem_h; // single int64_t
	ucp_mem_attr_t counter_attr;
	ucp_mem_h arrival_arr_mem_h; // array of int64_t
	ucp_mem_attr_t arrival_arr_attr;
} mca_coll_bkpap_local_syncstruct_t;

typedef struct mca_coll_bkpap_remote_syncstruct_t {
	int ss_counter_len;
	int ss_arrival_arr_len;
	int64_t* ss_arrival_arr_offsets;
	uint64_t counter_addr;
	ucp_rkey_h counter_rkey;
	uint64_t arrival_arr_addr;
	ucp_rkey_h arrival_arr_rkey;

} mca_coll_bkpap_remote_syncstruct_t;

typedef struct mca_coll_bkpap_local_postbuf_t {
	int num_buffs;
	ucp_mem_h dbell_h;
	ucp_mem_attr_t dbell_attrs;
	ucp_mem_h postbuf_h;
	ucp_mem_attr_t postbuf_attrs;
} mca_coll_bkpap_local_postbuf_t;

typedef struct mca_coll_bkpap_remote_postbuf_t {
	ucp_rkey_h* dbell_rkey_arr; // mpi_wsize array of dbell-buffers for each rank
	uint64_t* dbell_addr_arr;
	ucp_rkey_h* buffer_rkey_arr;// mpi_wsize array of postbuf-sized buffers for each rank
	uint64_t* buffer_addr_arr;
} mca_coll_bkpap_remote_postbuf_t;

typedef struct mca_coll_bkpap_module_t {
	mca_coll_base_module_t super;
	void* endof_super; // clever/hacky solution for memory allocation, see mca_coll_bkpap_module_construct for use, better solution migth exist
	// could just use fallback_allreduce_module, but this is more portable/easier to understand

	mca_coll_base_module_t* fallback_allreduce_module;
	mca_coll_base_module_allreduce_fn_t fallback_allreduce;

	int32_t wsize;
	int32_t rank; // these are saved for wiredown_ep

	ompi_communicator_t* inter_comm;
	ompi_communicator_t* intra_comm;

	int ucp_is_initialized;

	ucp_ep_h* ucp_ep_arr;

	mca_coll_bkpap_local_postbuf_t local_pbuffs;
	mca_coll_bkpap_remote_postbuf_t remote_pbuffs;

	int num_syncstructures;
	mca_coll_bkpap_local_syncstruct_t* local_syncstructure;
	mca_coll_bkpap_remote_syncstruct_t* remote_syncstructure;

} mca_coll_bkpap_module_t;

OBJ_CLASS_DECLARATION(mca_coll_bkpap_module_t);

typedef struct mca_coll_bkpap_component_t {
	mca_coll_base_component_t super;

	ucp_context_h ucp_context;
	ucp_worker_h ucp_worker;
	ucp_address_t* ucp_worker_addr;
	size_t ucp_worker_addr_len;

	size_t postbuff_size;
	size_t pipeline_segment_size;
	int allreduce_k_value;
	int allreduce_alg;
	int priority;
	int disabled;
} mca_coll_bkpap_component_t;

OMPI_MODULE_DECLSPEC extern mca_coll_bkpap_component_t mca_coll_bkpap_component;

typedef struct mca_coll_bkpap_req_t {
	ucs_status_t ucs_status;
	int complete;
} mca_coll_bkpap_req_t;

void mca_coll_bkpap_req_init(void* request);

int mca_coll_bkpap_init_ucx(int enable_mpi_threads);
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_postbuffs(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_syncstructure(int num_counters, int num_arrival_slots, int num_structures, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_hier_comms(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);

int mca_coll_bkpap_arrive_ss(mca_coll_bkpap_module_t* module, int64_t ss_rank, int counter_offset, int arrival_arr_offset, struct ompi_communicator_t* comm, int64_t* ret_pos);
int mca_coll_bkpap_leave_ss(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm, int arrival);
// int mca_coll_bkpap_get_rank_of_arrival(int arrival, int sync_round, mca_coll_bkpap_module_t* module, int* rank);
int mca_coll_bkpap_get_rank_of_arrival(int arrival, int arival_round_offset, mca_coll_bkpap_module_t* module, int* rank);
int mca_coll_bkpap_put_postbuf(const void* buf, struct ompi_datatype_t* dtype, int count, int send_rank, int slot, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int mca_coll_bkpap_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count, ompi_op_t* op, int num_buffers, mca_coll_bkpap_module_t* module);

enum mca_coll_bkpap_allreduce_algs {
	BKPAP_ALLREDUCE_ALG_KTREE = 0,
	BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE = 1,
	BKPAP_ALLREDUCE_ALG_RSA = 2,
	BKPAP_ALLREDUCE_ALG_COUNT
};

END_C_DECLS
#endif