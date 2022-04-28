#ifndef MCA_COLL_BKPAP_EXPORT_H
#define MCA_COLL_BKPAP_EXPORT_H

#include "ompi_config.h"
#include "mpi.h"

#include "bkpap_kernel.h"

#include "opal/class/opal_object.h"

#include "ompi/communicator/communicator.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/op/op.h"

#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

#include <ucp/api/ucp.h>

BEGIN_C_DECLS

#define BKPAP_MSETZ(_obj) memset(&_obj, 0, sizeof(_obj)) 
#define BKPAP_OUTPUT(_str,...) OPAL_OUTPUT_VERBOSE((9, ompi_coll_base_framework.framework_output,"BKOUT %s line %d: "_str, __FILE__, __LINE__, ##__VA_ARGS__))
#define BKPAP_PROFILE(_str,...) OPAL_OUTPUT_VERBOSE((5, ompi_coll_base_framework.framework_output," BKPAP_PROFILE: %.8f rank: %d "_str, MPI_Wtime(), ##__VA_ARGS__))
#define BKPAP_ERROR(_str,...) BKPAP_OUTPUT("ERROR "_str, ##__VA_ARGS__)
#define BKPAP_CHK_MALLOC(_buf, _lbl) if(OPAL_UNLIKELY(NULL == _buf)){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto _lbl;}
#define BKPAP_CHK_UCP(_status, _lbl) if(OPAL_UNLIKELY(UCS_OK != _status)){BKPAP_ERROR("UCP op failed, going to "#_lbl); ret = OMPI_ERROR; goto _lbl;}
#define BKPAP_CHK_MPI(_ret, _lbl) if(OPAL_UNLIKELY(OMPI_SUCCESS != _ret)){BKPAP_ERROR("MPI op failed, going to "#_lbl); goto _lbl;}
#define BKPAP_CHK_CUDA(_ret, _lbl) if(OPAL_UNLIKELY(cudaSuccess != _ret)){BKPAP_ERROR("CUDA op failed, going to "#_lbl); goto _lbl;}
#define BKPAP_POSTBUF_SIZE (1<<26)
#define BKPAP_SEGMENT_SIZE (1<<22)
#define BKPAP_OUTPUT_VARS(...) // would be cool if I had a function that takes a list of local vars, generates a string, and calls BKPAP_OUPUT

enum mca_coll_bkpap_dbell_state {
	BKPAP_DBELL_UNSET = -1,
	BKPAP_DBELL_SET = 1,
};

typedef enum mca_coll_bkpap_allreduce_algs_t {
	BKPAP_ALLREDUCE_ALG_KTREE = 0,
	BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE = 1,
	BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE = 2,
	BKPAP_ALLREDUCE_ALG_RSA = 3,
	BKPAP_ALLREDUCE_ALG_COUNT
} mca_coll_bkpap_allreduce_algs_t;

typedef enum mca_coll_bkpap_postbuf_memory_t {
	BKPAP_POSTBUF_MEMORY_TYPE_HOST = 0,
	BKPAP_POSTBUF_MEMORY_TYPE_CUDA = 1,
	BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED = 2,
	BKPAP_POSTBUF_MEMORY_TYPE_COUNT
} mca_coll_bkpap_postbuf_memory_t;

typedef enum mca_coll_bkpap_dataplane_t {
	BKPAP_DATAPLANE_RMA = 0,
	BKPAP_DATAPLANE_TAG = 1,
	BKPAP_DATAPLANE_COUNT
}mca_coll_bkpap_dataplane_t;

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

typedef struct mca_coll_bkpap_local_rma_postbuf_t {
	int num_buffs;
	ucp_mem_h dbell_h;
	ucp_mem_attr_t dbell_attrs;
	ucp_mem_h postbuf_h;
	ucp_mem_attr_t postbuf_attrs;
} mca_coll_bkpap_local_rma_postbuf_t;

typedef struct mca_coll_bkpap_remote_rma_postbuf_t {
	ucp_rkey_h* dbell_rkey_arr; // mpi_wsize array of dbell-buffers for each rank
	uint64_t* dbell_addr_arr;
	ucp_rkey_h* buffer_rkey_arr;// mpi_wsize array of postbuf-sized buffers for each rank
	uint64_t* buffer_addr_arr;
} mca_coll_bkpap_remote_rma_postbuf_t;

typedef struct mca_coll_bkpap_local_tag_postbuf_t {
	void* buff_arr;
	size_t buff_size;
	int num_buffs;
	mca_coll_bkpap_postbuf_memory_t mem_type;
} mca_coll_bkpap_local_tag_postbuf_t;

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

	union {
		mca_coll_bkpap_local_rma_postbuf_t rma;
		mca_coll_bkpap_local_tag_postbuf_t tag;
	} local_pbuffs;
	mca_coll_bkpap_remote_rma_postbuf_t remote_pbuffs;


	int num_syncstructures; // array of ss for pipelining
	mca_coll_bkpap_local_syncstruct_t* local_syncstructure;
	mca_coll_bkpap_remote_syncstruct_t* remote_syncstructure;

} mca_coll_bkpap_module_t;

OBJ_CLASS_DECLARATION(mca_coll_bkpap_module_t);

typedef struct mca_coll_bkpap_component_t {
	mca_coll_base_component_t super;

	int enable_threads;
	ucp_context_h ucp_context;
	ucp_worker_h ucp_worker;
	ucp_address_t* ucp_worker_addr;
	size_t ucp_worker_addr_len;

	size_t postbuff_size;
	size_t pipeline_segment_size;
	int allreduce_k_value;
	int allreduce_alg;
	int priority;

	mca_coll_bkpap_dataplane_t dataplane_type;

	mca_coll_bkpap_postbuf_memory_t bk_postbuf_memory_type;
	ucs_memory_type_t ucs_postbuf_memory_type;
} mca_coll_bkpap_component_t;

OMPI_MODULE_DECLSPEC extern mca_coll_bkpap_component_t mca_coll_bkpap_component;

typedef struct mca_coll_bkpap_req_t {
	ucs_status_t ucs_status;
	int complete;
} mca_coll_bkpap_req_t;

void mca_coll_bkpap_req_init(void* request);

int mca_coll_bkpap_lazy_init_module_ucx(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm, int alg);
int mca_coll_bkpap_init_ucx(int enable_mpi_threads);
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_syncstructure(int num_counters, int num_arrival_slots, int num_structures, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_hier_comms(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);

int mca_coll_bkpap_arrive_ss(int64_t ss_rank, uint64_t counter_offset, uint64_t arrival_arr_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm, int64_t* ret_pos);
int mca_coll_bkpap_leave_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_get_rank_of_arrival(int arrival, uint64_t arival_round_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, int* rank);
int mca_coll_bkpap_reset_remote_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);

int mca_coll_bkpap_rma_wireup(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_rma_send_postbuf(const void* buf, struct ompi_datatype_t* dtype, int count, int dest, int slot, struct ompi_communicator_t* comm, mca_coll_base_module_t* module);
int mca_coll_bkpap_rma_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count, ompi_op_t* op, int num_buffers, ompi_communicator_t* comm, mca_coll_base_module_t* module);

int mca_coll_bkpap_tag_wireup(int num_bufs, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_tag_send_postbuf(const void* buf, struct ompi_datatype_t* dtype, int count, int dest, int slot, struct ompi_communicator_t* comm, mca_coll_base_module_t* module);
int mca_coll_bkpap_tag_reduce_postbufs(void* local_buf, struct ompi_datatype_t* dtype, int count, ompi_op_t* op, int num_buffers, ompi_communicator_t* comm, mca_coll_base_module_t* module);

int mca_coll_bkpap_reduce_intra_inplace_binomial(void* buf, int count, ompi_datatype_t* datatype, ompi_op_t* op, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module, uint32_t segsize, int max_outstanding_reqs);
int mca_coll_bkpap_reduce_generic(const void* sendbuf, void* recvbuf, int original_count, ompi_datatype_t* datatype, ompi_op_t* op, int root, ompi_communicator_t* comm, mca_coll_base_module_t* module, ompi_coll_tree_t* tree, int count_by_segment, int max_outstanding_reqs);

END_C_DECLS
#endif
