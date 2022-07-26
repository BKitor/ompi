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

#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

BEGIN_C_DECLS

#define BKPAP_MSETZ(_obj) memset(&_obj, 0, sizeof(_obj)) 
#define BKPAP_OUTPUT(_str,...) OPAL_OUTPUT_VERBOSE((9, mca_coll_bkpap_output,"BKOUT %s line %d: "_str, __FILE__, __LINE__, ##__VA_ARGS__))
#define BKPAP_PROFILE(_str,...) 
// #define BKPAP_PROFILE(_str,...) OPAL_OUTPUT_VERBOSE((5, mca_coll_bkpap_output," BKPAP_PROFILE: %.8f rank: %d "_str, MPI_Wtime(), ##__VA_ARGS__))
#define BKPAP_ERROR(_str,...) BKPAP_OUTPUT("ERROR "_str, ##__VA_ARGS__)
#define BKPAP_CHK_MALLOC(_buf, _lbl) if(OPAL_UNLIKELY(NULL == _buf)){BKPAP_ERROR("malloc "#_buf" returned NULL"); goto _lbl;}
#define BKPAP_CHK_UCP(_status, _lbl) if(OPAL_UNLIKELY(UCS_OK != _status)){BKPAP_ERROR("UCP op failed, status %d (%s) going to %s", _status, ucs_status_string(_status), #_lbl); ret = OMPI_ERROR; goto _lbl;}
#define BKPAP_CHK_MPI(_ret, _lbl) if(OPAL_UNLIKELY(OMPI_SUCCESS != _ret)){BKPAP_ERROR("MPI op failed, going to "#_lbl); goto _lbl;}
#define BKPAP_CHK_MPI_MSG_LBL(_ret, _msg, _lbl) if(OPAL_UNLIKELY(OMPI_SUCCESS != _ret)){BKPAP_ERROR(_msg", going to"#_lbl); goto _lbl;}
#define BKPAP_CHK_CUDA(_ret, _lbl) if(OPAL_UNLIKELY(cudaSuccess != _ret)){BKPAP_ERROR("CUDA op failed, going to "#_lbl); goto _lbl;}
#define BKPAP_POSTBUF_SIZE (1<<26)
#define BKPAP_SEGMENT_SIZE (1<<22)
#define BKPAP_OUTPUT_VARS(...) // would be cool if I had a function that takes a list of local vars, generates a string, and calls BKPAP_OUPUT

/* RSA tag format:
 * |  63 --- 2  |   1   |     0     |
 * | comm_round | RS/AG | data/rank |
 * |   Round    | phase |    type   |
 *
 * Round and phase are populated by _bk_papaware_rsa_allreduce, and type is populated by bk_ucp_p2p
 * BK_RSA_MAKE_TAG calls opal_hibit with staring pos at 16, which will break if comm_size > 2^18 (256K proc)
*/
#define BK_RSA_MAKE_TAG(_tag, _tag_mask, _num_rounds, _round, _phase) { \
        _tag_mask = ((uint64_t)((( 1ul << opal_hibit(_num_rounds, 18)) - 1 ) << 3 ) | 7ul); \
        _tag = (((uint64_t)(_round) << 2 ) | ((uint64_t)(_phase) << 1)); \
    }
#define BK_RSA_SET_TAG_TYPE_DATA(_tag) (_tag | 1ul)
#define BK_RSA_SET_TAG_TYPE_RANK(_tag) (_tag & ~(1ul))

/* Binomial tag format:
 * | 63 --- 2 |   1    |     0     |
 * |   child  | RED/BC | data/rank |
 * | arrival  |  phase |    type   |
*/
#define BK_BINOMIAL_MAKE_TAG(_arrival, _phase, _tag, _tag_mask) { \
	_tag_mask = 0xfffful; \
	_tag = ((uint64_t)(_arrival)<<2) | ((uint64_t)(_phase) << 1); \
	}
#define BK_BINOMIAL_MAKE_LAST_PROC_TAG(_wsize, _phase, _tag, _tag_mask){ \
	BK_BINOMIAL_MAKE_TAG((_wsize - 1), _phase, _tag, _tag_mask); \
}
#define BK_BINOMIAL_TAG_SET_DATA(_tag) (_tag | 1ul)
#define BK_BINOMIAL_TAG_SET_RANK(_tag) (_tag & ~(1ul))
#define BK_BINOMIAL_LAST_PROC_TAG(_wsize, _phase) (((~(3lu)) & ((_wsize - 1)<<2)) | 1lu)
extern int mca_coll_bkpap_output;

enum bkpap_dbell_state {
	BKPAP_DBELL_EMPTY = 0,
	BKPAP_DBELL_INUSE = 1,
	BKPAP_DBELL_FULL = 2,
};

typedef enum bkpap_allreduce_algs_t {
	BKPAP_ALLREDUCE_ALG_KTREE = 0,
	BKPAP_ALLREDUCE_ALG_KTREE_PIPELINE = 1,
	BKPAP_ALLREDUCE_ALG_KTREE_FULLPIPE = 2,
	BKPAP_ALLREDUCE_ALG_RSA = 3,
	BKPAP_ALLREDUCE_ALG_BASE_RSA_GPU = 4,
	BKPAP_ALLREDUCE_ALG_BINOMIAL = 5,
	BKPAP_ALLREDUCE_ALG_CHAIN = 6,
	BKPAP_ALLREDUCE_ALG_COUNT
} bkpap_allreduce_algs_t;

typedef enum bkpap_dplane_mem_t {
	BKPAP_DPLANE_MEM_TYPE_HOST = 0,
	BKPAP_DPLANE_MEM_TYPE_CUDA = 1,
	BKPAP_DPLANE_MEM_TYPE_CUDA_MANAGED = 2,
	BKPAP_DPLANE_MEM_TYPE_COUNT
} bkpap_dplane_mem_t;

typedef enum bkpap_dplane_t {
	BKPAP_DPLANE_RMA = 0,
	BKPAP_DPLANE_TAG = 1,
	BKPAP_DPLANE_COUNT
} bkpap_dplane_t;

typedef struct mca_coll_bkpap_module_t mca_coll_bkpap_module_t; // forward decl
/*Typedefs for the communication functions*/
typedef int (*mca_coll_bkpap_dplane_send_to_early_ft)(void* send_buf, int send_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_send_to_late_ft)(void* send_buf, int send_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_recv_from_early_ft)(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_recv_from_late_ft)(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_sendrecv_from_early_ft)(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_sendrecv_from_late_ft)(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_sendrecv_ft)(void* sbuf, int send_count, void* rbuf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
typedef int (*mca_coll_bkpap_dplane_reset_late_recv_ft)(mca_coll_bkpap_module_t* module);

typedef struct coll_bkpap_dplane_ftbl_t {
	mca_coll_bkpap_dplane_send_to_early_ft send_to_early;
	mca_coll_bkpap_dplane_send_to_late_ft send_to_late;
	mca_coll_bkpap_dplane_recv_from_early_ft recv_from_early;
	mca_coll_bkpap_dplane_recv_from_late_ft recv_from_late;
	mca_coll_bkpap_dplane_sendrecv_from_early_ft  sendrecv_from_early;
	mca_coll_bkpap_dplane_sendrecv_from_late_ft sendrecv_from_late;
	mca_coll_bkpap_dplane_sendrecv_ft sendrecv;
	mca_coll_bkpap_dplane_reset_late_recv_ft reset_late_recv_buf;
}coll_bkpap_dplane_ftbl_t;

int mca_coll_bkpap_init_query(bool enable_progress_threads,
	bool enable_mpi_threads);

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority);

int mca_coll_bkpap_module_enable(mca_coll_base_module_t* moduel, struct ompi_communicator_t* comm);

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module);

typedef struct mca_coll_bkpap_req_t {
	ucs_status_t ucs_status;
	int complete;
} mca_coll_bkpap_req_t;

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

typedef struct mca_coll_bkpap_rma_dplane_t {
	struct {
		ucp_mem_h dbell_h;
		ucp_mem_attr_t dbell_attrs;
		ucp_mem_h postbuf_h;
		ucp_mem_attr_t postbuf_attrs;
	} local;
	struct {
		ucp_rkey_h* dbell_rkey_arr; // mpi_wsize array of dbell-buffers for each rank
		uint64_t* dbell_addr_arr;
		ucp_rkey_h* buffer_rkey_arr;// mpi_wsize array of postbuf-sized buffers for each rank
		uint64_t* buffer_addr_arr;
	} remote;
} mca_coll_bkpap_rma_dplane_t;

typedef struct mca_coll_bkpap_tag_dplane_t {
	void* buff_arr;
	size_t buff_size;
	bkpap_dplane_mem_t mem_type;
	int prepost_req_set;
	mca_coll_bkpap_req_t* prepost_req;
} mca_coll_bkpap_tag_dplane_t;

typedef struct bkpap_mempool_buf {
	void* buf;
	struct bkpap_mempool_buf* next;
	size_t size;
	int num_passes;
	bool allocated;
}bkpap_mempool_buf_t;

typedef struct bkpap_mempool {
	bkpap_mempool_buf_t* head;
	bkpap_dplane_mem_t memtype;
} bkpap_mempool_t;


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

	bkpap_mempool_t mempool[BKPAP_DPLANE_MEM_TYPE_COUNT];

	bkpap_dplane_t dplane_t;
	bkpap_dplane_mem_t dplane_mem_t;
	union {
		mca_coll_bkpap_rma_dplane_t rma;
		mca_coll_bkpap_tag_dplane_t tag;
	} dplane;
	coll_bkpap_dplane_ftbl_t dplane_ftbl;

	int num_syncstructures; // array of ss for pipelining
	mca_coll_bkpap_local_syncstruct_t* local_syncstructure;
	mca_coll_bkpap_remote_syncstruct_t* remote_syncstructure;

	cudaStream_t bk_cs[2];
	void* host_pinned_buf;

} mca_coll_bkpap_module_t;

OBJ_CLASS_DECLARATION(mca_coll_bkpap_module_t);

typedef enum bk_progress_t_state_t {
	BK_PROGRESS_T_RUN = 0,
	BK_PROGRESS_T_KILL = 1,
	BK_PROGRESS_T_IDLE = 2,
	BK_PROGRESS_T_ERROR = 3,
}bk_progress_t_state_t;

typedef struct bk_t_args_t{
	int set_cu_ctx;
	CUcontext cu_ctx;
}bk_t_args_t;

void* bk_background_progress_thread(void* bk_t_args_t);

typedef struct mca_coll_bkpap_component_t {
	mca_coll_base_component_t super;

	int enable_threads;
	ucp_context_h ucp_context;
	ucp_worker_h ucp_worker;
	ucp_address_t* ucp_worker_addr;
	size_t ucp_worker_addr_len;

	bk_t_args_t bk_t_args;
	pthread_t progress_tid;
	int progress_thread_flag;

	size_t postbuff_size;
	size_t pipeline_segment_size;
	int allreduce_k_value;
	int allreduce_alg;
	int force_flat;
	int priority;
	int verbose;

	bkpap_dplane_t dplane_t;
	bkpap_dplane_mem_t dplane_mem_t;
} mca_coll_bkpap_component_t;

OMPI_MODULE_DECLSPEC extern mca_coll_bkpap_component_t mca_coll_bkpap_component;


int bk_fill_array_str_ld(size_t arr_len, int64_t* arr, size_t str_limit, char* out_str);

void mca_coll_bkpap_req_init(void* request);

int bk_launch_background_thread(void);
int mca_coll_bkpap_lazy_init_module_ucx(mca_coll_bkpap_module_t* bkpap_module, struct ompi_communicator_t* comm, int alg);
int mca_coll_bkpap_init_ucx(int enable_mpi_threads);
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_syncstructure(int num_counters, int num_arrival_slots, int num_structures, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_hier_comms(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);

int mca_coll_bkpap_arrive_ss(int64_t ss_rank, uint64_t counter_offset, uint64_t arrival_arr_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm, int64_t* ret_pos);
int mca_coll_bkpap_leave_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_get_rank_of_arrival(int arrival, uint64_t arival_round_offset, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* module, int* rank);
int mca_coll_bkpap_reset_remote_ss(mca_coll_bkpap_remote_syncstruct_t* remote_ss, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int mca_coll_bkpap_get_last_arrival(ompi_communicator_t* comm, mca_coll_bkpap_remote_syncstruct_t* remote_ss, mca_coll_bkpap_module_t* bkpap_module, int* rank_ret);

int coll_bkpap_papaware_ktree_allreduce_fullpipelined(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size, struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module);
int coll_bkpap_papaware_ktree_allreduce_pipelined(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size, struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module);
int coll_bkpap_papaware_ktree_allreduce(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module);
int ompi_coll_bkpap_base_allreduce_intra_redscat_allgather_gpu(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_papaware_binomial_allreduce(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module);
int coll_bkpap_papaware_chain_allreduce(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module);

int bkpap_init_mempool(mca_coll_bkpap_module_t* bkpap_module);
int bkpap_finalize_mempool(mca_coll_bkpap_module_t* bkpap_module);

int bk_inter_bcast(void* buf, int count, struct ompi_datatype_t* dtype, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module, uint32_t seg_size);
int bk_intra_bcast(void* buf, int count, struct ompi_datatype_t* dtype, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module);
int coll_bkpap_bcast_intra_generic_gpu(void* buffer, int original_count, struct ompi_datatype_t* datatype, int root, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module, uint32_t count_by_segment, ompi_coll_tree_t* tree);

int bk_intra_reduce(void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module);
int mca_coll_bkpap_reduce_intra_inplace_binomial(const void* sendbuf, void* recvbuf, int count, ompi_datatype_t* datatype, ompi_op_t* op, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module); //, uint32_t segsize, int max_outstanding_reqs);
int mca_coll_bkpap_reduce_generic(const void* sendbuf, void* recvbuf, int original_count, ompi_datatype_t* datatype, ompi_op_t* op, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module, ompi_coll_tree_t* tree, int count_by_segment, int max_outstanding_reqs);

int mca_coll_bkpap_rma_dplane_wireup(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
void mca_coll_bkpap_rma_dplane_destroy(mca_coll_bkpap_rma_dplane_t* rma_dplane, mca_coll_bkpap_module_t* bkpap_module);
int mca_coll_bkpap_tag_dplane_wireup(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
void mca_coll_bkpap_tag_dplane_destroy(mca_coll_bkpap_tag_dplane_t* tag_dplane);

int coll_bkpap_rma_send_to_early(void* send_buf, int send_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_send_to_late(void* send_buf, int send_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_recv_from_early(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_recv_from_late(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_sendrecv_from_early(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_sendrecv_from_late(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_sendrecv(void* sbuf, int send_count, void* rbuf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_rma_reset_late_recv_buf(mca_coll_bkpap_module_t* bkpap_module);

int coll_bkpap_tag_send_to_early(void* send_buf, int send_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_send_to_late(void* send_buf, int send_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_recv_from_early(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_recv_from_late(void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_sendrecv_from_early(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_sendrecv_from_late(void* send_buf, int send_count, void* recv_buf, int recv_count, struct ompi_datatype_t* dtype, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_sendrecv(void* sbuf, int send_count, void* rbuf, int recv_count, struct ompi_datatype_t* dtype, int peer_rank, uint64_t tag, uint64_t tag_mask, ompi_communicator_t* comm, mca_coll_bkpap_module_t* module);
int coll_bkpap_tag_reset_late_recv_buf(mca_coll_bkpap_module_t* bkpap_module);
int coll_bkpap_tag_prepost_recv(ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module);

END_C_DECLS
#endif
