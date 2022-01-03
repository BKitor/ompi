#ifndef MCA_COLL_BKPAP_EXPORT_H
#define MCA_COLL_BKPAP_EXPORT_H

#include "ompi_config.h"
#include "mpi.h"

#include "opal/class/opal_object.h"

#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/communicator/communicator.h"

#include <ucp/api/ucp.h>

BEGIN_C_DECLS

#define BKPAP_MSETZ(_obj) memset(&_obj, 0, sizeof(_obj)) 
#define BKPAP_OUTPUT(_str,...) opal_output(mca_coll_bkpap_component.out_stream,"%s line %d: "_str, __FILE__, __LINE__, ##__VA_ARGS__)
#define BKPAP_ERROR(_str,...) BKPAP_OUTPUT("ERROR"_str, ##__VA_ARGS__)
#define BKPAP_POSTBUF_SIZE (1<<26)

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
typedef struct mca_coll_bkpap_syncstruct_t{
	ucp_mem_h counter_mem_h;
	ucp_mem_attr_t counter_attr;
	ucp_mem_h arrival_arr_mem_h;
	ucp_mem_attr_t arrival_arr_attr;
} mca_coll_bkpap_syncstruct_t;

typedef struct mca_coll_bkpap_module_t {
	mca_coll_base_module_t super;
	
	mca_coll_base_module_t *fallback_allreduce_module;
	mca_coll_base_module_allreduce_fn_t fallback_allreduce;
	
	int32_t wsize;
	int32_t rank; // these are saved for wiredown_ep
	ucp_ep_h *ucp_ep_arr;

	ucp_mem_h local_postbuf_h;
	ucp_mem_attr_t local_postbuf_attrs;

	uint64_t *remote_postbuff_addr_arr;
	ucp_rkey_h *remote_postbuff_rkey_arr;
	
	ompi_communicator_t *inter_comm;
	ompi_communicator_t *intra_comm;

	mca_coll_bkpap_syncstruct_t *local_syncstructure;
	uint64_t remote_syncstructure_counter_addr;
	ucp_rkey_h remote_syncstructure_counter_rkey;
	uint64_t remote_syncstructure_arrival_addr;
	ucp_rkey_h remote_syncstructure_arrival_rkey;
} mca_coll_bkpap_module_t;

OBJ_CLASS_DECLARATION(mca_coll_bkpap_module_t);

typedef struct mca_coll_bkpap_component_t {
	mca_coll_base_component_t super;
	
	ucp_context_h ucp_context;
	ucp_worker_h ucp_worker;
	ucp_address_t *ucp_worker_addr;
	size_t ucp_worker_addr_len;

	uint64_t postbuff_size;
	int allreduce_k_value;
	int out_stream;
	int priority;
	int disabled;
} mca_coll_bkpap_component_t;

OMPI_MODULE_DECLSPEC extern mca_coll_bkpap_component_t mca_coll_bkpap_component;

typedef struct mca_coll_bkpap_amoreq_t{
	ucs_status_t ucs_status;
	int complete;
} mca_coll_bkpap_amoreq_t;

void mca_coll_bkpap_amoreq_init(void *request);

int mca_coll_bkpap_init_ucx(int enable_mpi_threads);
int mca_coll_bkpap_wireup_endpoints(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_remote_postbuffs(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wireup_syncstructure(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_wirup_hier_comms(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm);
int mca_coll_bkpap_arrive_at_inter(mca_coll_bkpap_module_t* module, struct ompi_communicator_t* comm, int64_t* ret_pos); // can drop the 'comm' param

END_C_DECLS

#endif