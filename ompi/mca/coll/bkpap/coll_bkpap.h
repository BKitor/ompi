#ifndef MCA_COLL_BKPAP_EXPORT_H
#define MCA_COLL_BKPAP_EXPORT_H

#include "ompi_config.h"
#include "mpi.h"

#include "opal/class/opal_object.h"

#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/communicator/communicator.h"

BEGIN_C_DECLS

int mca_coll_bkpap_init_query(bool enable_progress_threads,
	bool enable_mpi_threads);

mca_coll_base_module_t* mca_coll_bkpap_comm_query(struct ompi_communicator_t* comm, int* priority);

int mca_coll_bkpap_module_enable(mca_coll_base_module_t* moduel, struct ompi_communicator_t* comm);

int mca_coll_bkpap_allreduce(const void* sbuf, void* rbuf, int count,
	struct ompi_datatype_t* dtype,
	struct ompi_op_t* op,
	struct ompi_communicator_t* comm,
	mca_coll_base_module_t* module);


typedef struct mca_coll_bkpap_module_t {
	mca_coll_base_module_t super;
	
	mca_coll_base_module_2_4_0_t *fallback_allreduce_module;
	mca_coll_base_module_allreduce_fn_t fallback_allreduce;
	
} mca_coll_bkpap_module_t;

OBJ_CLASS_DECLARATION(mca_coll_bkpap_module_t);

typedef struct mca_coll_bkpap_component_t {
	mca_coll_base_component_2_4_0_t super;

	int priority;
	int disabled;
} mca_coll_bkpap_component_t;

OMPI_MODULE_DECLSPEC extern mca_coll_bkpap_component_t mca_coll_bkpap_component;

END_C_DECLS

#endif