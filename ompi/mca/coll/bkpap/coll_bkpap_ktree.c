#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"
#include "opal/cuda/common_cuda.h"

int coll_bkpap_papaware_ktree_allreduce_fullpipelined(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {
    BKPAP_ERROR("The ktree algorithms were removed");
    return OMPI_ERR_NOT_SUPPORTED;
}

int coll_bkpap_papaware_ktree_allreduce_pipelined(const void* sbuf, void* rbuf, int count,
    struct ompi_datatype_t* dtype, struct ompi_op_t* op, size_t seg_size,
    struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
    mca_coll_bkpap_module_t* bkpap_module) {

    BKPAP_ERROR("The ktree algorithms were removed");
    return OMPI_ERR_NOT_SUPPORTED;

}
