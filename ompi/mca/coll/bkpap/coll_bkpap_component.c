#include "coll_bkpap.h"

const char* mca_coll_bkpap_component_version_string =
"Open MPI cuda collective MCA component version " OMPI_VERSION;

static int mca_coll_bkpap_register(void);
static int mca_coll_bkpap_open(void);
static int mca_coll_bkpap_close(void);

int mca_coll_bkpap_output = -1;

mca_coll_bkpap_component_t mca_coll_bkpap_component = {
    {
        .collm_version = {
            MCA_COLL_BASE_VERSION_2_4_0,
            .mca_component_name = "bkpap",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION),
            .mca_register_component_params = mca_coll_bkpap_register,
            .mca_open_component = mca_coll_bkpap_open,
            .mca_close_component = mca_coll_bkpap_close,
            .mca_query_component = NULL,
        },
        .collm_data = {
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },
        .collm_init_query = mca_coll_bkpap_init_query,
        .collm_comm_query = mca_coll_bkpap_comm_query,
    },

    .enable_threads = MPI_THREAD_SINGLE,
    .ucp_context = NULL,
    .ucp_worker = NULL,
    .ucp_worker_addr = NULL,
    .ucp_worker_addr_len = 0,

    .postbuff_size = BKPAP_POSTBUF_SIZE,
    .pipeline_segment_size = BKPAP_SEGMENT_SIZE,
    .allreduce_k_value = 4,
    .allreduce_alg = BKPAP_ALLREDUCE_ALG_KTREE,
    .force_flat = 0,
    .priority = 35,
    .verbose = 0,
    .dplane_t = BKPAP_DPLANE_TAG,
    .dplane_mem_t = BKPAP_DPLANE_MEM_TYPE_HOST,
};

int mca_coll_bkpap_init_query(bool enable_progress_threads, bool enable_mpi_threads) {

    // TODO: this isn't the place to do this, it's bad form to do allocations in init_query
    // a proper solution would involve ref-counters and construction/destruction with modules
    // ret = mca_coll_bkpap_init_ucx(enable_mpi_threads);
    // if (OMPI_SUCCESS != ret) {
    //     return OMPI_ERR_NOT_SUPPORTED;
    // }
    mca_coll_bkpap_component.enable_threads = enable_mpi_threads;

    return OMPI_SUCCESS;
}

static int mca_coll_bkpap_register(void) {
    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "priority", "Priority of the component",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.priority);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "verbose", "Verbosity of the component",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.verbose);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "postbuff_size", "Size of preposted buffer, default 64MB",
        MCA_BASE_VAR_TYPE_UINT64_T, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.postbuff_size);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "pipeline_segment_size", "Segment size for pipeline",
        MCA_BASE_VAR_TYPE_UINT64_T, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.pipeline_segment_size);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "allreduce_k_value", "Fannout of inter-node tree in allreduce",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.allreduce_k_value);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "allreduce_alg", "Select pap-awareness alg for inter-stage allreduce, {0:ktree, 1:ktree-pipeline, 2:ktree-fullpipe, 3:RSA, 4:base_rsa_gpu}",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.allreduce_alg);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "postbuf_mem_type", "postbuf memory types, {0:Host, 1:CUDA, 2:CUDA Managed}",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.dplane_mem_t);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "dataplane_type", "UCX primative to use for dataplane, {0:RMA, 1:TAG}",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.dplane_t);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "force_flat", "force allreduce alg to be flat instead of hierarchical",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.force_flat);

    return OMPI_SUCCESS;
}

static int mca_coll_bkpap_open(void) {
    mca_coll_bkpap_output = opal_output_open(NULL);
    opal_output_set_verbosity(mca_coll_bkpap_output, mca_coll_bkpap_component.verbose);
    // TODO: Init UCX here?
    return OMPI_SUCCESS;
}
static int mca_coll_bkpap_close(void) {
    // TODO: Close UCX here?
    return OMPI_SUCCESS;
}