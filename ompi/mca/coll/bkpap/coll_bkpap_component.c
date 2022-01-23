#include "coll_bkpap.h"

const char* mca_coll_bkpap_component_version_string =
"Open MPI cuda collective MCA component version " OMPI_VERSION;

static int bkpap_register(void);

mca_coll_bkpap_component_t mca_coll_bkpap_component = {
    {
        .collm_version = {
            MCA_COLL_BASE_VERSION_2_4_0,
            .mca_component_name = "bkpap",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION),
            .mca_register_component_params = bkpap_register,
        },
        .collm_data = {
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },
        .collm_init_query = mca_coll_bkpap_init_query,
        .collm_comm_query = mca_coll_bkpap_comm_query,
    },

    .ucp_context = NULL,
    .ucp_worker = NULL,
    .ucp_worker_addr = NULL,
    .ucp_worker_addr_len = 0,

    .postbuff_size = BKPAP_POSTBUF_SIZE,
    .allreduce_k_value = 4,
    .allreduce_alg = BKPAP_ALLREDUCE_ALG_KTREE,
    .out_stream = -1,
    .priority = 35,
    .disabled = 0,
};

int mca_coll_bkpap_init_query(bool enable_progress_threads, bool enable_mpi_threads) {
    int ret;
#if OPAL_ENABLE_DEBUG
    if (ompi_coll_base_framework.framework_verbose) {
        mca_coll_bkpap_component.out_stream = opal_output_open(NULL);
    }
#endif  /* OPAL_ENABLE_DEBUG */

    // TODO: this isn't the place to do this, it's bad form to do allocations in init_query
    // a proper solution would involve ref-counters and construction/destruction with modules
    ret = mca_coll_bkpap_init_ucx(enable_mpi_threads);
    if (OMPI_SUCCESS != ret) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    return OMPI_SUCCESS;
}

static int bkpap_register(void) {
    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "allreduce_k_value", "Fannout of inter-node tree in allreduce",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.allreduce_k_value);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "allreduce_alg", "Select pap-awareness alg for inter-stage allreduce, {0:ktree, 1:RSA}",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.allreduce_alg);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "priority", "Priority of the component",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.priority);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "disabled", "Turn bkpap off",
        MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.disabled);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "postbuff_size", "Size of preposted buffer, default 64MB",
        MCA_BASE_VAR_TYPE_UINT64_T, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.postbuff_size);

    return OMPI_SUCCESS;
}