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

    // priority
    30,
    // disabled
    0,
};

static int bkpap_register(void) {
    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "priority", "Priority of the component",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.priority);

    (void)mca_base_component_var_register(&mca_coll_bkpap_component.super.collm_version,
        "disabled", "Turn bkpap off",
        MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_6,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_bkpap_component.disabled);
    return OMPI_SUCCESS;
}