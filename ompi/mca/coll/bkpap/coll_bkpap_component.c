#include "coll_bkpap.h"

const char* mca_coll_bkpap_component_version_string =
"Open MPI cuda collective MCA component version " OMPI_VERSION;

static int bkpap_register(void);
int mca_coll_bkpap_init_ucx(int enable_mpi_threads);

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

    .out_stream = -1,
    .priority = 30,
    .disabled = 0,
};

int mca_coll_bkpap_init_ucx(int enable_mpi_threads){
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_config_t *config;
    ucs_status_t status;
    
    BKPAP_MSETZ(ucp_params);
    BKPAP_MSETZ(worker_params);
    
    status = ucp_config_read("MPI", NULL, &config);
    if (UCS_OK != status) {
        return OMPI_ERROR;
    }

    ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                               UCP_PARAM_FIELD_REQUEST_SIZE |
                               UCP_PARAM_FIELD_REQUEST_INIT |
                               UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                               UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    ucp_params.features          = UCP_FEATURE_AMO64;
    ucp_params.request_size      = sizeof(mca_coll_bkpap_amoreq_t);
    ucp_params.request_init      = mca_coll_bkpap_amoreq_init;
    ucp_params.mt_workers_shared = 0; /* we do not need mt support for context
                                     since it will be protected by worker */
    ucp_params.estimated_num_eps = ompi_proc_world_size();

#if HAVE_DECL_UCP_PARAM_FIELD_ESTIMATED_NUM_PPN
    ucp_params.estimated_num_ppn = opal_process_info.num_local_peers + 1;
    ucp_params.field_mask       |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
#endif

    status = ucp_init(&ucp_params, config, &mca_coll_bkpap_component.ucp_context);
    ucp_config_release(config);

    if (UCS_OK != status) {
        return OMPI_ERROR;
    }
    
	worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = (enable_mpi_threads == MPI_THREAD_SINGLE) ? UCS_THREAD_MODE_SINGLE : UCS_THREAD_MODE_MULTI;
	status = ucp_worker_create(mca_coll_bkpap_component.ucp_context, &worker_params, &mca_coll_bkpap_component.ucp_worker);

    if (UCS_OK != status) {
        ucp_cleanup(mca_coll_bkpap_component.ucp_context);
        return OMPI_ERROR;
    }
    
    status = ucp_worker_get_address(
        mca_coll_bkpap_component.ucp_worker,
        &mca_coll_bkpap_component.ucp_worker_addr,
        &mca_coll_bkpap_component.ucp_worker_addr_len
    );

    if (UCS_OK != status) {
        ucp_cleanup(mca_coll_bkpap_component.ucp_context);
        return OMPI_ERROR;
    }

    return OMPI_SUCCESS;
}


int mca_coll_bkpap_init_query(bool enable_progress_threads, bool enable_mpi_threads) {
    int ret;
#if OPAL_ENABLE_DEBUG
    if (ompi_coll_base_framework.framework_verbose) {
        mca_coll_bkpap_component.out_stream = opal_output_open(NULL);
    }
#endif  /* OPAL_ENABLE_DEBUG */

    ret = mca_coll_bkpap_init_ucx(enable_mpi_threads);
    if (OMPI_SUCCESS != ret){
        return OMPI_ERR_NOT_SUPPORTED;
    }
    
    return OMPI_SUCCESS;
}

void mca_coll_bkpap_amoreq_init(void *request){
    mca_coll_bkpap_amoreq_t *r = request;
    r->ucs_status = UCS_OK;
    r->complete = 0;
}

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