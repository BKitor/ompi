
static inline int _bk_intra_reduce(void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* intra_comm, mca_coll_bkpap_module_t* bkpap_module) {
    int intra_rank = ompi_comm_rank(intra_comm);

    void* intra_reduce_sbuf = (0 == intra_rank) ? MPI_IN_PLACE : rbuf;
    void* intra_reduce_rbuf = (0 == intra_rank) ? rbuf : NULL;

    switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
    case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
        return intra_comm->c_coll->coll_reduce(
            intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0,
            intra_comm,
            intra_comm->c_coll->coll_reduce_module);
        break;
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
        return mca_coll_bkpap_reduce_intra_inplace_binomial(intra_reduce_sbuf, intra_reduce_rbuf, count, dtype, op, 0, intra_comm, bkpap_module);
        break;
    default:
        BKPAP_ERROR("Bad memory type, intra-node reduce failed");
        return OMPI_ERROR;
        break;
    }
}

static inline int _bk_int_pow(int base, int exp) {
    int res = 1;
    while (1) {
        if (exp & 1)
            res *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
    return res;
}

// floor of log base k
static inline int _bk_log_k(int k, int n) {
    int ret = 0;
    for (int i = 1; i < n; i *= k)
        ret++;
    return ret;
}


static inline int bk_get_pbuff(void** buf, mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    if (OPAL_UNLIKELY(NULL == bkpap_module->local_pbuffs.tag.buff_arr)) {
        ret = bk_alloc_pbufft(&bkpap_module->local_pbuffs.tag.buff_arr, mca_coll_bkpap_component.postbuff_size);
        BKPAP_OUTPUT("ALLOC_TMP_BUF: %p", bkpap_module->local_pbuffs.tag.buff_arr);
    }
    *buf = bkpap_module->local_pbuffs.tag.buff_arr;
    BKPAP_OUTPUT("TMP_BUF: %p", bkpap_module->local_pbuffs.tag.buff_arr);
    return ret;
}

static inline int bkpap_get_mempool(void** ptr, size_t size, mca_coll_bkpap_module_t* bkpap_module) {
	bkpap_mempool_t* m = &bkpap_module->mempool;

	m->offset += 1;
	
	if(OPAL_UNLIKELY(size > m->partition_size)){
		BKPAP_ERROR("requested buffer larger than available (size: %ld, avail: %ld)", size, m->partition_size);
		return OMPI_ERR_BAD_PARAM;
	}

	if (OPAL_UNLIKELY(m->offset >= m->num_partitions)) {
		BKPAP_ERROR("Requested to many resources");
		return OMPI_ERROR;
	}

	if (OPAL_UNLIKELY(NULL == m->buff[m->offset])) {
		int ret = bk_alloc_pbufft(&m->buff[m->offset], m->partition_size);
		if (OMPI_SUCCESS != ret) {
			BKPAP_ERROR("bk_alloc_pbufft failed");
			return ret;
		}
	}
	
	*ptr = m->buff[m->offset];
	return OMPI_SUCCESS;
}

static inline int bkpap_reset_mempool(mca_coll_bkpap_module_t* bkpap_module) {
	bkpap_mempool_t* m = &bkpap_module->mempool;
	m->offset = -1;
	return OMPI_SUCCESS;
}