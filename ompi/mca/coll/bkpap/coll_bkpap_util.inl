#include "coll_bkpap.h"

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

static int bkpap_mempool_create_buf(bkpap_mempool_buf_t** b, size_t size, bkpap_mempool_t* mempool) {
    int ret = OMPI_SUCCESS;
    bkpap_mempool_buf_t* tmp_b = malloc(sizeof(*tmp_b));
    BKPAP_CHK_MALLOC(tmp_b, bk_abort_mempool_new_buf);
    tmp_b->allocated = false;
    tmp_b->next = NULL;
    tmp_b->num_passes = 0;
    ret = bk_alloc_pbufft(&tmp_b->buf, size, mempool->memtype);
    BKPAP_CHK_MPI(ret, bk_abort_mempool_new_buf);
    tmp_b->size = size;
    *b = tmp_b;
    return OMPI_SUCCESS;

bk_abort_mempool_new_buf:
    BKPAP_ERROR("Allocating new mempool_buf failed");
    return ret;
}

static int bkpap_mempool_destroy_buf(bkpap_mempool_buf_t* b, bkpap_mempool_t *mempool) {
    BKPAP_OUTPUT("DESTROY BUF OF SIZE %ld", b->size);
    bk_free_pbufft(b->buf, mempool->memtype);
    free(b);
    return OMPI_SUCCESS;
}

static inline int bkpap_mempool_alloc(void** ptr, size_t size, mca_coll_bkpap_module_t* bkpap_module) {
    int ret = OMPI_SUCCESS;
    bkpap_mempool_t* m = &bkpap_module->mempool;
    bkpap_mempool_buf_t* b = m->head;
    bkpap_mempool_buf_t* b_prev = NULL;
    if (NULL == m->head) {
        ret = bkpap_mempool_create_buf(&m->head, size, m);
        BKPAP_CHK_MPI(ret, bk_abort_mempool_alloc);
        b = m->head;
        b->allocated = true;
        *ptr = b->buf;
        return OMPI_SUCCESS;
    }
    while (NULL != b) {
        if (!b->allocated && b->size >= size) {
            b->allocated = true;
            b->num_passes = 0;
            *ptr = b->buf;
            return OMPI_SUCCESS;
        }
        b->num_passes++;
        b_prev = b;
        b = b->next;
    }
    ret = bkpap_mempool_create_buf(&b_prev->next, size, m);
    BKPAP_CHK_MPI(ret, bk_abort_mempool_alloc);
    b = b_prev->next;
    b->allocated = true;
    *ptr = b->buf;
    return OMPI_SUCCESS;

bk_abort_mempool_alloc:
    BKPAP_ERROR("Error allocating new mempool buffer");
    return ret;
}

static inline int bk_mempool_free(void* ptr, mca_coll_bkpap_module_t* bkpap_module) {
    bkpap_mempool_t* m = &bkpap_module->mempool;
    bkpap_mempool_buf_t* b = m->head;
    if (NULL == m->head) {
        BKPAP_ERROR("Head of mempool is null, bad free");
        return OMPI_ERROR;
    }
    while (NULL != b) {
        if (ptr == b->buf) {
            b->allocated = false;
            return OMPI_SUCCESS;
        }
        b = b->next;
    }
    BKPAP_ERROR("Reached end of mempool LL without finding buf");
    return OMPI_ERROR;
}

static inline int bk_mempool_trim(mca_coll_bkpap_module_t* bkpap_module) {
    int ret = 0;
    bkpap_mempool_t* m = &bkpap_module->mempool;
    bkpap_mempool_buf_t* b_prev, * b, *b_tmp;
    int pass_cutoff = 5;// should be tunable

    if (NULL == m->head)
        return OMPI_SUCCESS;
    while (m->head->num_passes > pass_cutoff) {
        b = m->head;
        m->head = m->head->next;
        ret = bkpap_mempool_destroy_buf(b, m);
        BKPAP_CHK_MPI(ret, bk_abort_mempool_trim);
        if(NULL == m->head)
            return OMPI_SUCCESS;
    }

    b = m->head->next;
    b_prev = m->head;
    while (NULL != b) {
        if (b->num_passes > pass_cutoff) {
            b_tmp = b->next;
            ret = bkpap_mempool_destroy_buf(b, m);
            b = b_tmp;
            b_prev->next = b;
            BKPAP_CHK_MPI(ret, bk_abort_mempool_trim);
            continue;
        }
        b_prev = b;
        b = b->next;
    }
    return OMPI_SUCCESS;
bk_abort_mempool_trim:
    BKPAP_ERROR("");
    return ret;
}
