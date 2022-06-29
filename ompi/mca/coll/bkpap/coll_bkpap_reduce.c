#include "coll_bkpap.h"
#include "coll_bkpap_ucp.inl"
#include "coll_bkpap_util.inl"
#include "bkpap_kernel.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/pml/pml.h"

// NEED (pbuft_t * msgsize) * 2
// this is ripped from coll_base_reduce.c, and bk_gpu_op_reduce() as replaced ompi_reduce_local() 
// count_by_segment will always == count, max_outstanding_reqs will always == 0
// num_segments will always == 1
int mca_coll_bkpap_reduce_generic(const void* sendbuf, void* recvbuf, int original_count,
	ompi_datatype_t* datatype, ompi_op_t* op,
	int root, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module,
	ompi_coll_tree_t* tree, int count_by_segment,
	int max_outstanding_reqs) {
	char* inbuf[2] = { NULL, NULL }, * inbuf_free[2] = { NULL, NULL };
	char* accumbuf = NULL, * accumbuf_free = NULL;
	char* local_op_buffer = NULL, * sendtmpbuf = NULL;
	ptrdiff_t extent, size, gap = 0, segment_increment;
	ompi_request_t** sreq = NULL, * reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
	int num_segments, line, ret, segindex, i, rank;
	int recvcount, prevcount, inbi;
	mca_coll_base_module_t* module = &(bkpap_module->super);
	mca_coll_bkpap_postbuf_memory_t bk_rbuf_memtype = mca_coll_bkpap_component.bk_postbuf_memory_type;
	

	/**
	 * Determine number of segments and number of elements
	 * sent per operation
	 */
	ompi_datatype_type_extent(datatype, &extent);
	num_segments = (int)(((size_t)original_count + (size_t)count_by_segment - (size_t)1) / (size_t)count_by_segment);
	segment_increment = (ptrdiff_t)count_by_segment * extent;

	sendtmpbuf = (char*)sendbuf;
	if (sendbuf == MPI_IN_PLACE) {
		sendtmpbuf = (char*)recvbuf;
	}

	BKPAP_OUTPUT("coll:base:reduce_generic count %d, msg size %ld, segsize %ld, max_requests %d",
		original_count, (unsigned long)((ptrdiff_t)num_segments * (ptrdiff_t)segment_increment),
		(unsigned long)segment_increment, max_outstanding_reqs);

	rank = ompi_comm_rank(comm);

	/* non-leaf nodes - wait for children to send me data & forward up
	   (if needed) */
	if (tree->tree_nextsize > 0) {
		ptrdiff_t real_segment_size;

		/* handle non existant recv buffer (i.e. its NULL) and
		   protect the recv buffer on non-root nodes */
		accumbuf = (char*)recvbuf;
		if ((NULL == accumbuf) || (root != rank)) {
			/* Allocate temporary accumulator buffer. */
			size = opal_datatype_span(&datatype->super, original_count, &gap);
			ret = bkpap_mempool_alloc((void**)&accumbuf_free, size, bk_rbuf_memtype, bkpap_module);
			if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
				line = __LINE__; ret = -1; goto error_hndl;
			}
			accumbuf = accumbuf_free - gap;
		}

		/* If this is a non-commutative operation we must copy
		   sendbuf to the accumbuf, in order to simplfy the loops */

		if (!ompi_op_is_commute(op) && MPI_IN_PLACE != sendbuf) {
			ompi_datatype_copy_content_same_ddt(datatype, original_count,
				(char*)accumbuf,
				(char*)sendtmpbuf);
		}
		/* Allocate two buffers for incoming segments */
		real_segment_size = opal_datatype_span(&datatype->super, count_by_segment, &gap);
		ret = bkpap_mempool_alloc((void**)&inbuf_free[0], real_segment_size, bk_rbuf_memtype, bkpap_module);
		if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
			line = __LINE__; ret = -1; goto error_hndl;
		}
		inbuf[0] = inbuf_free[0] - gap;
		/* if there is chance to overlap communication -
		   allocate second buffer */
		if ((num_segments > 1) || (tree->tree_nextsize > 1)) {
			ret = bkpap_mempool_alloc((void**)&inbuf_free[1], real_segment_size, bk_rbuf_memtype, bkpap_module);
			if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
				line = __LINE__; ret = -1; goto error_hndl;
			}
			inbuf[1] = inbuf_free[1] - gap;
		}

		/* reset input buffer index and receive count */
		inbi = 0;
		recvcount = 0;
		/* for each segment */
		for (segindex = 0; segindex <= num_segments; segindex++) {
			prevcount = recvcount;
			/* recvcount - number of elements in current segment */
			recvcount = count_by_segment;
			if (segindex == (num_segments - 1))
				recvcount = original_count - (ptrdiff_t)count_by_segment * (ptrdiff_t)segindex;

			/* for each child */
			for (i = 0; i < tree->tree_nextsize; i++) {
				/**
				 * We try to overlap communication:
				 * either with next segment or with the next child
				 */
				 /* post irecv for current segindex on current child */
				if (segindex < num_segments) {
					void* local_recvbuf = inbuf[inbi];
					if (0 == i) {
						/* for the first step (1st child per segment) and
						 * commutative operations we might be able to irecv
						 * directly into the accumulate buffer so that we can
						 * reduce(op) this with our sendbuf in one step as
						 * ompi_op_reduce only has two buffer pointers,
						 * this avoids an extra memory copy.
						 *
						 * BUT if the operation is non-commutative or
						 * we are root and are USING MPI_IN_PLACE this is wrong!
						 */
						if ((ompi_op_is_commute(op)) &&
							!((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root))) {
							local_recvbuf = accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
						}
					}

					ret = MCA_PML_CALL(irecv(local_recvbuf, recvcount, datatype,
						tree->tree_next[i],
						MCA_COLL_BASE_TAG_REDUCE, comm,
						&reqs[inbi]));
					if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
				}
				/* wait for previous req to complete, if any.
				   if there are no requests reqs[inbi ^1] will be
				   MPI_REQUEST_NULL. */
				   /* wait on data from last child for previous segment */
				ret = ompi_request_wait(&reqs[inbi ^ 1],
					MPI_STATUSES_IGNORE);
				if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
				local_op_buffer = inbuf[inbi ^ 1];
				if (i > 0) {
					/* our first operation is to combine our own [sendbuf] data
					 * with the data we recvd from down stream (but only
					 * the operation is commutative and if we are not root and
					 * not using MPI_IN_PLACE)
					 */
					if (1 == i) {
						if ((ompi_op_is_commute(op)) &&
							!((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root))) {
							local_op_buffer = sendtmpbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
						}
					}
					/* apply operation */
					bk_gpu_op_reduce(op, local_op_buffer,
						accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
						recvcount, datatype);
				}
				else if (segindex > 0) {
					void* accumulator = accumbuf + (ptrdiff_t)(segindex - 1) * (ptrdiff_t)segment_increment;
					if (tree->tree_nextsize <= 1) {
						if ((ompi_op_is_commute(op)) &&
							!((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root))) {
							local_op_buffer = sendtmpbuf + (ptrdiff_t)(segindex - 1) * (ptrdiff_t)segment_increment;
						}
					}
					bk_gpu_op_reduce(op, local_op_buffer, accumulator, prevcount,
						datatype);

					/* all reduced on available data this step (i) complete,
					 * pass to the next process unless you are the root.
					 */
					if (rank != tree->tree_root) {
						/* send combined/accumulated data to parent */
						ret = MCA_PML_CALL(send(accumulator, prevcount,
							datatype, tree->tree_prev,
							MCA_COLL_BASE_TAG_REDUCE,
							MCA_PML_BASE_SEND_STANDARD,
							comm));
						if (ret != MPI_SUCCESS) {
							line = __LINE__; goto error_hndl;
						}
					}

					/* we stop when segindex = number of segments
					   (i.e. we do num_segment+1 steps for pipelining */
					if (segindex == num_segments) break;
				}

				/* update input buffer index */
				inbi = inbi ^ 1;
			} /* end of for each child */
		} /* end of for each segment */

		/* clean up */
		if (inbuf_free[0] != NULL) bk_mempool_free(inbuf_free[0], bk_rbuf_memtype, bkpap_module);
		if (inbuf_free[1] != NULL) bk_mempool_free(inbuf_free[1], bk_rbuf_memtype, bkpap_module);
		if (accumbuf_free != NULL) bk_mempool_free(accumbuf_free, bk_rbuf_memtype, bkpap_module);
	}

	/* leaf nodes
	   Depending on the value of max_outstanding_reqs and
	   the number of segments we have two options:
	   - send all segments using blocking send to the parent, or
	   - avoid overflooding the parent nodes by limiting the number of
	   outstanding requests to max_oustanding_reqs.
	   TODO/POSSIBLE IMPROVEMENT: If there is a way to determine the eager size
	   for the current communication, synchronization should be used only
	   when the message/segment size is smaller than the eager size.
	*/
	else {

		/* If the number of segments is less than a maximum number of oustanding
		   requests or there is no limit on the maximum number of outstanding
		   requests, we send data to the parent using blocking send */
		if ((0 == max_outstanding_reqs) ||
			(num_segments <= max_outstanding_reqs)) {

			segindex = 0;
			while (original_count > 0) {
				if (original_count < count_by_segment) {
					count_by_segment = original_count;
				}
				ret = MCA_PML_CALL(send((char*)sendbuf +
					(ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
					count_by_segment, datatype,
					tree->tree_prev,
					MCA_COLL_BASE_TAG_REDUCE,
					MCA_PML_BASE_SEND_STANDARD,
					comm));
				if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
				segindex++;
				original_count -= count_by_segment;
			}
		}

		/* Otherwise, introduce flow control:
		   - post max_outstanding_reqs non-blocking synchronous send,
		   - for remaining segments
		   - wait for a ssend to complete, and post the next one.
		   - wait for all outstanding sends to complete.
		*/
		else {

			int creq = 0;

			sreq = ompi_coll_base_comm_get_reqs(module->base_data, max_outstanding_reqs);
			if (NULL == sreq) { line = __LINE__; ret = -1; goto error_hndl; }

			/* post first group of requests */
			for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
				ret = MCA_PML_CALL(isend((char*)sendbuf +
					(ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
					count_by_segment, datatype,
					tree->tree_prev,
					MCA_COLL_BASE_TAG_REDUCE,
					MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
					&sreq[segindex]));
				if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
				original_count -= count_by_segment;
			}

			creq = 0;
			while (original_count > 0) {
				/* wait on a posted request to complete */
				ret = ompi_request_wait(&sreq[creq], MPI_STATUS_IGNORE);
				if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }

				if (original_count < count_by_segment) {
					count_by_segment = original_count;
				}
				ret = MCA_PML_CALL(isend((char*)sendbuf +
					(ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
					count_by_segment, datatype,
					tree->tree_prev,
					MCA_COLL_BASE_TAG_REDUCE,
					MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
					&sreq[creq]));
				if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
				creq = (creq + 1) % max_outstanding_reqs;
				segindex++;
				original_count -= count_by_segment;
			}

			/* Wait on the remaining request to complete */
			ret = ompi_request_wait_all(max_outstanding_reqs, sreq,
				MPI_STATUSES_IGNORE);
			if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		}
	}
	return OMPI_SUCCESS;

error_hndl:  /* error handler */
	/* find a real error code */
	if (MPI_ERR_IN_STATUS == ret) {
		for (i = 0; i < 2; i++) {
			if (MPI_REQUEST_NULL == reqs[i]) continue;
			if (MPI_ERR_PENDING == reqs[i]->req_status.MPI_ERROR) continue;
			if (reqs[i]->req_status.MPI_ERROR != MPI_SUCCESS) {
				ret = reqs[i]->req_status.MPI_ERROR;
				break;
			}
		}
	}
	ompi_coll_base_free_reqs(reqs, 2);
	if (NULL != sreq) {
		if (MPI_ERR_IN_STATUS == ret) {
			for (i = 0; i < max_outstanding_reqs; i++) {
				if (MPI_REQUEST_NULL == sreq[i]) continue;
				if (MPI_ERR_PENDING == sreq[i]->req_status.MPI_ERROR) continue;
				if (sreq[i]->req_status.MPI_ERROR != MPI_SUCCESS) {
					ret = sreq[i]->req_status.MPI_ERROR;
					break;
				}
			}
		}
		ompi_coll_base_free_reqs(sreq, max_outstanding_reqs);
	}
	if (inbuf_free[0] != NULL) bk_mempool_free(inbuf_free[0], bk_rbuf_memtype, bkpap_module);
	if (inbuf_free[1] != NULL) bk_mempool_free(inbuf_free[1], bk_rbuf_memtype, bkpap_module);
	if (accumbuf_free != NULL) bk_mempool_free(accumbuf, bk_rbuf_memtype, bkpap_module);
	BKPAP_OUTPUT(
		"ERROR_HNDL: node %d file %s line %d error %d\n",
		rank, __FILE__, line, ret);
	(void)line;  // silence compiler warning
	return ret;
}

int mca_coll_bkpap_reduce_intra_inplace_binomial(const void* sendbuf, void* recvbuf,
	int count, ompi_datatype_t* datatype,
	ompi_op_t* op, int root,
	ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module) {

	uint32_t segsize = 0;
	int max_outstanding_reqs = 0;
	int segcount = count;
	size_t typelng;
	mca_coll_base_module_t* base_module = &(bkpap_module->super);
	mca_coll_base_comm_t* data = base_module->base_data;

	if (OPAL_UNLIKELY(1 == ompi_comm_size(comm)))return OMPI_SUCCESS;

	BKPAP_OUTPUT("coll:bkpap:reduce_intra_binomial rank %d ss %5d [%p]",
		ompi_comm_rank(comm), segsize, recvbuf);

	COLL_BASE_UPDATE_IN_ORDER_BMTREE(comm, base_module, root);

	/**
	 * Determine number of segments and number of elements
	 * sent per operation
	 */
	ompi_datatype_type_size(datatype, &typelng);
	COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelng, segcount);

	return mca_coll_bkpap_reduce_generic(sendbuf, recvbuf, count, datatype,
		op, root, comm, bkpap_module,
		data->cached_in_order_bmtree,
		segcount, max_outstanding_reqs);
}

int bk_intra_reduce(void* rbuf, int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op, struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {
    int rank = ompi_comm_rank(comm);

    void* reduce_sbuf = (0 == rank) ? MPI_IN_PLACE : rbuf;
    void* reduce_rbuf = (0 == rank) ? rbuf : NULL;

    switch (mca_coll_bkpap_component.bk_postbuf_memory_type) {
    case BKPAP_POSTBUF_MEMORY_TYPE_HOST:
        return comm->c_coll->coll_reduce(
            reduce_sbuf, reduce_rbuf, count, dtype, op, 0,
            comm, comm->c_coll->coll_reduce_module);
        break;
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA:
    case BKPAP_POSTBUF_MEMORY_TYPE_CUDA_MANAGED:
        return mca_coll_bkpap_reduce_intra_inplace_binomial(reduce_sbuf, reduce_rbuf, count, dtype, op, 0, comm, bkpap_module);
        break;
    default:
        BKPAP_ERROR("Bad memory type, intra-node reduce failed");
        return OMPI_ERROR;
        break;
    }
}