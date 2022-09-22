#include "coll_bkpap.h"
#include "ompi/mca/pml/pml.h"
#include "coll_bkpap_util.inl"

int coll_bkpap_bcast_intra_generic_gpu(void* buffer, int original_count,
	struct ompi_datatype_t* datatype, int root,
	struct ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module,
	uint32_t count_by_segment, ompi_coll_tree_t* tree) {

	mca_coll_base_module_t* module = &bkpap_module->super;
	int err = 0, line, i, rank, segindex, req_index;
	int num_segments; /* Number of segments */
	int sendcount;    /* number of elements sent in this segment */
	size_t realsegsize, type_size;
	char* tmpbuf;
	ptrdiff_t extent, lb;
	ompi_request_t* recv_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
	ompi_request_t** send_reqs = NULL;


#if OPAL_ENABLE_DEBUG
	int size;
	size = ompi_comm_size(comm);
	assert(size > 1);
#endif
	rank = ompi_comm_rank(comm);

	ompi_datatype_get_extent(datatype, &lb, &extent);
	ompi_datatype_type_size(datatype, &type_size);
	num_segments = (original_count + count_by_segment - 1) / count_by_segment;
	realsegsize = (ptrdiff_t)count_by_segment * extent;

	if ((size_t)(extent * original_count) > mca_coll_bkpap_component.postbuff_size) {
		opal_show_help("help-mpi-coll-bkpap.txt", "custom error", true, 
		"postbuf size to small for bk_bcast_intra_generic_gpu");
		BKPAP_ERROR("postbuf not large enough for bk bcast");
		err = OMPI_ERROR;
		goto error_hndl;
	}

	/* Set the buffer pointers */
	tmpbuf = (char*)buffer;

	if (tree->tree_nextsize != 0) {
		send_reqs = ompi_coll_base_comm_get_reqs(module->base_data, tree->tree_nextsize);
		if (NULL == send_reqs) { err = OMPI_ERR_OUT_OF_RESOURCE; line = __LINE__; goto error_hndl; }
	}

	cudaStream_t* bk_cs = bkpap_module->bk_cs;
	uint8_t* bk_h_buf = bkpap_module->host_pinned_buf;

	/* Root code */
	if (rank == root) {
		/*
		   For each segment:
		   - send segment to all children.
		   The last segment may have less elements than other segments.
		*/
		sendcount = count_by_segment;
		cudaMemcpyAsync(bk_h_buf, tmpbuf, sendcount * extent, cudaMemcpyDeviceToHost, bk_cs[0]);
		for (segindex = 0; segindex < num_segments; segindex++) {

			if (segindex == (num_segments - 1)) {
				sendcount = original_count - segindex * count_by_segment;
			}
			cudaStreamSynchronize(bk_cs[(segindex % 2)]);
			for (i = 0; i < tree->tree_nextsize; i++) {
				err = MCA_PML_CALL(isend(bk_h_buf, sendcount, datatype,
					tree->tree_next[i],
					MCA_COLL_BASE_TAG_BCAST,
					MCA_PML_BASE_SEND_STANDARD, comm,
					&send_reqs[i]));
				if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
			}

			/* update tmp buffer */
			tmpbuf += realsegsize;
			bk_h_buf += realsegsize;
			if (segindex + 1 == (num_segments - 1)) {
				sendcount = original_count - (segindex + 1) * count_by_segment;
			}

			// issue next memcpy
			if(segindex - 1 < num_segments)
				cudaMemcpyAsync(bk_h_buf, tmpbuf, sendcount * extent, cudaMemcpyDeviceToHost, bk_cs[(segindex % 2) ^ 0x1]);

			/* complete the sends before starting the next sends */
			err = ompi_request_wait_all(tree->tree_nextsize, send_reqs,
				MPI_STATUSES_IGNORE);
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }


		}
	}

	/* Intermediate nodes code */
	else if (tree->tree_nextsize > 0) {
		/*
		   Create the pipeline.
		   1) Post the first receive
		   2) For segments 1 .. num_segments
		   - post new receive
		   - wait on the previous receive to complete
		   - send this data to children
		   3) Wait on the last segment
		   4) Compute number of elements in last segment.
		   5) Send the last segment to children
		*/

		req_index = 0;
		err = MCA_PML_CALL(irecv(bk_h_buf, count_by_segment, datatype,
			tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
			comm, &recv_reqs[req_index]));
		if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }

		for (segindex = 1; segindex < num_segments; segindex++) {

			req_index = req_index ^ 0x1;

			/* post new irecv */
			err = MCA_PML_CALL(irecv(bk_h_buf + realsegsize, count_by_segment,
				datatype, tree->tree_prev,
				MCA_COLL_BASE_TAG_BCAST,
				comm, &recv_reqs[req_index]));
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }

			/* wait for and forward the previous segment to children */
			err = ompi_request_wait(&recv_reqs[req_index ^ 0x1],
				MPI_STATUS_IGNORE);
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }

			for (i = 0; i < tree->tree_nextsize; i++) {
				err = MCA_PML_CALL(isend(bk_h_buf, count_by_segment, datatype,
					tree->tree_next[i],
					MCA_COLL_BASE_TAG_BCAST,
					MCA_PML_BASE_SEND_STANDARD, comm,
					&send_reqs[i]));
				if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
			}

			cudaMemcpyAsync(tmpbuf, bk_h_buf, realsegsize, cudaMemcpyHostToDevice, bk_cs[0]);

			/* complete the sends before starting the next iteration */
			err = ompi_request_wait_all(tree->tree_nextsize, send_reqs,
				MPI_STATUSES_IGNORE);
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }

			/* Update the receive buffer */
			tmpbuf += realsegsize;
			bk_h_buf += realsegsize;
		}

		/* Process the last segment */
		err = ompi_request_wait(&recv_reqs[req_index], MPI_STATUS_IGNORE);
		if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		sendcount = original_count - (ptrdiff_t)(num_segments - 1) * count_by_segment;
		for (i = 0; i < tree->tree_nextsize; i++) {
			err = MCA_PML_CALL(isend(bk_h_buf, sendcount, datatype,
				tree->tree_next[i],
				MCA_COLL_BASE_TAG_BCAST,
				MCA_PML_BASE_SEND_STANDARD, comm,
				&send_reqs[i]));
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		}

		cudaMemcpyAsync(tmpbuf, bk_h_buf, sendcount * extent, cudaMemcpyHostToDevice, bk_cs[0]);

		err = ompi_request_wait_all(tree->tree_nextsize, send_reqs,
			MPI_STATUSES_IGNORE);
		if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		cudaStreamSynchronize(bk_cs[0]);
	}

	/* Leaf nodes */
	else {
		/*
		   Receive all segments from parent in a loop:
		   1) post irecv for the first segment
		   2) for segments 1 .. num_segments
		   - post irecv for the next segment
		   - wait on the previous segment to arrive
		   3) wait for the last segment
		*/

		req_index = 0;
		err = MCA_PML_CALL(irecv(bk_h_buf, count_by_segment, datatype,
			tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
			comm, &recv_reqs[req_index]));
		if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }


		for (segindex = 1; segindex < num_segments; segindex++) {
			req_index = req_index ^ 0x1;
			bk_h_buf += realsegsize;
			tmpbuf += realsegsize;
			/* post receive for the next segment */
			err = MCA_PML_CALL(irecv(bk_h_buf, count_by_segment, datatype,
				tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
				comm, &recv_reqs[req_index]));
			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
			/* wait on the previous segment */
			err = ompi_request_wait(&recv_reqs[req_index ^ 0x1],
				MPI_STATUS_IGNORE);

			cudaMemcpyAsync(tmpbuf - realsegsize, bk_h_buf - realsegsize, realsegsize, cudaMemcpyHostToDevice, bk_cs[0]);

			if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		}

		err = ompi_request_wait(&recv_reqs[req_index], MPI_STATUS_IGNORE);
		if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
		size_t f_rsize = original_count - (ptrdiff_t)(num_segments - 1) * count_by_segment;
		cudaMemcpyAsync(tmpbuf, bk_h_buf, f_rsize * extent, cudaMemcpyHostToDevice, bk_cs[0]);
		cudaStreamSynchronize(bk_cs[0]);
	}

	return (MPI_SUCCESS);

error_hndl:
	if (MPI_ERR_IN_STATUS == err) {
		for (req_index = 0; req_index < 2; req_index++) {
			if (MPI_REQUEST_NULL == recv_reqs[req_index]) continue;
			if (MPI_ERR_PENDING == recv_reqs[req_index]->req_status.MPI_ERROR) continue;
			if (recv_reqs[req_index]->req_status.MPI_ERROR != MPI_SUCCESS) {
				err = recv_reqs[req_index]->req_status.MPI_ERROR;
				break;
			}
		}
	}
	ompi_coll_base_free_reqs(recv_reqs, 2);
	if (NULL != send_reqs) {
		if (MPI_ERR_IN_STATUS == err) {
			for (req_index = 0; req_index < tree->tree_nextsize; req_index++) {
				if (MPI_REQUEST_NULL == send_reqs[req_index]) continue;
				if (MPI_ERR_PENDING == send_reqs[req_index]->req_status.MPI_ERROR) continue;
				if (send_reqs[req_index]->req_status.MPI_ERROR != MPI_SUCCESS) {
					err = send_reqs[req_index]->req_status.MPI_ERROR;
					break;
				}
			}
		}
		ompi_coll_base_free_reqs(send_reqs, tree->tree_nextsize);
	}
	BKPAP_OUTPUT("%s:%4d\tError occurred %d, rank %2d",
		__FILE__, line, err, rank);
	(void)line;  // silence compiler warnings

	return err;
}


int bk_inter_bcast(void* buf, int count, struct ompi_datatype_t* dtype,
	int root, ompi_communicator_t* comm,
	mca_coll_bkpap_module_t* bkpap_module, uint32_t seg_size) {
	int ret = OMPI_SUCCESS;

	int seg_count = count;
	size_t typelen;
	ompi_datatype_type_size(dtype, &typelen);

	bkpap_dplane_mem_t  memtype = get_bk_memtype(buf);

	mca_coll_base_comm_t* data = bkpap_module->super.base_data;
	COLL_BASE_UPDATE_PIPELINE(comm, &bkpap_module->super, root);
	COLL_BASE_COMPUTED_SEGCOUNT(seg_size, typelen, seg_count);

	switch (memtype) {
	case BKPAP_DPLANE_MEM_TYPE_HOST:
		ret = ompi_coll_base_bcast_intra_generic(buf, count, dtype, root, comm, &bkpap_module->super, seg_count, data->cached_pipeline);
		break;
	case BKPAP_DPLANE_MEM_TYPE_CUDA:
	case BKPAP_DPLANE_MEM_TYPE_CUDA_MANAGED:
		ret = coll_bkpap_bcast_intra_generic_gpu(buf, count, dtype, root, comm, bkpap_module, seg_count, data->cached_pipeline);
		break;
	default:
		opal_show_help("help-mpi-coll-bkpap.txt", "bad selection", true,
			ompi_process_info.nodename,
			"coll_bkpap_postbuf_mem_type", memtype);
		BKPAP_ERROR("Bad memtype in bkpap_inter_bcast");
		ret = OMPI_ERROR;
		break;
	}
	BKPAP_CHK_MPI_MSG_LBL(ret, "selected alg failed in bkpap_inter_gpu", bkpap_abort_inter_gpu);

	return OMPI_SUCCESS;
bkpap_abort_inter_gpu:
	return ret;
}

int bk_intra_bcast(void* buf, int count, struct ompi_datatype_t* dtype, int root, ompi_communicator_t* comm, mca_coll_bkpap_module_t* bkpap_module) {
	if (ompi_comm_size(comm) <= 1)
		return OMPI_SUCCESS;

	return ompi_coll_base_bcast_intra_knomial(buf, count, dtype, root, comm, &bkpap_module->super, 0, 4);

}