#include "coll_bkpap.h"
#include "coll_bkpap_util.inl"
#include "coll_bkpap_ucp.inl"

#include "ompi/mca/pml/pml.h"

int coll_bkpap_papaware_chain_allreduce(const void* sbuf, void* rbuf,
	int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op,
	struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
	mca_coll_bkpap_module_t* bkpap_module) {

	int ret = OMPI_SUCCESS;
	int inter_size = ompi_comm_size(inter_comm), inter_rank = ompi_comm_rank(inter_comm);
	int intra_rank = ompi_comm_rank(intra_comm);
	int is_inter = (0 == intra_rank);

	ptrdiff_t extent;
	ompi_datatype_type_extent(dtype, &extent);

	if (is_inter)BKPAP_PROFILE("bkpap_chain_start", inter_rank);

	ret = bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_abort_chain_allreduce);
	if (is_inter)BKPAP_PROFILE("bkpap_chain_intra_reduce", inter_rank);

	ret = intra_comm->c_coll->coll_barrier(intra_comm, intra_comm->c_coll->coll_barrier_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce barrier failed", bkpap_abort_chain_allreduce);

	if (is_inter) {
		void* tmp_recv = NULL;
		mca_coll_bkpap_get_coll_recvbuf(&tmp_recv, bkpap_module);

		// Syncronize and get arrival
		uint64_t tag, tag_mask;
		int64_t arrival_pos = -1;
		ret = mca_coll_bkpap_arrive_ss(inter_rank, 0, 0, bkpap_module->remote_syncstructure, bkpap_module, inter_comm, &arrival_pos);
		BKPAP_CHK_MPI_MSG_LBL(ret, "arrive_ss failed", bkpap_abort_chain_allreduce);
		int arrival = arrival_pos + 1;
		BKPAP_PROFILE("bkpap_chain_get_arrival", inter_rank);
		BKPAP_OUTPUT("CHAIN_START rank: %d, arrival:%d", inter_rank, arrival);

		ompi_request_t* ibarrer_req = (void*)OMPI_REQUEST_NULL;
		inter_comm->c_coll->coll_ibarrier(inter_comm, &ibarrer_req, inter_comm->c_coll->coll_ibarrier_module);

		if (0 == arrival) { // first
			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.send_to_late(rbuf, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_send_parent", inter_rank);
		}
		else if ((inter_size - 2) == arrival) { //second-last
			int peer_rank = -1;
			while (-1 == peer_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.recv_from_early(tmp_recv, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_local_reduce", inter_rank);

			peer_rank = -1;
			ret = mca_coll_bkpap_get_last_arrival(inter_comm, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
			BKPAP_CHK_MPI_MSG_LBL(ret, "get_last_arrival failed", bkpap_abort_chain_allreduce);
			BKPAP_OUTPUT("rank %d, arrival: %d, last_rank: %d", inter_rank, arrival, peer_rank);

			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.send_to_early(rbuf, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_send_parent", inter_rank);
		}
		else if ((inter_size - 1) == arrival) { //last
			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.recv_from_late(tmp_recv, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_local_reduce", inter_rank);
			ret = bkpap_module->dplane_ftbl.reset_late_recv_buf(bkpap_module);
		}
		else {
			int peer_rank = -1;
			while (-1 == peer_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.recv_from_early(tmp_recv, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_local_reduce", inter_rank);

			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, tag, tag_mask);
			ret = bkpap_module->dplane_ftbl.send_to_late(rbuf, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_send_parent", inter_rank);
		}

		bk_ompi_request_wait_all(&ibarrer_req, 1);

		int bcast_root = -1;
		while (-1 == bcast_root) {
			ret = mca_coll_bkpap_get_rank_of_arrival(inter_size - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &bcast_root);
			BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival bcast_root failed", bkpap_abort_chain_allreduce);
		}

		// ret = inter_comm->c_coll->coll_barrier(inter_comm, inter_comm->c_coll->coll_barrier_module);
		// BKPAP_CHK_MPI_MSG_LBL(ret, "coll barrier failed", bkpap_abort_chain_allreduce);

		ret = bk_inter_bcast(rbuf, count, dtype, bcast_root, inter_comm, bkpap_module,
			mca_coll_bkpap_component.pipeline_segment_size);
		BKPAP_CHK_MPI_MSG_LBL(ret, "bk_inter_bcast failed", bkpap_abort_chain_allreduce);
		BKPAP_PROFILE("bkpap_chain_inter_bcast", inter_rank);

		if (inter_rank == 0) {
			ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_abort_chain_allreduce);
		}

		if (BKPAP_DPLANE_TAG == bkpap_module->dplane_t
			&& !bkpap_module->dplane.tag.prepost_req_set) {
			ret = coll_bkpap_tag_prepost_recv(inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "prepost failed", bkpap_abort_chain_allreduce);
		}

	}

	ret = bk_intra_bcast(rbuf, count, dtype, 0, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "bk_intra_bcast failed", bkpap_abort_chain_allreduce);

	BKPAP_PROFILE("bkpap_chain_intra_bcast", inter_rank);

	return OMPI_SUCCESS;

bkpap_abort_chain_allreduce:
	return ret;
}


int coll_bkpap_papaware_chain_v2_allreduce(const void* sbuf, void* rbuf,
	int count, struct ompi_datatype_t* dtype, struct ompi_op_t* op,
	struct ompi_communicator_t* intra_comm, struct ompi_communicator_t* inter_comm,
	mca_coll_bkpap_module_t* bkpap_module) {

	int ret = OMPI_SUCCESS;
	int inter_size = ompi_comm_size(inter_comm), inter_rank = ompi_comm_rank(inter_comm);
	int intra_rank = ompi_comm_rank(intra_comm);
	int is_inter = (0 == intra_rank);
	MPI_Request send_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
	MPI_Request recv_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
	MPI_Request tmp_reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };

	size_t segsize = mca_coll_bkpap_component.pipeline_segment_size, seg_count = count, typelen;
	ompi_datatype_type_size(dtype, &typelen);
	COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelen, seg_count);

	ptrdiff_t extent;
	ompi_datatype_type_extent(dtype, &extent);
	int num_segments = (count + seg_count - 1) / seg_count;
	size_t realsegsize = (ptrdiff_t)seg_count * extent;
	BKPAP_OUTPUT("count: %d, dsize: %ld, seg_size: %ld num_segs: %d, realsegsize: %ld", count, extent, segsize, num_segments, realsegsize);

	ret = bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_abort_chain_allreduce);

	ret = intra_comm->c_coll->coll_barrier(intra_comm, intra_comm->c_coll->coll_barrier_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce barrier failed", bkpap_abort_chain_allreduce);

	if (is_inter) {
		void* tmp_recv = NULL;
		mca_coll_bkpap_get_coll_recvbuf(&tmp_recv, bkpap_module);

		// Syncronize and get arrival
		uint64_t p_tag, c_tag, c_tag_mask, p_tag_mask;
		int64_t arrival_pos = -1;
		ret = mca_coll_bkpap_arrive_ss(inter_rank, 0, 0, bkpap_module->remote_syncstructure, bkpap_module, inter_comm, &arrival_pos);
		BKPAP_CHK_MPI_MSG_LBL(ret, "arrive_ss failed", bkpap_abort_chain_allreduce);
		int arrival = arrival_pos + 1;
		BKPAP_OUTPUT("CHAIN_START rank: %d, arrival:%d", inter_rank, arrival);

		// DELETEME
		// ompi_request_t* ibarrer_req = (void*)OMPI_REQUEST_NULL;
		// inter_comm->c_coll->coll_ibarrier(inter_comm, &ibarrer_req, inter_comm->c_coll->coll_ibarrier_module);

		if (0 == arrival) { // first
			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, c_tag, c_tag_mask);

			ret = bkpap_module->dplane_ftbl.send_to_late(rbuf, count, dtype, c_tag, c_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_chain_allreduce);

			int child_rank = -1;
			while (-1 == child_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival + 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &child_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg 0 failed", bkpap_abort_chain_allreduce);
			for (int segidx = 0; segidx < num_segments; segidx++) {
				void* rb_tmp = ((int8_t*)rbuf) + (realsegsize * segidx);
				int reqidx = segidx % 2;

				bk_ompi_request_wait_all(&recv_reqs[reqidx], 1);
				ret = MCA_PML_CALL(irecv(rb_tmp, seg_count, dtype, child_rank, c_tag, inter_comm, &recv_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg n failed", bkpap_abort_chain_allreduce);
			}
			bk_ompi_request_wait_all(recv_reqs, 2);
		}
		else if ((inter_size - 2) == arrival) { //second-last
			int parent_rank = -1;
			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, c_tag, c_tag_mask);
			BK_BINOMIAL_MAKE_TAG(arrival, 0, p_tag, p_tag_mask);
			while (-1 == parent_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &parent_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			// recv data from parent
			ret = bkpap_module->dplane_ftbl.recv_from_early(tmp_recv, count, dtype, parent_rank, p_tag, p_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);

			// send data to child
			int child_rank = -1;
			ret = mca_coll_bkpap_get_last_arrival(inter_comm, bkpap_module->remote_syncstructure, bkpap_module, &child_rank);
			BKPAP_CHK_MPI_MSG_LBL(ret, "get_last_arrival failed", bkpap_abort_chain_allreduce);
			BKPAP_OUTPUT("rank %d, arrival: %d, last_rank: %d", inter_rank, arrival, child_rank);

			ret = bkpap_module->dplane_ftbl.send_to_early(rbuf, count, dtype, child_rank, c_tag, c_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_send_parent", inter_rank);

			// pipeline bcast
			ret = MCA_PML_CALL(irecv(rbuf, seg_count, dtype, child_rank, c_tag, inter_comm, &recv_reqs[0]));
			BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv failed", bkpap_abort_chain_allreduce);
			for (int segidx = 1; segidx < num_segments; segidx++) {
				void* rcv_buf = ((int8_t*)rbuf) + (realsegsize * segidx);
				void* snd_buf = ((int8_t*)rbuf) + (realsegsize * (segidx - 1));
				int reqidx = segidx % 2;

				ret = MCA_PML_CALL(irecv(rcv_buf, seg_count, dtype, child_rank, c_tag, inter_comm, &recv_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg n failed", bkpap_abort_chain_allreduce);

				tmp_reqs[0] = recv_reqs[reqidx ^ 1]; tmp_reqs[1] = send_reqs[reqidx];
				bk_ompi_request_wait_all(tmp_reqs, 2);
				ret = MCA_PML_CALL(isend(snd_buf, seg_count, dtype, parent_rank, p_tag, MCA_PML_BASE_SEND_STANDARD, inter_comm, &send_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg n failed", bkpap_abort_chain_allreduce);
			}

			bk_ompi_request_wait_all(&recv_reqs[(num_segments % 2) ^ 1], 1);
			ret = MCA_PML_CALL(isend(((int8_t*)rbuf) + (realsegsize * (num_segments - 1)),
				seg_count, dtype, parent_rank, p_tag, MCA_PML_BASE_SEND_STANDARD, inter_comm, &send_reqs[num_segments % 2]));
			bk_ompi_request_wait_all(send_reqs, 2);
		}
		else if ((inter_size - 1) == arrival) { //last
			int parent_rank = -1;
			BK_BINOMIAL_MAKE_TAG(arrival, 0, p_tag, p_tag_mask);
			while (-1 == parent_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &parent_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			// recv data from parent
			ret = bkpap_module->dplane_ftbl.recv_from_late(tmp_recv, count, dtype, p_tag, p_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			for (int segidx = 0; segidx < num_segments; segidx++) {
				ptrdiff_t offset = realsegsize * segidx;
				void* src_buf = (int8_t*)tmp_recv + offset;
				void* dst_buf = (int8_t*)rbuf + offset;
				int reqidx = segidx % 2;

				ret = mca_coll_bkpap_reduce_local(op, src_buf, dst_buf, seg_count, dtype, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);

				bk_ompi_request_wait_all(send_reqs + reqidx, 1);
				ret = MCA_PML_CALL(isend(dst_buf, seg_count, dtype, parent_rank, p_tag, MCA_PML_BASE_SEND_STANDARD, inter_comm, &send_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast isend failed", bkpap_abort_chain_allreduce);
			}
			bk_ompi_request_wait_all(send_reqs, 2);
		}
		else {
			BK_BINOMIAL_MAKE_TAG(arrival, 0, p_tag, p_tag_mask);
			BK_BINOMIAL_MAKE_TAG(arrival + 1, 0, c_tag, c_tag_mask);
			int parent_rank = -1, child_rank = -1;
			while (-1 == parent_rank) {
				ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &parent_rank);
				BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			}

			// recv data from parent
			ret = bkpap_module->dplane_ftbl.recv_from_early(tmp_recv, count, dtype, parent_rank, p_tag, p_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_recv_child", inter_rank);

			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "local_reduce failed", bkpap_abort_chain_allreduce);

			// send data to child
			ret = bkpap_module->dplane_ftbl.send_to_late(rbuf, count, dtype, c_tag, c_tag_mask, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_chain_allreduce);
			BKPAP_PROFILE("bkpap_chain_send_parent", inter_rank);

			// pipeline bcast
			ret = MCA_PML_CALL(irecv(rbuf, seg_count, dtype, child_rank, c_tag, inter_comm, &recv_reqs[0]));
			BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv failed", bkpap_abort_chain_allreduce);
			for (int segidx = 1; segidx < num_segments; segidx++) {
				void* rcv_buf = ((int8_t*)rbuf) + (realsegsize * segidx);
				void* snd_buf = ((int8_t*)rbuf) + (realsegsize * (segidx - 1));
				int reqidx = segidx % 2;

				ret = MCA_PML_CALL(irecv(rcv_buf, seg_count, dtype, child_rank, c_tag, inter_comm, &recv_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg n failed", bkpap_abort_chain_allreduce);

				tmp_reqs[0] = recv_reqs[reqidx ^ 1]; tmp_reqs[1] = send_reqs[reqidx];
				bk_ompi_request_wait_all(tmp_reqs, 2);
				ret = MCA_PML_CALL(isend(snd_buf, seg_count, dtype, parent_rank, p_tag, MCA_PML_BASE_SEND_STANDARD, inter_comm, &send_reqs[reqidx]));
				BKPAP_CHK_MPI_MSG_LBL(ret, "bcast data recv seg n failed", bkpap_abort_chain_allreduce);
			}

			bk_ompi_request_wait_all(&recv_reqs[(num_segments % 2) ^ 1], 1);
			ret = MCA_PML_CALL(isend(((int8_t*)rbuf) + (realsegsize * (num_segments - 1)),
				seg_count, dtype, parent_rank, p_tag, MCA_PML_BASE_SEND_STANDARD, inter_comm, &send_reqs[num_segments % 2]));
			bk_ompi_request_wait_all(send_reqs, 2);
		}

		// bk_ompi_request_wait_all(&ibarrer_req, 1);
		inter_comm->c_coll->coll_barrier(inter_comm, inter_comm->c_coll->coll_barrier_module);

		if (inter_rank == 0) {
			ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_abort_chain_allreduce);
		}

		if (BKPAP_DPLANE_TAG == bkpap_module->dplane_t
			&& !bkpap_module->dplane.tag.prepost_req_set) {
			ret = coll_bkpap_tag_prepost_recv(inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "prepost failed", bkpap_abort_chain_allreduce);
		}
	}

	ret = bk_intra_bcast(rbuf, count, dtype, 0, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "bk_intra_bcast failed", bkpap_abort_chain_allreduce);


	return OMPI_SUCCESS;

bkpap_abort_chain_allreduce:
	return ret;
}