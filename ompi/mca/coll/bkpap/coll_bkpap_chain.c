#include "coll_bkpap.h"
#include "coll_bkpap_util.inl"
#include "coll_bkpap_ucp.inl"

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
			// int peer_rank = -1;
			// while (-1 == peer_rank) {
			// 	ret = mca_coll_bkpap_get_rank_of_arrival(arrival - 1, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
			// 	BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival failed", bkpap_abort_chain_allreduce);
			// }

			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			// ret = bkpap_module->dplane_ftbl.recv_from_early(tmp_recv, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
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

		ret = inter_comm->c_coll->coll_barrier(inter_comm, inter_comm->c_coll->coll_barrier_module);
		BKPAP_CHK_MPI_MSG_LBL(ret, "coll barrier failed", bkpap_abort_chain_allreduce);


		ret = bk_inter_bcast(rbuf, count, dtype, bcast_root, inter_comm, bkpap_module,
			mca_coll_bkpap_component.pipeline_segment_size);
		BKPAP_CHK_MPI_MSG_LBL(ret, "bk_inter_bcast failed", bkpap_abort_chain_allreduce);
		BKPAP_PROFILE("bkpap_chain_inter_bcast", inter_rank);

		if (inter_rank == 0) {
			ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
			BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_abort_chain_allreduce);
		}

	}

	ret = bk_intra_bcast(rbuf, count, dtype, 0, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "bk_intra_bcast failed", bkpap_abort_chain_allreduce);

	BKPAP_PROFILE("bkpap_chain_intra_bcast", inter_rank);

	return OMPI_SUCCESS;

bkpap_abort_chain_allreduce:
	return ret;
}