#include "coll_bkpap.h"
#include "coll_bkpap_util.inl"
#include "coll_bkpap_ucp.inl"
#include "opal/util/bit_ops.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

int coll_bkpap_papaware_binomial_allreduce(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype,
	struct ompi_op_t* op, struct ompi_communicator_t* intra_comm,
	struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module) {

	int ret = OMPI_SUCCESS;
	int inter_size = ompi_comm_size(inter_comm), inter_rank = ompi_comm_rank(inter_comm);
	int intra_rank = ompi_comm_rank(intra_comm);
	int is_inter = (0 == intra_rank);
	int vrank, mask = 1, p = -1, * c = NULL, tree_depth, nc = 0;

	if (is_inter)BKPAP_PROFILE("bkpap_bin_start", inter_rank);

	// intra_reduce
	ret = bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_abort_binomial_allreduce);
	if (is_inter)BKPAP_PROFILE("bkpap_bin_intra_reduce", inter_rank);

	if (is_inter) {
		uint64_t tag, tag_mask;

		int64_t arrival_pos = -1;
		ret = mca_coll_bkpap_arrive_ss(inter_rank, 0, 0, bkpap_module->remote_syncstructure, bkpap_module, inter_comm, &arrival_pos);
		BKPAP_CHK_MPI_MSG_LBL(ret, "arrive_ss failed", bkpap_abort_binomial_allreduce);
		int arrival = arrival_pos + 1;
		BKPAP_PROFILE("bkpap_bin_get_arrival", inter_rank);

		ompi_request_t* ibarrer_req = (void*)OMPI_REQUEST_NULL;
		inter_comm->c_coll->coll_ibarrier(inter_comm, &ibarrer_req, inter_comm->c_coll->coll_ibarrier_module);

		vrank = (arrival + inter_size + 2) % inter_size;
		tree_depth = opal_hibit(inter_size, 31);
		size_t c_arr_size = sizeof(*c) * tree_depth;
		c = malloc(c_arr_size);
		memset(c, -1, c_arr_size);
		while (mask < inter_size) {
			if (vrank % (2 * mask)) {
				p = vrank / (mask * 2) * (mask * 2);
				p = (p + inter_size - 2) % inter_size;
				break;
			}
			mask *= 2;
		}
		mask /= 2;
		while (mask > 0) {
			for (int i = 1; i < 2; i++) {
				int nxt_c = vrank + mask * i;
				if (nxt_c < inter_size) {
					nxt_c = (nxt_c + inter_size - 2) % inter_size;
					c[nc] = nxt_c;
					nc++;
				}
			}
			mask /= 2;
		}
		BKPAP_OUTPUT("CALC TREE rank: %d, arrival: %d, depth: %d, parent: %d, nc: %d, childs:[%d, %d, %d, %d]", inter_rank, arrival, tree_depth, p, nc, c[0], c[1], c[2], c[3]);
		BKPAP_PROFILE("bkpap_bin_calc_tree", inter_rank);

		for (int i = 0; i < nc; i++) {
			int peer = c[i];
			void* tmp_recv = bkpap_module->local_pbuffs.tag.buff_arr; // TODO: replace with mempool
			BK_BINOMIAL_MAKE_TAG(peer, 0, tag, tag_mask);
			if (peer > arrival) {
				BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (late) child: %d", inter_rank, arrival, peer);

				ret = mca_coll_bkpap_dplane_recv_from_late(tmp_recv, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_late failed", bkpap_abort_binomial_allreduce);
			}
			else {
				int peer_rank = -1;
				while (-1 == peer_rank) {
					ret = mca_coll_bkpap_get_rank_of_arrival(peer, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
					BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival peer failed", bkpap_abort_binomial_allreduce);
				}
				BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (early) child: %d(%d)", inter_rank, arrival, peer, peer_rank);

				ret = mca_coll_bkpap_dplane_recv_from_early(tmp_recv, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_binomial_allreduce);
			}
			ret = mca_coll_bkpap_reduce_local(op, tmp_recv, rbuf, count, dtype);
			BKPAP_CHK_MPI_MSG_LBL(ret, "reduce_local failed", bkpap_abort_binomial_allreduce);
		}
		BKPAP_PROFILE("bkpap_bin_recv_child", inter_rank);

		if (-1 != p) {
			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			if (p > arrival) {
				BKPAP_OUTPUT("rank: %d, arrival: %d, send to (late) parent: %d", inter_rank, arrival, p);

				ret = mca_coll_bkpap_dplane_send_to_late(rbuf, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_binomial_allreduce);
			}
			else {
				int peer_rank = -1;
				while (-1 == peer_rank) {
					ret = mca_coll_bkpap_get_rank_of_arrival(p, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
					BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival parent failed", bkpap_abort_binomial_allreduce);
				}
				BKPAP_OUTPUT("rank: %d, arrival: %d, send to (early) parent: %d(%d)", inter_rank, arrival, p, peer_rank);

				ret = mca_coll_bkpap_dplane_send_to_early(rbuf, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_early failed", bkpap_abort_binomial_allreduce);
			}
		}
		BKPAP_PROFILE("bkpap_bin_send_parent", inter_rank);
		bk_ompi_request_wait_all(&ibarrer_req, 1);

		int bcast_root = -1;
		while (-1 == bcast_root) {
			ret = mca_coll_bkpap_get_rank_of_arrival(inter_size - 2, 0, bkpap_module->remote_syncstructure, bkpap_module, &bcast_root);
			BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival bcast_root failed", bkpap_abort_binomial_allreduce);
		}

		ret = bk_inter_bcast(rbuf, count, dtype, bcast_root, inter_comm, bkpap_module);
		BKPAP_CHK_MPI_MSG_LBL(ret, "bk_opt_bc bcast failed", bkpap_abort_binomial_allreduce);

	}

	BKPAP_OUTPUT("rank (%d, %d) calling bcast on rbuf [%p], is_cu: %d, count: %d", ompi_comm_rank(inter_comm), ompi_comm_rank(intra_comm), rbuf, opal_cuda_check_one_buf(rbuf, NULL), count);
	// ret = intra_comm->c_coll->coll_bcast(
	// 	rbuf, count, dtype, 0, intra_comm,
	// 	intra_comm->c_coll->coll_bcast_module);
	// ret = ompi_coll_base_bcast_intra_bintree(rbuf, count, dtype, 0, intra_comm, &bkpap_module->super, 0);
	ret = bk_intra_bcast(rbuf, count, dtype, 0, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra-stage bcast failed", bkpap_abort_binomial_allreduce);
	if (is_inter)BKPAP_PROFILE("bkpap_bin_intra_bcast", inter_rank);


	if (0 == inter_rank && 0 == intra_rank) {
		BKPAP_OUTPUT("RESET ATTEMPT");
		ret = mca_coll_bkpap_reset_remote_ss(bkpap_module->remote_syncstructure, inter_comm, bkpap_module);
		BKPAP_CHK_MPI_MSG_LBL(ret, "reset_remote_ss failed", bkpap_abort_binomial_allreduce);
	}
	if (is_inter)BKPAP_PROFILE("bkpap_bin_end", inter_rank);

	return OMPI_SUCCESS;
	// return OPAL_ERR_NOT_IMPLEMENTED;
bkpap_abort_binomial_allreduce:
	free(c);
	return ret;
}