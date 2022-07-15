#include "coll_bkpap.h"
#include "coll_bkpap_util.inl"
#include "coll_bkpap_ucp.inl"
#include "opal/util/bit_ops.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

/*
 * Under current desing, the root needs to recieve multiple messages,
 * This creates a lot of overhead for pre-posted buffer on the root proc
 * For a 256 wsize, log(256)*64MB would need to be reserved, (512MB data for each proc)
 *
 * modify tree so that the root has a parent (being last proc)
 *
 *
 * Tree is built by taking ompi_coll_base_topo_build_kmtree(), and substracting 2 from each rank
 * Example, comm_size=10
 *    radix=2
 *      0          		       6
 *     /  \  \       		 /  \  \
 *    4    2  1     r - 2	2     0  7
 *    | \  |     	----> 	| \   |
 *    6  5 3     			4  3  1
 *    |						|
 *    7						5
 *
*/

int coll_bkpap_papaware_binomial_allreduce(const void* sbuf, void* rbuf, int count, struct ompi_datatype_t* dtype,
	struct ompi_op_t* op, struct ompi_communicator_t* intra_comm,
	struct ompi_communicator_t* inter_comm, mca_coll_bkpap_module_t* bkpap_module) {

	int ret = OMPI_SUCCESS;
	int inter_size = ompi_comm_size(inter_comm), inter_rank = ompi_comm_rank(inter_comm);
	int intra_rank = ompi_comm_rank(intra_comm);
	int is_inter = (0 == intra_rank);
	int vrank, mask = 1, p = -1, * c = NULL, tree_depth, nc = 0;
	ptrdiff_t extent, lb;
	ompi_datatype_get_extent(dtype, &lb, &extent);

	if (is_inter)BKPAP_PROFILE("bkpap_bin_start", inter_rank);

	// intra_reduce
	ret = bk_intra_reduce(rbuf, count, dtype, op, intra_comm, bkpap_module);
	BKPAP_CHK_MPI_MSG_LBL(ret, "intra reduce failed", bkpap_abort_binomial_allreduce);
	if (is_inter)BKPAP_PROFILE("bkpap_bin_intra_reduce", inter_rank);

	if (is_inter) {
		void* late_recv_buf = NULL, * early_recv_buf = NULL;
		mca_coll_bkpap_get_coll_recvbuf(&late_recv_buf, bkpap_module);
		bkpap_mempool_alloc(&early_recv_buf, extent * count, bkpap_module->dplane_mem_t, bkpap_module);
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
			BK_BINOMIAL_MAKE_TAG(peer, 0, tag, tag_mask);
			if (peer > arrival) {
				BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (late) child: %d", inter_rank, arrival, peer);

				ret = bkpap_module->dplane_ftbl.recv_from_late(late_recv_buf, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_late failed", bkpap_abort_binomial_allreduce);

				// BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (late) child: %d, sample_val: %f", inter_rank, arrival, peer, ((float*)late_recv_buf)[0]);
				// BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (late) child: %d", inter_rank, arrival, peer);

				ret = mca_coll_bkpap_reduce_local(op, late_recv_buf, rbuf, count, dtype, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "reduce_local failed", bkpap_abort_binomial_allreduce);
				// BKPAP_OUTPUT("rank: %d, arrival: %d, reduce_loca: %f", inter_rank, arrival, ((float*)rbuf)[0]);
				ret = bkpap_module->dplane_ftbl.reset_late_recv_buf(bkpap_module);
			}
			else {
				int peer_rank = -1;
				while (-1 == peer_rank) {
					ret = mca_coll_bkpap_get_rank_of_arrival(peer, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
					BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival peer failed", bkpap_abort_binomial_allreduce);
				}
				BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (early) child: %d(%d)", inter_rank, arrival, peer, peer_rank);

				ret = bkpap_module->dplane_ftbl.recv_from_early(early_recv_buf, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "recv_from_early failed", bkpap_abort_binomial_allreduce);
				// BKPAP_OUTPUT("rank: %d, arrival: %d, recv from (early) child: %d(%d), sample: %f", inter_rank, arrival, peer, peer_rank, ((float*)early_recv_buf)[0]);

				ret = mca_coll_bkpap_reduce_local(op, early_recv_buf, rbuf, count, dtype, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "reduce_local failed", bkpap_abort_binomial_allreduce);
				BKPAP_OUTPUT("rank: %d, arrival: %d, reduce_loca: %f", inter_rank, arrival, ((float*)rbuf)[0]);
			}
		}
		BKPAP_PROFILE("bkpap_bin_recv_child", inter_rank);

		if (-1 != p) {
			BK_BINOMIAL_MAKE_TAG(arrival, 0, tag, tag_mask);
			if (p > arrival) {
				BKPAP_OUTPUT("rank: %d, arrival: %d, send to (late) parent: %d", inter_rank, arrival, p);
				// BKPAP_OUTPUT("rank: %d, arrival: %d, send to (late) parent: %d, samlpe: %f", inter_rank, arrival, p, ((float*)rbuf)[0]);

				ret = bkpap_module->dplane_ftbl.send_to_late(rbuf, count, dtype, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_late failed", bkpap_abort_binomial_allreduce);
			}
			else {
				int peer_rank = -1;
				while (-1 == peer_rank) {
					ret = mca_coll_bkpap_get_rank_of_arrival(p, 0, bkpap_module->remote_syncstructure, bkpap_module, &peer_rank);
					BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival parent failed", bkpap_abort_binomial_allreduce);
				}
				// BKPAP_OUTPUT("rank: %d, arrival: %d, send to (early) parent: %d(%d), sample: %f", inter_rank, arrival, p, peer_rank, ((float*)rbuf)[0]);
				BKPAP_OUTPUT("rank: %d, arrival: %d, send to (early) parent: %d(%d)", inter_rank, arrival, p, peer_rank);

				ret = bkpap_module->dplane_ftbl.send_to_early(rbuf, count, dtype, peer_rank, tag, tag_mask, inter_comm, bkpap_module);
				BKPAP_CHK_MPI_MSG_LBL(ret, "send_to_early failed", bkpap_abort_binomial_allreduce);

				BKPAP_OUTPUT("DBG rank: %d, arrival: %d, send to (early) done", inter_rank, arrival);
			}
		}
		BKPAP_PROFILE("bkpap_bin_send_parent", inter_rank);
		bk_ompi_request_wait_all(&ibarrer_req, 1);

		int bcast_root = -1;
		while (-1 == bcast_root) {
			ret = mca_coll_bkpap_get_rank_of_arrival(inter_size - 2, 0, bkpap_module->remote_syncstructure, bkpap_module, &bcast_root);
			BKPAP_CHK_MPI_MSG_LBL(ret, "get_rank_of_arrival bcast_root failed", bkpap_abort_binomial_allreduce);
		}

		ret = bk_inter_bcast(rbuf, count, dtype, bcast_root, inter_comm, bkpap_module,
			mca_coll_bkpap_component.pipeline_segment_size);
		BKPAP_CHK_MPI_MSG_LBL(ret, "bk_opt_bc bcast failed", bkpap_abort_binomial_allreduce);

		BKPAP_PROFILE("bkpap_bin_inter_bcast", inter_rank);

		bkpap_mempool_free(early_recv_buf, bkpap_module->dplane_mem_t, bkpap_module);
	}

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

bkpap_abort_binomial_allreduce:
	free(c);
	return ret;
}