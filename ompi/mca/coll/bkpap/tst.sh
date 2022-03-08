#!/usr/bin/bash

if [ "$BK_OMB_DIR" == "" ];then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

bk_osu_tst(){
	echo "bkpap_allreduce_alg: $OMPI_MCA_coll_bkpap_allreduce_alg"
	echo "bkpap_postbuf_mem_type: $OMPI_MCA_coll_bkpap_postbuf_mem_type"
	mpirun -n $BK_NUM_PROC \
		--display bind \
		$BK_OSU_PAP \
		-m "$BK_MIN_MSIZE:$BK_MAX_MSIZE" \
		$BK_OMB_FLAGS $BK_EXP_FLAGS
		# -m "$((1<<3)):$((1<<23))" # -i 130 -x 1 
		# -m "$((1<<20)):$((1<<23))" -i 130 -x 1 

		# --map-by :OVERSUBSCRIBE \
		# --mca coll_bkpap_allreduce_k_value 2 \
		# --map-by ppr:4:package \
}

bk_val_tst(){
	mpicc -o ar_val.out ar_val.c
	mpirun -n $BK_NUM_PROC \
		--map-by core \
		./ar_val.out
}

bk_val_cu_tst(){
	mpicc -o ar_val_cu.out ar_val_cu.c
	mpirun -n $BK_NUM_PROC \
		--map-by core \
		./ar_val_cu.out
}

BK_MIN_MSIZE=$((1<<22))
BK_MAX_MSIZE=$((1<<24))
BK_EXP_FLAGS="-x 0 -i 10"
# BK_EXP_FLAGS="-x 5 -i 50"

# export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_pml=^ucx
export OMPI_MCA_coll_bkpap_pipeline_segment_size=$((1<<20))
export OMPI_MCA_coll_bkpap_postbuff_size=$BK_MAX_MSIZE
export UCX_IB_PREFER_NEAREST_DEVICE=no

BK_OSU_PAP="$BK_OMB_DIR/collective/osu_allreduce"

export OMPI_MCA_coll_bkpap_allreduce_alg=0
export OMPI_MCA_coll_cuda_priority=31
export OMPI_MCA_coll_bkpap_priority=35
export OMPI_MCA_coll_base_verbose=9
# BK_OMB_FLAGS=""
# BK_NUM_PROC=4 bk_osu_tst

export OMPI_MCA_coll_bkpap_postbuf_mem_type=0
BK_OMB_FLAGS=""
# BK_NUM_PROC=4 UCX_LOG_LEVEL=debug bk_osu_tst
BK_NUM_PROC=4 bk_osu_tst

export OMPI_MCA_coll_bkpap_allreduce_alg=1
BK_NUM_PROC=4 bk_osu_tst

# export OMPI_MCA_coll_bkpap_postbuf_mem_type=1
# BK_OMB_FLAGS="-d cuda"
# BK_NUM_PROC=4 bk_osu_tst

# export OMPI_MCA_coll_ucc_priority=35
# export OMPI_MCA_coll_ucc_enable=1
# export OMPI_MCA_coll_bkpap_priority=29
# BK_OMB_FLAGS="-d cuda"
# BK_NUM_PROC=4 bk_osu_tst

# bk_osu_tst
# export OMPI_MCA_coll_bkpap_priority=10
# bk_osu_tst

# bk_osu_def
