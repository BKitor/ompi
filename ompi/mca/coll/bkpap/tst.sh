#!/cvmfs/soft.computecanada.ca/gentoo/2020/bin/bash

set -e

if [ "$BK_OMB_DIR" == "" ];then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

# --mca coll ^tuned \
# make install

BK_PBUF_SIZE=$((1<<22))
BK_NUM_PROC=4

export OMPI_MCA_coll_bkpap_postbuff_size=$BK_PBUF_SIZE
export OMPI_MCA_coll_bkpap_postbuff_size=$((1<<20))
export UCX_IB_PREFER_NEAREST_DEVICE=no

BK_OSU_PAP="$BK_OMB_DIR/build/libexec/osu-micro-benchmarks/mpi/collective/bk_osu_pap_allreduce"

bk_osu_tst(){
	mpirun -n $BK_NUM_PROC \
		--display bind \
		$BK_OSU_PAP \
		-i 1 -x 0 -m "$BK_PBUF_SIZE:$BK_PBUF_SIZE"
		# -m "$((1<<3)):$((1<<23))" # -i 130 -x 1 
		# -m "$((1<<20)):$((1<<23))" -i 130 -x 1 

		# --map-by :OVERSUBSCRIBE \
		# --mca coll_bkpap_allreduce_k_value 2 \
		# --map-by ppr:4:package \
}

bk_osu_def(){
	export OMPI_MCA_coll=^bkpap
	bk_osu_tst
	export OMPI_MCA_coll=^tuned
}

bk_val_tst(){
	mpicc -o ar_val.out ar_val.c
	mpirun -n $BK_NUM_PROC \
		--map-by core \
		./ar_val.out
}


# BK_NUM_PROC=4
# bk_val_tst
# BK_NUM_PROC=8
# bk_val_tst

export OMPI_MCA_coll_bkpap_priority=35
export OMPI_MCA_coll_base_verbose=9
BK_NUM_PROC=4
bk_osu_tst
# BK_NUM_PROC=8
# bk_osu_tst
# export OMPI_MCA_coll_bkpap_priority=10
# bk_osu_tst

# bk_osu_def