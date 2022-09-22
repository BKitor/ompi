#!/usr/bin/bash

BK_NUM_PROC=4
BK_MEM_TYPE="c"
BK_DPLANE_T=1
print_help() {
	echo "Usage: -v <verbosity> -a <bkpap_alg> -m <memtype> -u -n -s"
	echo "-v <verbosity> -- 6 for profiling, 9 for full output"
	echo "-a <bkpap_alg> -- run a bkapap algorithm, options are [0..5]"
	echo "-u -- flag to run ucp"
	echo "-n -- flag to run nccl"
	echo "-m <memory location> -- [h/c/m] for host/cuda/managed, default: $BK_MEM_TYPE"
	echo "-s -- single run, sets msize to 1MB and OMB flags '-x 0 -i 1'"
	echo "-p <num_procs> -- number of processes to run, default: $BK_NUM_PROC"
	echo "-D run default"
	echo "-d <dataplane> 0: RMA 1: TAG default: $BK_DPLANE_T"
	exit
}

if [ "$BK_OMB_DIR" == "" ]; then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

if [ "$#" == "0" ]; then
	print_help
fi

while getopts ":v:a:unm:d:sp:D" bk_opt; do
	case "${bk_opt}" in
	v)
		export OMPI_MCA_coll_bkpap_verbose="${OPTARG}"
		;;
	a)
		case "${OPTARG}" in
		0)
			BK_RUN_ALG0=1
			;;
		1)
			BK_RUN_ALG1=1
			;;
		2)
			BK_RUN_ALG2=1
			;;
		3)
			BK_RUN_ALG3=1
			;;
		4)
			BK_RUN_ALG4=1
			;;
		5)
			BK_RUN_ALG5=1
			;;
		6)
			BK_RUN_ALG6=1
			;;
		*)
			echo "ERORR: bad value '-a ${OPTARG}'"
			print_help
			;;
		esac
		;;
	m)
		if [ "${OPTARG}" = "h" -o "${OPTARG}" = "c" -o "${OPTARG}" = "m" ]; then
			BK_MEM_TYPE="${OPTARG}"
		else
			echo "ERORR: bad value '-m ${OPTARG}'"
			print_help
		fi
		;;
	u)
		BK_RUN_UCC=1
		;;
	n)
		BK_RUN_NCCL=1
		;;
	D)
		BK_RUN_DEF=1
		;;
	s)
		BK_SINGLE_OMB=1
		;;
	p)
		BK_NUM_PROC="${OPTARG}"
		;;
	d)
		BK_DPLANE_T="${OPTARG}"
		;;
	:)
		echo "ERROR: -${OPTARG} requires and argument."
		print_help
		;;
	\?)
		print_help
		;;
	*) ;;
	esac
done

bk_exp_out() {
	echo "bkpap_prio: $OMPI_MCA_coll_bkpap_priority"
	echo "bkpap_alg: $OMPI_MCA_coll_bkpap_allreduce_alg"
	echo "bkpap_postbuf_mem_type: $OMPI_MCA_coll_bkpap_postbuf_mem_type"
	echo "bkpap_dataplane_type: $OMPI_MCA_coll_bkpap_dataplane_type"
	echo "bkpap_seg_size: $OMPI_MCA_coll_bkpap_pipeline_segment_size"
	echo "ucc_prio: $OMPI_MCA_coll_ucc_priority"
	echo "ucc_enable: $OMPI_MCA_coll_ucc_enable"
	echo "cuda_prio: $OMPI_MCA_coll_cuda_priority"
}

bk_cond_osu_tst() {
	if [ "$1" == "1" ]; then
		bk_exp_out
		mpirun -n $BK_NUM_PROC \
			--display bind \
			--bind-to pack \
			--map-by core \
			$BK_OSU_PAP \
			-m "$BK_MIN_MSIZE:$BK_MAX_MSIZE" \
			$BK_EXP_FLAGS # -F 10
	fi
}

# export OMPI_MCA_pml=^ucx
# export OMPI_MCA_osc=^ucx
export UCX_IB_MLX5_DEVX=no
# export UCX_SHM_DEVICES=""
# export UCX_IB_PREFER_NEAREST_DEVICE=no
# export UCX_NET_DEVICES=mlx5_0:1

BK_EXP_FLAGS="-x 10 -i 100"
if [ "1" == "$BK_SINGLE_OMB" ]; then
	BK_EXP_FLAGS="-x 0 -i 1"
fi
BK_MIN_MSIZE=$((1 << 19))

export OMPI_MCA_coll_bkpap_postbuf_mem_type=0
if [ "c" = "$BK_MEM_TYPE" ]; then
	BK_EXP_FLAGS+=" -d cuda"
	export OMPI_MCA_coll_bkpap_postbuf_mem_type=1
	BK_MIN_MSIZE=$((1 << 25))
elif [ "m" = "$BK_MEM_TYPE" ]; then
	BK_EXP_FLAGS+=" -d managed"
	export OMPI_MCA_coll_bkpap_postbuf_mem_type=2
	BK_MIN_MSIZE=$((1 << 25))
fi

if [ "1" == "$BK_SINGLE_OMB" ]; then
	BK_MAX_MSIZE=$BK_MIN_MSIZE
else
	BK_MAX_MSIZE=$((BK_MIN_MSIZE << 4))
fi

echo "exp_flags: $BK_EXP_FLAGS, min_size: $BK_MIN_MSIZE, max_size: $BK_MAX_MSIZE"

BK_OSU_PAP="$BK_OMB_DIR/build/libexec/osu-micro-benchmarks/mpi/collective/bk_osu_pap_allreduce"

export OMPI_MCA_coll_cuda_priority=31
export OMPI_MCA_coll_bkpap_priority=35
export OMPI_MCA_coll_ucc_priority=20
export OMPI_MCA_coll_ucc_enable=0
export UCC_CL_BASIC_TLS=all

export OMPI_MCA_coll_bkpap_dataplane_type=$BK_DPLANE_T
export OMPI_MCA_coll_bkpap_postbuff_size=$BK_MAX_MSIZE
export OMPI_MCA_coll_bkpap_pipeline_segment_size=$BK_MIN_MSIZE

export OMPI_MCA_coll_bkpap_allreduce_alg=0
bk_cond_osu_tst $BK_RUN_ALG0

export OMPI_MCA_coll_bkpap_allreduce_alg=1
bk_cond_osu_tst $BK_RUN_ALG1

export OMPI_MCA_coll_bkpap_allreduce_alg=2
bk_cond_osu_tst $BK_RUN_ALG2

export OMPI_MCA_coll_bkpap_allreduce_alg=3
bk_cond_osu_tst $BK_RUN_ALG3

export OMPI_MCA_coll_bkpap_allreduce_alg=4
bk_cond_osu_tst $BK_RUN_ALG4

export OMPI_MCA_coll_bkpap_allreduce_alg=5
bk_cond_osu_tst $BK_RUN_ALG5

export OMPI_MCA_coll_bkpap_allreduce_alg=6
bk_cond_osu_tst $BK_RUN_ALG6

export OMPI_MCA_coll_ucc_priority=35
export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_bkpap_priority=29
export UCC_CL_BASIC_TLS=ucp
bk_cond_osu_tst $BK_RUN_UCC
export UCC_CL_BASIC_TLS=all
bk_cond_osu_tst $BK_RUN_NCCL

export OMPI_MCA_coll_cuda_priority=78
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll_ucc_priority=28
export OMPI_MCA_coll_bkpap_priority=28
bk_cond_osu_tst $BK_RUN_DEF
