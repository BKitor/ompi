#!/usr/bin/bash

BK_NUM_PROC=4
BK_MEM_TYPE="c"
print_help() {
	echo "Usage: -v <verbosity> -a <bkpap_alg> -m <memtype> -u -n"
	echo "-v <verbosity> -- 6 for profiling, 9 for full output"
	echo "-a <bkpap_alg> -- run a bkapap algorithm, options are [0..5]"
	echo "-u -- flag to run ucp"
	echo "-n -- flag to run nccl"
	echo "-m <memory location> -- [h/c/m] for host/cuda/managed, default: $BK_MEM_TYPE"
	echo "-p <num_procs> -- number of processes to run, default: $BK_NUM_PROC"
	exit
}

if [ "$BK_OMB_DIR" == "" ]; then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

if [ "$#" == "0" ]; then
	print_help
fi

while getopts ":v:a:unm:dp:" bk_opt; do
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
	d)
		BK_RUN_DEF=1
		;;
	p)
		BK_NUM_PROC="${OPTARG}"
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
	echo "bkpap_allreduce_alg: $OMPI_MCA_coll_bkpap_allreduce_alg"
	echo "bkpap_postbuf_mem_type: $OMPI_MCA_coll_bkpap_postbuf_mem_type"
	echo "ucc_en: $OMPI_MCA_coll_ucc_enable"
	echo "ucc_tl: $UCC_CL_BASIC_TLS"
	echo "val_nf: $BK_VAL_FN"
}

bk_val_cond_tst() {
	if [ "$1" == "1" ]; then
		bk_exp_out
		mpicc -o test/ar_val.out test/ar_val.c -Wall &&
			mpirun -n $BK_NUM_PROC \
				--display bind \
				--map-by core \
				./test/ar_val.out
	fi
}

bk_val_cu_cond_tst() {
	if [ "$1" == "1" ]; then
		bk_exp_out
		mpicc -o test/ar_val_cu.out test/ar_val_cu.c -Wall -lcuda -lcudart &&
			mpirun -n $BK_NUM_PROC \
				--display bind \
				--map-by core \
				./test/ar_val_cu.out
	fi
}

# export OMPI_MCA_pml=^ucx
# export OMPI_MCA_osc=^ucx
export UCX_IB_MLX5_DEVX=no
# export UCX_SHM_DEVICES=""
# export UCX_IB_PREFER_NEAREST_DEVICE=no
# export UCX_NET_DEVICES=mlx5_0:1
BK_VAL_FN=bk_val_cond_tst
export OMPI_MCA_coll_bkpap_postbuf_mem_type=0
if [ "c" = "$BK_MEM_TYPE" ]; then
	export OMPI_MCA_coll_bkpap_postbuf_mem_type=1
	BK_VAL_FN=bk_val_cu_cond_tst
elif [ "m" = "$BK_MEM_TYPE" ]; then
	export OMPI_MCA_coll_bkpap_postbuf_mem_type=2
	BK_VAL_FN=bk_val_cu_cond_tst
fi

export OMPI_MCA_coll_cuda_priority=31
export OMPI_MCA_coll_bkpap_priority=35
export OMPI_MCA_coll_ucc_enable=0
export UCC_CL_BASIC_TLS=all

export OMPI_MCA_coll_bkpap_dataplane_type=1
export OMPI_MCA_coll_bkpap_postbuff_size=$((1<<25))
export OMPI_MCA_coll_bkpap_pipeline_segment_size=$((1 << 25))

export OMPI_MCA_coll_bkpap_allreduce_alg=0
$BK_VAL_FN $BK_RUN_ALG0

export OMPI_MCA_coll_bkpap_allreduce_alg=1
$BK_VAL_FN $BK_RUN_ALG1

export OMPI_MCA_coll_bkpap_allreduce_alg=2
$BK_VAL_FN $BK_RUN_ALG2

export OMPI_MCA_coll_bkpap_allreduce_alg=3
$BK_VAL_FN $BK_RUN_ALG3

export OMPI_MCA_coll_bkpap_allreduce_alg=4
$BK_VAL_FN $BK_RUN_ALG4

export OMPI_MCA_coll_bkpap_allreduce_alg=5
$BK_VAL_FN $BK_RUN_ALG5

export OMPI_MCA_coll_bkpap_allreduce_alg=6
$BK_VAL_FN $BK_RUN_ALG6

export OMPI_MCA_coll_ucc_priority=35
export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_bkpap_priority=29
export UCC_CL_BASIC_TLS=ucp
$BK_VAL_FN $BK_RUN_UCC
export UCC_CL_BASIC_TLS=all
$BK_VAL_FN $BK_RUN_NCCL

export OMPI_MCA_coll_cuda_priority=78
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll_ucc_priority=28
export OMPI_MCA_coll_bkpap_priority=28
$BK_VAL_FN $BK_RUN_DEF
