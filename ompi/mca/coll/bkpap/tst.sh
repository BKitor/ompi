#!/cvmfs/soft.computecanada.ca/gentoo/2020/bin/bash

set -e

if [ "$BK_OMB_DIR" == "" ];then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

# --mca coll_base_verbose 9 \
# --mca coll ^tuned \
# make install

BK_PBUF_SIZE=$((1<<26))

export OMPI_MCA_coll_base_verbose=9
export OMPI_MCA_coll=^tuned
export OMPI_MCA_coll_postbuf_size=$BK_PBUF_SIZE
# export UCX_NET_DEVICES=mlx5_2:1
export UCX_IB_PREFER_NEAREST_DEVICE=no

mpirun -n 4 \
	--display bind \
	$BK_OMB_DIR/collective/osu_allreduce \
	-i 1 -x 0 -m "$BK_PBUF_SIZE:$BK_PBUF_SIZE"
	# -i 2 -x 0 -m "$((1<<20)):$((1<<23))"

	# --map-by :OVERSUBSCRIBE \
	# --mca coll_bkpap_allreduce_k_value 2 \
	# --map-by ppr:4:package \
