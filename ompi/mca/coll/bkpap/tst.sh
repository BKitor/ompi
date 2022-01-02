#!/cvmfs/soft.computecanada.ca/gentoo/2020/bin/bash

set -e

if [ "$BK_OMB_DIR" == "" ];then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

# --mca coll_base_verbose 9 \
# --mca coll ^tuned \
# make install

export OMPI_MCA_coll_base_verbose=9
export OMPI_MCA_coll=^tuned

mpirun -n 8 \
	--display bind \
	$BK_OMB_DIR/collective/osu_allreduce \
	-i 2 -x 0 -m "$((1<<20)):$((1<<23))"
	# -i 1 -x 0 -m "$((1<<20)):$((1<<20))"

	# --map-by ppr:4:package \