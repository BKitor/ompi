#!/cvmfs/soft.computecanada.ca/gentoo/2020/bin/bash

if [ "$BK_OMB_DIR" == "" ];then
	echo "No BK_OMPI_DIR, you need to initenv"
fi

pushd ../../..
make install
popd

mpirun -n 4 \
	--mca coll ^tuned \
	--mca coll_base_verbose 9 \
	$BK_OMB_DIR/collective/osu_allreduce \
	-i 1 -x 0 -m "$((1<<20)):$((1<<20))"