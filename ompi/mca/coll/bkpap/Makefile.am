dist_ompidata_DATA = help-mpi-coll-bkpap.txt

sources = coll_bkpap_module.c coll_bkpap_allreduce.c \
		  coll_bkpap_component.c coll_bkpap_ucp.c \
		  coll_bkpap.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_bkpap_DSO
component_noinst =
component_install = mca_coll_bkpap.la
else
component_noinst = libmca_coll_bkpap.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_bkpap_la_SOURCES = $(sources)
mca_coll_bkpap_la_LDFLAGS = -module -avoid-version
mca_coll_bkpap_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_bkpap_la_SOURCES =$(sources)
libmca_coll_bkpap_la_LDFLAGS = -module -avoid-version
