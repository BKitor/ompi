dist_ompidata_DATA = help-mpi-coll-bkpap.txt

NVCCFLAGS = ${NVCC_CFLAGS} ${AM_CPPFLAGS} ${coll_bkpap_CPPFLAGS} --compiler-options -fno-rtti,-fno-exceptions 

# LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(NVCC_ARCH) 

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) -c $< $(NVCCFLAGS) $(NVCC_ARCH)
	
headers = coll_bkpap.h coll_bkpap_ucp.inl 

sources = coll_bkpap_module.c coll_bkpap_allreduce.c coll_bkpap_component.c \
		  coll_bkpap_ucp.c \
		  bkpap_kernel.h bkpap_kernel.cu \
		  coll_bkpap_reduce.c \
		  coll_bkpap_base_rsa_gpu.c \
		  coll_bkpap_dplane_rma.c coll_bkpap_dplane_tag.c 
		 

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

AM_CPPFLAGS = $(coll_bkpap_CPPFLAGS)

if MCA_BUILD_ompi_coll_bkpap_DSO
component_noinst =
component_install = mca_coll_bkpap.la
else
component_noinst = libmca_coll_bkpap.la
component_install =
endif

coll_bkpap_LDFLAGS += -lcudart

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_bkpap_la_SOURCES = $(sources)
mca_coll_bkpap_la_LDFLAGS = -module -avoid-version $(coll_bkpap_LDFLAGS)
mca_coll_bkpap_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la $(coll_bkpap_LIBS) 

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_bkpap_la_SOURCES =$(sources)
libmca_coll_bkpap_la_LIBADD = $(coll_bkpap_LIBS) 
libmca_coll_bkpap_la_LDFLAGS = -module -avoid-version $(coll_bkpap_LDFLAGS) 
