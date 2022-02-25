#
# Benjamin Kitor PPRL 2021
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#


AC_DEFUN([MCA_ompi_coll_bkpap_CONFIG], [
    AC_CONFIG_FILES([ompi/mca/coll/bkpap/Makefile])

    OMPI_CHECK_UCX([coll_bkpap],
                   [coll_bkpap_happy="yes"],
                   [coll_bkpap_happy="no"])

    AS_IF([test "$coll_bkpap_happy" = "yes"],
          [$1],
          [$2])

    # make sure that CUDA-aware checks have been done
    AC_REQUIRE([OPAL_CHECK_CUDA])

    # Only build if CUDA support is available
    AS_IF([test "x$CUDA_SUPPORT" = "x1"],
          [$1],
          [$2])

    # substitute in the things needed to build ucx
    AC_SUBST([coll_bkpap_CPPFLAGS])
    AC_SUBST([coll_bkpap_LDFLAGS])
    AC_SUBST([coll_bkpap_LIBS])
])
