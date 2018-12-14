TRIBITS_PACKAGE_DEFINE_DEPENDENCIES(
  SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
    #SubPackageName       Directory         Class    Req/Opt
    #
    # New Kokkos subpackages:
    KokkosKernels         src              PS       REQUIRED
    Example               example          EX      OPTIONAL
  )
