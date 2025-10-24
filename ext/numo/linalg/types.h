#ifndef NUMO_LINALG_ALT_BLAS_TYPES_H
#define NUMO_LINALG_ALT_BLAS_TYPES_H 1

#include "extconf.h"

#ifndef HAVE_TYPE_BLAS_INT
#ifdef f77_int
typedef f77_int blasint;
#else
#ifdef CBLAS_INT
typedef CBLAS_INT blasint;
#else
typedef int blasint;
#endif
#endif
#endif

#ifndef _DEFINED_SCOMPLEX
#define _DEFINED_SCOMPLEX 1
#endif
#ifndef _DEFINED_DCOMPLEX
#define _DEFINED_DCOMPLEX 1
#endif

#endif /* NUMO_LINALG_ALT_BLAS_TYPES_H */
