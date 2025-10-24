#ifndef NUMO_LINALG_ALT_BLAS_TYPES_H
#define NUMO_LINALG_ALT_BLAS_TYPES_H

#include "extconf.h"

#ifndef HAVE_TYPE_BLAS_INT
#ifdef MKL_INT
typedef MKL_INT blasint;
#else
#ifdef CBLAS_INT
typedef CBLAS_INT blasint;
#else
typedef int blasint;
#endif
#endif
#endif

#endif /* NUMO_LINALG_ALT_BLAS_TYPES_H */
