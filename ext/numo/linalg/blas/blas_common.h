#ifndef NUMO_LINALG_ALT_BLAS_COMMON_H
#define NUMO_LINALG_ALT_BLAS_COMMON_H 1

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

#ifndef _DEFINED_SCOMPLEX
#define _DEFINED_SCOMPLEX 1
#endif
#ifndef _DEFINED_DCOMPLEX
#define _DEFINED_DCOMPLEX 1
#endif

#include <cblas.h>

#ifndef CBLAS_INT
#define CBLAS_INT blasint
#endif

#endif /* NUMO_LINALG_ALT_BLAS_COMMON_H */
