#ifndef NUMO_LINALG_ALT_BLAS_GEMV_H
#define NUMO_LINALG_ALT_BLAS_GEMV_H 1

#include <ruby.h>

#include <cblas.h>

#include <numo/narray.h>
#include <numo/template.h>

#include "blas_util.h"
#include "converter.h"

void define_linalg_blas_gemv(VALUE mBlas);

#endif /* NUMO_LINALG_ALT_BLAS_GEMV_H */
