#ifndef NUMO_LINALG_ALT_BLAS_GEMM_H
#define NUMO_LINALG_ALT_BLAS_GEMM_H 1

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

#include "../converter.h"
#include "../types.h"
#include "../util.h"

#include <cblas.h>

void define_linalg_blas_gemm(VALUE mBlas);

#endif /* NUMO_LINALG_ALT_BLAS_GEMM_H */
