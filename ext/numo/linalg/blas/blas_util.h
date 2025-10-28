#ifndef NUMO_LINALG_ALT_BLAS_UTIL_H
#define NUMO_LINALG_ALT_BLAS_UTIL_H 1

#include <ruby.h>

#include <cblas.h>

enum CBLAS_TRANSPOSE get_cblas_trans(VALUE val);
enum CBLAS_ORDER get_cblas_order(VALUE val);

#endif // NUMO_LINALG_ALT_BLAS_UTIL_H
