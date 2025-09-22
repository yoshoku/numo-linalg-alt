#ifndef NUMO_LINALG_ALT_UTIL_H
#define NUMO_LINALG_ALT_UTIL_H 1

#include <ruby.h>

#include <cblas.h>
#include <lapacke.h>

lapack_int get_itype(VALUE val);
char get_jobz(VALUE val);
char get_range(VALUE val);
char get_uplo(VALUE val);
int get_matrix_layout(VALUE val);
enum CBLAS_TRANSPOSE get_cblas_trans(VALUE val);
enum CBLAS_ORDER get_cblas_order(VALUE val);

#endif // NUMO_LINALG_ALT_UTIL_H
