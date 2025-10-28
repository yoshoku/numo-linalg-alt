#ifndef NUMO_LINALG_ALT_LAPACK_UTIL_H
#define NUMO_LINALG_ALT_LAPACK_UTIL_H 1

#include <ruby.h>

#include <lapacke.h>

lapack_int get_itype(VALUE val);
char get_job(VALUE val, const char* param_name);
char get_range(VALUE val);
char get_uplo(VALUE val);
int get_matrix_layout(VALUE val);

#endif // NUMO_LINALG_ALT_LAPACK_UTIL_H
