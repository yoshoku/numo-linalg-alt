#ifndef NUMO_LINALG_ALT_LAPACK_UNGRQ_H
#define NUMO_LINALG_ALT_LAPACK_UNGRQ_H 1

#include <lapacke.h>

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

#include "lapack_util.h"

void define_linalg_lapack_ungrq(VALUE mLapack);

#endif /* NUMO_LINALG_ALT_LAPACK_UNGRQ_H */
