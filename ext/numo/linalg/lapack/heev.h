#ifndef NUMO_LINALG_ALT_LAPACK_HEEV_H
#define NUMO_LINALG_ALT_LAPACK_HEEV_H 1

#include <lapacke.h>

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

#include "../util.h"

void define_linalg_lapack_heev(VALUE mLapack);

#endif /* NUMO_LINALG_ALT_LAPACK_HEEV_H */
