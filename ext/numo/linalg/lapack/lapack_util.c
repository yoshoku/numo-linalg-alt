#include "lapack_util.h"

lapack_int get_itype(VALUE val) {
  const lapack_int itype = NUM2INT(val);

  if (itype != 1 && itype != 2 && itype != 3) {
    rb_raise(rb_eArgError, "itype must be 1, 2 or 3");
  }

  return itype;
}

char get_job(VALUE val, const char* param_name) {
  const char job = NUM2CHR(val);

  if (job != 'N' && job != 'V') {
    rb_raise(rb_eArgError, "%s must be 'N' or 'V'", param_name);
  }

  return job;
}

char get_range(VALUE val) {
  const char range = NUM2CHR(val);

  if (range != 'A' && range != 'V' && range != 'I') {
    rb_raise(rb_eArgError, "range must be 'A', 'V' or 'I'");
  }

  return range;
}

char get_uplo(VALUE val) {
  const char uplo = NUM2CHR(val);

  if (uplo != 'U' && uplo != 'L') {
    rb_raise(rb_eArgError, "uplo must be 'U' or 'L'");
  }

  return uplo;
}

int get_matrix_layout(VALUE val) {
  const char option = NUM2CHR(val);

  switch (option) {
  case 'r':
  case 'R':
    break;
  case 'c':
  case 'C':
    rb_warn("Numo::Linalg does not support column major.");
    break;
  }

  return LAPACK_ROW_MAJOR;
}
