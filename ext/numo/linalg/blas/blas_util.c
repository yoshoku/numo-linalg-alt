#include "blas_util.h"

enum CBLAS_TRANSPOSE get_cblas_trans(VALUE val) {
  const char option = NUM2CHR(val);
  enum CBLAS_TRANSPOSE res = CblasNoTrans;

  switch (option) {
  case 'n':
  case 'N':
    res = CblasNoTrans;
    break;
  case 't':
  case 'T':
    res = CblasTrans;
    break;
  case 'c':
  case 'C':
    res = CblasConjTrans;
    break;
  }

  return res;
}

enum CBLAS_ORDER get_cblas_order(VALUE val) {
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

  return CblasRowMajor;
}
