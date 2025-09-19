#include "dot_sub.h"

#define DEF_LINALG_FUNC(tElType, tNArrType, fBlasFnc)                            \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                            \
    tElType* x = (tElType*)NDL_PTR(lp, 0);                                       \
    tElType* y = (tElType*)NDL_PTR(lp, 1);                                       \
    tElType* d = (tElType*)NDL_PTR(lp, 2);                                       \
    const size_t n = NDL_SHAPE(lp, 0)[0];                                        \
    cblas_##fBlasFnc(n, x, 1, y, 1, d);                                          \
  }                                                                              \
                                                                                 \
  static VALUE _linalg_blas_##fBlasFnc(VALUE self, VALUE x, VALUE y) {           \
    if (CLASS_OF(x) != tNArrType) {                                              \
      x = rb_funcall(tNArrType, rb_intern("cast"), 1, x);                        \
    }                                                                            \
    if (!RTEST(nary_check_contiguous(x))) {                                      \
      x = nary_dup(x);                                                           \
    }                                                                            \
    if (CLASS_OF(y) != tNArrType) {                                              \
      y = rb_funcall(tNArrType, rb_intern("cast"), 1, y);                        \
    }                                                                            \
    if (!RTEST(nary_check_contiguous(y))) {                                      \
      y = nary_dup(y);                                                           \
    }                                                                            \
                                                                                 \
    narray_t* x_nary = NULL;                                                     \
    GetNArray(x, x_nary);                                                        \
    narray_t* y_nary = NULL;                                                     \
    GetNArray(y, y_nary);                                                        \
                                                                                 \
    if (NA_NDIM(x_nary) != 1) {                                                  \
      rb_raise(rb_eArgError, "x must be 1-dimensional");                         \
      return Qnil;                                                               \
    }                                                                            \
    if (NA_NDIM(y_nary) != 1) {                                                  \
      rb_raise(rb_eArgError, "y must be 1-dimensional");                         \
      return Qnil;                                                               \
    }                                                                            \
    if (NA_SIZE(x_nary) == 0) {                                                  \
      rb_raise(rb_eArgError, "x must not be empty");                             \
      return Qnil;                                                               \
    }                                                                            \
    if (NA_SIZE(y_nary) == 0) {                                                  \
      rb_raise(rb_eArgError, "x must not be empty");                             \
      return Qnil;                                                               \
    }                                                                            \
    if (NA_SIZE(x_nary) != NA_SIZE(y_nary)) {                                    \
      rb_raise(rb_eArgError, "x and y must have same size");                     \
      return Qnil;                                                               \
    }                                                                            \
                                                                                 \
    ndfunc_arg_in_t ain[2] = { { tNArrType, 1 }, { tNArrType, 1 } };             \
    size_t shape_out[1] = { 1 };                                                 \
    ndfunc_arg_out_t aout[1] = { { tNArrType, 0, shape_out } };                  \
    ndfunc_t ndf = { _iter_##fBlasFnc, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout }; \
    VALUE ret = na_ndloop(&ndf, 2, x, y);                                        \
                                                                                 \
    RB_GC_GUARD(x);                                                              \
    RB_GC_GUARD(y);                                                              \
    return ret;                                                                  \
  }

DEF_LINALG_FUNC(double, numo_cDComplex, zdotu_sub)
DEF_LINALG_FUNC(float, numo_cSComplex, cdotu_sub)

#undef DEF_LINALG_FUNC

void define_linalg_blas_dot_sub(VALUE mBlas) {
  rb_define_module_function(mBlas, "zdotu", RUBY_METHOD_FUNC(_linalg_blas_zdotu_sub), 2);
  rb_define_module_function(mBlas, "cdotu", RUBY_METHOD_FUNC(_linalg_blas_cdotu_sub), 2);
}
