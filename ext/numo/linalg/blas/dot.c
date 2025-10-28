#include "dot.h"

#define DEF_LINALG_FUNC(tDType, tNAryClass, fBlasFunc)                                         \
  static void _iter_##fBlasFunc(na_loop_t* const lp) {                                         \
    tDType* x = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* y = (tDType*)NDL_PTR(lp, 1);                                                       \
    tDType* d = (tDType*)NDL_PTR(lp, 2);                                                       \
    const CBLAS_INT n = (CBLAS_INT)NDL_SHAPE(lp, 0)[0];                                        \
    tDType ret = cblas_##fBlasFunc(n, x, 1, y, 1);                                             \
    *d = ret;                                                                                  \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_blas_##fBlasFunc(VALUE self, VALUE x, VALUE y) {                        \
    if (CLASS_OF(x) != tNAryClass) {                                                           \
      x = rb_funcall(tNAryClass, rb_intern("cast"), 1, x);                                     \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(x))) {                                                    \
      x = nary_dup(x);                                                                         \
    }                                                                                          \
    if (CLASS_OF(y) != tNAryClass) {                                                           \
      y = rb_funcall(tNAryClass, rb_intern("cast"), 1, y);                                     \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(y))) {                                                    \
      y = nary_dup(y);                                                                         \
    }                                                                                          \
                                                                                               \
    narray_t* x_nary = NULL;                                                                   \
    GetNArray(x, x_nary);                                                                      \
    narray_t* y_nary = NULL;                                                                   \
    GetNArray(y, y_nary);                                                                      \
                                                                                               \
    if (NA_NDIM(x_nary) != 1) {                                                                \
      rb_raise(rb_eArgError, "x must be 1-dimensional");                                       \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_NDIM(y_nary) != 1) {                                                                \
      rb_raise(rb_eArgError, "y must be 1-dimensional");                                       \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SIZE(x_nary) == 0) {                                                                \
      rb_raise(rb_eArgError, "x must not be empty");                                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SIZE(y_nary) == 0) {                                                                \
      rb_raise(rb_eArgError, "x must not be empty");                                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SIZE(x_nary) != NA_SIZE(y_nary)) {                                                  \
      rb_raise(rb_eArgError, "x and y must have same size");                                   \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    ndfunc_arg_in_t ain[2] = { { tNAryClass, 1 }, { tNAryClass, 1 } };                         \
    size_t shape_out[1] = { 1 };                                                               \
    ndfunc_arg_out_t aout[1] = { { tNAryClass, 0, shape_out } };                               \
    ndfunc_t ndf = { _iter_##fBlasFunc, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };              \
    VALUE ret = na_ndloop(&ndf, 2, x, y);                                                      \
                                                                                               \
    RB_GC_GUARD(x);                                                                            \
    RB_GC_GUARD(y);                                                                            \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, ddot)
DEF_LINALG_FUNC(float, numo_cSFloat, sdot)

#undef DEF_LINALG_FUNC

void define_linalg_blas_dot(VALUE mBlas) {
  rb_define_module_function(mBlas, "ddot", _linalg_blas_ddot, 2);
  rb_define_module_function(mBlas, "sdot", _linalg_blas_sdot, 2);
}
