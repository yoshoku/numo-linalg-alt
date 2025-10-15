#include "nrm2.h"

#define DEF_LINALG_FUNC(tDType, tRtDType, tNAryClass, tRtNAryClass, fBlasFunc)                 \
  static void _iter_##fBlasFunc(na_loop_t* const lp) {                                         \
    tDType* x = (tDType*)NDL_PTR(lp, 0);                                                       \
    tRtDType* d = (tRtDType*)NDL_PTR(lp, 1);                                                   \
    const blasint n = (blasint)NDL_SHAPE(lp, 0)[0];                                            \
    tRtDType ret = cblas_##fBlasFunc(n, x, 1);                                                 \
    *d = ret;                                                                                  \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_blas_##fBlasFunc(int argc, VALUE* argv, VALUE self) {                   \
    VALUE x = Qnil;                                                                            \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &x, &kw_args);                                              \
                                                                                               \
    ID kw_table[1] = { rb_intern("keepdims") };                                                \
    VALUE kw_values[1] = { Qundef };                                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);                                         \
    const bool keepdims = kw_values[0] != Qundef ? RTEST(kw_values[0]) : false;                \
                                                                                               \
    if (CLASS_OF(x) != tNAryClass) {                                                           \
      x = rb_funcall(tNAryClass, rb_intern("cast"), 1, x);                                     \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(x))) {                                                    \
      x = nary_dup(x);                                                                         \
    }                                                                                          \
                                                                                               \
    narray_t* x_nary = NULL;                                                                   \
    GetNArray(x, x_nary);                                                                      \
                                                                                               \
    if (NA_NDIM(x_nary) != 1) {                                                                \
      rb_raise(rb_eArgError, "x must be 1-dimensional");                                       \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SIZE(x_nary) == 0) {                                                                \
      rb_raise(rb_eArgError, "x must not be empty");                                           \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    ndfunc_arg_in_t ain[1] = { { tNAryClass, 1 } };                                            \
    size_t shape_out[1] = { 1 };                                                               \
    ndfunc_arg_out_t aout[1] = { { tRtNAryClass, 0, shape_out } };                             \
    ndfunc_t ndf = { _iter_##fBlasFunc, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };              \
    if (keepdims) {                                                                            \
      ndf.flag |= NDF_KEEP_DIM;                                                                \
    }                                                                                          \
                                                                                               \
    VALUE ret = na_ndloop(&ndf, 1, x);                                                         \
                                                                                               \
    RB_GC_GUARD(x);                                                                            \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, double, numo_cDFloat, numo_cDFloat, dnrm2)
DEF_LINALG_FUNC(float, float, numo_cSFloat, numo_cSFloat, snrm2)
DEF_LINALG_FUNC(dcomplex, double, numo_cDComplex, numo_cDFloat, dznrm2)
DEF_LINALG_FUNC(scomplex, float, numo_cSComplex, numo_cSFloat, scnrm2)

#undef DEF_LINALG_FUNC

void define_linalg_blas_nrm2(VALUE mBlas) {
  rb_define_module_function(mBlas, "dnrm2", RUBY_METHOD_FUNC(_linalg_blas_dnrm2), -1);
  rb_define_module_function(mBlas, "snrm2", RUBY_METHOD_FUNC(_linalg_blas_snrm2), -1);
  rb_define_module_function(mBlas, "dznrm2", RUBY_METHOD_FUNC(_linalg_blas_dznrm2), -1);
  rb_define_module_function(mBlas, "scnrm2", RUBY_METHOD_FUNC(_linalg_blas_scnrm2), -1);
}
