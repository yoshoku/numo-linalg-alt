#include "gemv.h"

#define DEF_LINALG_OPTIONS(tElType) \
  struct _gemv_options_##tElType {  \
    tElType alpha;                  \
    tElType beta;                   \
    enum CBLAS_ORDER order;         \
    enum CBLAS_TRANSPOSE trans;     \
    blasint m;                      \
    blasint n;                      \
  };

#define DEF_LINALG_ITER_FUNC(tElType, fBlasFnc)                                                 \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                                           \
    const tElType* a = (tElType*)NDL_PTR(lp, 0);                                                \
    const tElType* x = (tElType*)NDL_PTR(lp, 1);                                                \
    tElType* y = (tElType*)NDL_PTR(lp, 2);                                                      \
    const struct _gemv_options_##tElType* opt = (struct _gemv_options_##tElType*)(lp->opt_ptr); \
    const blasint lda = opt->n;                                                                 \
    cblas_##fBlasFnc(opt->order, opt->trans, opt->m, opt->n,                                    \
                     opt->alpha, a, lda, x, 1, opt->beta, y, 1);                                \
  }

#define DEF_LINALG_ITER_FUNC_COMPLEX(tElType, fBlasFnc)                                         \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                                           \
    const tElType* a = (tElType*)NDL_PTR(lp, 0);                                                \
    const tElType* x = (tElType*)NDL_PTR(lp, 1);                                                \
    tElType* y = (tElType*)NDL_PTR(lp, 2);                                                      \
    const struct _gemv_options_##tElType* opt = (struct _gemv_options_##tElType*)(lp->opt_ptr); \
    const blasint lda = opt->n;                                                                 \
    cblas_##fBlasFnc(opt->order, opt->trans, opt->m, opt->n,                                    \
                     &opt->alpha, a, lda, x, 1, &opt->beta, y, 1);                              \
  }

#define DEF_LINALG_FUNC(tElType, tNArrType, fBlasFnc)                                                   \
  static VALUE _linalg_blas_##fBlasFnc(int argc, VALUE* argv, VALUE self) {                             \
    VALUE a = Qnil;                                                                                     \
    VALUE x = Qnil;                                                                                     \
    VALUE y = Qnil;                                                                                     \
    VALUE kw_args = Qnil;                                                                               \
    rb_scan_args(argc, argv, "21:", &a, &x, &y, &kw_args);                                              \
                                                                                                        \
    ID kw_table[4] = { rb_intern("alpha"), rb_intern("beta"),                                           \
                       rb_intern("order"), rb_intern("trans") };                                        \
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };                                            \
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);                                                  \
                                                                                                        \
    if (CLASS_OF(a) != tNArrType) {                                                                     \
      a = rb_funcall(tNArrType, rb_intern("cast"), 1, a);                                               \
    }                                                                                                   \
    if (!RTEST(nary_check_contiguous(a))) {                                                             \
      a = nary_dup(a);                                                                                  \
    }                                                                                                   \
    if (CLASS_OF(x) != tNArrType) {                                                                     \
      x = rb_funcall(tNArrType, rb_intern("cast"), 1, x);                                               \
    }                                                                                                   \
    if (!RTEST(nary_check_contiguous(x))) {                                                             \
      x = nary_dup(x);                                                                                  \
    }                                                                                                   \
    if (!NIL_P(y)) {                                                                                    \
      if (CLASS_OF(y) != tNArrType) {                                                                   \
        y = rb_funcall(tNArrType, rb_intern("cast"), 1, y);                                             \
      }                                                                                                 \
      if (!RTEST(nary_check_contiguous(y))) {                                                           \
        y = nary_dup(y);                                                                                \
      }                                                                                                 \
    }                                                                                                   \
                                                                                                        \
    tElType alpha = kw_values[0] != Qundef ? conv_##tElType(kw_values[0]) : one_##tElType();            \
    tElType beta = kw_values[1] != Qundef ? conv_##tElType(kw_values[1]) : zero_##tElType();            \
    enum CBLAS_ORDER order = kw_values[2] != Qundef ? get_cblas_order(kw_values[2]) : CblasRowMajor;    \
    enum CBLAS_TRANSPOSE trans = kw_values[3] != Qundef ? get_cblas_trans(kw_values[3]) : CblasNoTrans; \
                                                                                                        \
    narray_t* a_nary = NULL;                                                                            \
    GetNArray(a, a_nary);                                                                               \
    narray_t* x_nary = NULL;                                                                            \
    GetNArray(x, x_nary);                                                                               \
                                                                                                        \
    if (NA_NDIM(a_nary) != 2) {                                                                         \
      rb_raise(rb_eArgError, "a must be 2-dimensional");                                                \
      return Qnil;                                                                                      \
    }                                                                                                   \
    if (NA_NDIM(x_nary) != 1) {                                                                         \
      rb_raise(rb_eArgError, "x must be 1-dimensional");                                                \
      return Qnil;                                                                                      \
    }                                                                                                   \
    if (NA_SIZE(a_nary) == 0) {                                                                         \
      rb_raise(rb_eArgError, "a must not be empty");                                                    \
      return Qnil;                                                                                      \
    }                                                                                                   \
    if (NA_SIZE(x_nary) == 0) {                                                                         \
      rb_raise(rb_eArgError, "x must not be empty");                                                    \
      return Qnil;                                                                                      \
    }                                                                                                   \
                                                                                                        \
    const blasint ma = NA_SHAPE(a_nary)[0];                                                             \
    const blasint na = NA_SHAPE(a_nary)[1];                                                             \
    const blasint mx = NA_SHAPE(x_nary)[0];                                                             \
    const blasint m = trans == CblasNoTrans ? ma : na;                                                  \
    const blasint n = trans == CblasNoTrans ? na : ma;                                                  \
                                                                                                        \
    if (n != mx) {                                                                                      \
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", n, mx);                            \
      return Qnil;                                                                                      \
    }                                                                                                   \
                                                                                                        \
    struct _gemv_options_##tElType opt = { alpha, beta, order, trans, ma, na };                         \
    size_t shape_out[1] = { (size_t)(m) };                                                              \
    ndfunc_arg_out_t aout[1] = { { tNArrType, 1, shape_out } };                                         \
    VALUE ret = Qnil;                                                                                   \
                                                                                                        \
    if (!NIL_P(y)) {                                                                                    \
      narray_t* y_nary = NULL;                                                                          \
      GetNArray(y, y_nary);                                                                             \
      blasint my = NA_SHAPE(y_nary)[0];                                                                 \
      if (m > my) {                                                                                     \
        rb_raise(nary_eShapeError, "shape3[0](=%d) >= shape1[0]=%d", my, m);                            \
        return Qnil;                                                                                    \
      }                                                                                                 \
      ndfunc_arg_in_t ain[3] = { { tNArrType, 2 }, { tNArrType, 1 }, { OVERWRITE, 1 } };                \
      ndfunc_t ndf = { _iter_##fBlasFnc, NO_LOOP, 3, 0, ain, aout };                                    \
      na_ndloop3(&ndf, &opt, 3, a, x, y);                                                               \
      ret = y;                                                                                          \
    } else {                                                                                            \
      y = INT2NUM(0);                                                                                   \
      ndfunc_arg_in_t ain[3] = { { tNArrType, 2 }, { tNArrType, 1 }, { sym_init, 0 } };                 \
      ndfunc_t ndf = { _iter_##fBlasFnc, NO_LOOP, 3, 1, ain, aout };                                    \
      ret = na_ndloop3(&ndf, &opt, 3, a, x, y);                                                         \
    }                                                                                                   \
                                                                                                        \
    RB_GC_GUARD(a);                                                                                     \
    RB_GC_GUARD(x);                                                                                     \
    RB_GC_GUARD(y);                                                                                     \
                                                                                                        \
    return ret;                                                                                         \
  }

DEF_LINALG_OPTIONS(double)
DEF_LINALG_OPTIONS(float)
DEF_LINALG_OPTIONS(dcomplex)
DEF_LINALG_OPTIONS(scomplex)
DEF_LINALG_ITER_FUNC(double, dgemv)
DEF_LINALG_ITER_FUNC(float, sgemv)
DEF_LINALG_ITER_FUNC_COMPLEX(dcomplex, zgemv)
DEF_LINALG_ITER_FUNC_COMPLEX(scomplex, cgemv)
DEF_LINALG_FUNC(double, numo_cDFloat, dgemv)
DEF_LINALG_FUNC(float, numo_cSFloat, sgemv)
DEF_LINALG_FUNC(dcomplex, numo_cDComplex, zgemv)
DEF_LINALG_FUNC(scomplex, numo_cSComplex, cgemv)

#undef DEF_LINALG_OPTIONS
#undef DEF_LINALG_ITER_FUNC
#undef DEF_LINALG_FUNC

void define_linalg_blas_gemv(VALUE mBlas) {
  rb_define_module_function(mBlas, "dgemv", RUBY_METHOD_FUNC(_linalg_blas_dgemv), -1);
  rb_define_module_function(mBlas, "sgemv", RUBY_METHOD_FUNC(_linalg_blas_sgemv), -1);
  rb_define_module_function(mBlas, "zgemv", RUBY_METHOD_FUNC(_linalg_blas_zgemv), -1);
  rb_define_module_function(mBlas, "cgemv", RUBY_METHOD_FUNC(_linalg_blas_cgemv), -1);
}
