#include "gemm.h"

#define DEF_LINALG_OPTIONS(tDType) \
  struct _gemm_options_##tDType {  \
    tDType alpha;                  \
    tDType beta;                   \
    enum CBLAS_ORDER order;        \
    enum CBLAS_TRANSPOSE transa;   \
    enum CBLAS_TRANSPOSE transb;   \
    blasint m;                     \
    blasint n;                     \
    blasint k;                     \
  };

#define DEF_LINALG_ITER_FUNC(tDType, fBlasFnc)                                                \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                                         \
    const tDType* a = (tDType*)NDL_PTR(lp, 0);                                                \
    const tDType* b = (tDType*)NDL_PTR(lp, 1);                                                \
    tDType* c = (tDType*)NDL_PTR(lp, 2);                                                      \
    const struct _gemm_options_##tDType* opt = (struct _gemm_options_##tDType*)(lp->opt_ptr); \
    const blasint lda = opt->transa == CblasNoTrans ? opt->k : opt->m;                        \
    const blasint ldb = opt->transb == CblasNoTrans ? opt->n : opt->k;                        \
    const blasint ldc = opt->n;                                                               \
    cblas_##fBlasFnc(opt->order, opt->transa, opt->transb, opt->m, opt->n, opt->k,            \
                     opt->alpha, a, lda, b, ldb, opt->beta, c, ldc);                          \
  }

#define DEF_LINALG_ITER_FUNC_COMPLEX(tDType, fBlasFnc)                                        \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                                         \
    const tDType* a = (tDType*)NDL_PTR(lp, 0);                                                \
    const tDType* b = (tDType*)NDL_PTR(lp, 1);                                                \
    tDType* c = (tDType*)NDL_PTR(lp, 2);                                                      \
    const struct _gemm_options_##tDType* opt = (struct _gemm_options_##tDType*)(lp->opt_ptr); \
    const blasint lda = opt->transa == CblasNoTrans ? opt->k : opt->m;                        \
    const blasint ldb = opt->transb == CblasNoTrans ? opt->n : opt->k;                        \
    const blasint ldc = opt->n;                                                               \
    cblas_##fBlasFnc(opt->order, opt->transa, opt->transb, opt->m, opt->n, opt->k,            \
                     &opt->alpha, a, lda, b, ldb, &opt->beta, c, ldc);                        \
  }

#define DEF_LINALG_ITER_FUNC(tDType, fBlasFnc)                                                \
  static void _iter_##fBlasFnc(na_loop_t* const lp) {                                         \
    const tDType* a = (tDType*)NDL_PTR(lp, 0);                                                \
    const tDType* b = (tDType*)NDL_PTR(lp, 1);                                                \
    tDType* c = (tDType*)NDL_PTR(lp, 2);                                                      \
    const struct _gemm_options_##tDType* opt = (struct _gemm_options_##tDType*)(lp->opt_ptr); \
    const blasint lda = opt->transa == CblasNoTrans ? opt->k : opt->m;                        \
    const blasint ldb = opt->transb == CblasNoTrans ? opt->n : opt->k;                        \
    const blasint ldc = opt->n;                                                               \
    cblas_##fBlasFnc(opt->order, opt->transa, opt->transb, opt->m, opt->n, opt->k,            \
                     opt->alpha, a, lda, b, ldb, opt->beta, c, ldc);                          \
  }

#define DEF_LINALG_FUNC(tDType, tNAryType, fBlasFnc)                                                     \
  static VALUE _linalg_blas_##fBlasFnc(int argc, VALUE* argv, VALUE self) {                              \
    VALUE a = Qnil;                                                                                      \
    VALUE b = Qnil;                                                                                      \
    VALUE c = Qnil;                                                                                      \
    VALUE kw_args = Qnil;                                                                                \
    rb_scan_args(argc, argv, "21:", &a, &b, &c, &kw_args);                                               \
                                                                                                         \
    ID kw_table[5] = { rb_intern("alpha"), rb_intern("beta"), rb_intern("order"),                        \
                       rb_intern("transa"), rb_intern("transb") };                                       \
    VALUE kw_values[5] = { Qundef, Qundef, Qundef, Qundef, Qundef };                                     \
    rb_get_kwargs(kw_args, kw_table, 0, 5, kw_values);                                                   \
                                                                                                         \
    if (CLASS_OF(a) != tNAryType) {                                                                      \
      a = rb_funcall(tNAryType, rb_intern("cast"), 1, a);                                                \
    }                                                                                                    \
    if (!RTEST(nary_check_contiguous(a))) {                                                              \
      a = nary_dup(a);                                                                                   \
    }                                                                                                    \
    if (CLASS_OF(b) != tNAryType) {                                                                      \
      b = rb_funcall(tNAryType, rb_intern("cast"), 1, b);                                                \
    }                                                                                                    \
    if (!RTEST(nary_check_contiguous(b))) {                                                              \
      b = nary_dup(b);                                                                                   \
    }                                                                                                    \
    if (!NIL_P(c)) {                                                                                     \
      if (CLASS_OF(c) != tNAryType) {                                                                    \
        c = rb_funcall(tNAryType, rb_intern("cast"), 1, c);                                              \
      }                                                                                                  \
      if (!RTEST(nary_check_contiguous(c))) {                                                            \
        c = nary_dup(c);                                                                                 \
      }                                                                                                  \
    }                                                                                                    \
                                                                                                         \
    tDType alpha = kw_values[0] != Qundef ? conv_##tDType(kw_values[0]) : one_##tDType();                \
    tDType beta = kw_values[1] != Qundef ? conv_##tDType(kw_values[1]) : zero_##tDType();                \
    enum CBLAS_ORDER order = kw_values[2] != Qundef ? get_cblas_order(kw_values[2]) : CblasRowMajor;     \
    enum CBLAS_TRANSPOSE transa = kw_values[3] != Qundef ? get_cblas_trans(kw_values[3]) : CblasNoTrans; \
    enum CBLAS_TRANSPOSE transb = kw_values[4] != Qundef ? get_cblas_trans(kw_values[4]) : CblasNoTrans; \
                                                                                                         \
    narray_t* a_nary = NULL;                                                                             \
    GetNArray(a, a_nary);                                                                                \
    narray_t* b_nary = NULL;                                                                             \
    GetNArray(b, b_nary);                                                                                \
                                                                                                         \
    if (NA_NDIM(a_nary) != 2) {                                                                          \
      rb_raise(rb_eArgError, "a must be 2-dimensional");                                                 \
      return Qnil;                                                                                       \
    }                                                                                                    \
    if (NA_NDIM(b_nary) != 2) {                                                                          \
      rb_raise(rb_eArgError, "b must be 2-dimensional");                                                 \
      return Qnil;                                                                                       \
    }                                                                                                    \
    if (NA_SIZE(a_nary) == 0) {                                                                          \
      rb_raise(rb_eArgError, "a must not be empty");                                                     \
      return Qnil;                                                                                       \
    }                                                                                                    \
    if (NA_SIZE(b_nary) == 0) {                                                                          \
      rb_raise(rb_eArgError, "b must not be empty");                                                     \
      return Qnil;                                                                                       \
    }                                                                                                    \
                                                                                                         \
    const blasint ma = (blasint)NA_SHAPE(a_nary)[0];                                                     \
    const blasint ka = (blasint)NA_SHAPE(a_nary)[1];                                                     \
    const blasint kb = (blasint)NA_SHAPE(b_nary)[0];                                                     \
    const blasint nb = (blasint)NA_SHAPE(b_nary)[1];                                                     \
    const blasint m = transa == CblasNoTrans ? ma : ka;                                                  \
    const blasint n = transb == CblasNoTrans ? nb : kb;                                                  \
    const blasint k = transa == CblasNoTrans ? ka : ma;                                                  \
    const blasint l = transb == CblasNoTrans ? kb : nb;                                                  \
                                                                                                         \
    if (k != l) {                                                                                        \
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", k, l);                              \
      return Qnil;                                                                                       \
    }                                                                                                    \
                                                                                                         \
    struct _gemm_options_##tDType opt = { alpha, beta, order, transa, transb, m, n, k };                 \
    size_t shape_out[2] = { (size_t)m, (size_t)n };                                                      \
    ndfunc_arg_out_t aout[1] = { { tNAryType, 2, shape_out } };                                          \
    VALUE ret = Qnil;                                                                                    \
                                                                                                         \
    if (!NIL_P(c)) {                                                                                     \
      narray_t* c_nary = NULL;                                                                           \
      GetNArray(c, c_nary);                                                                              \
      blasint nc = (blasint)NA_SHAPE(c_nary)[0];                                                         \
      if (m > nc) {                                                                                      \
        rb_raise(nary_eShapeError, "shape3[0](=%d) >= shape1[0]=%d", nc, m);                             \
        return Qnil;                                                                                     \
      }                                                                                                  \
      ndfunc_arg_in_t ain[3] = { { tNAryType, 2 }, { tNAryType, 2 }, { OVERWRITE, 2 } };                 \
      ndfunc_t ndf = { _iter_##fBlasFnc, NO_LOOP, 3, 0, ain, aout };                                     \
      na_ndloop3(&ndf, &opt, 3, a, b, c);                                                                \
      ret = c;                                                                                           \
    } else {                                                                                             \
      c = INT2NUM(0);                                                                                    \
      ndfunc_arg_in_t ain[3] = { { tNAryType, 2 }, { tNAryType, 2 }, { sym_init, 0 } };                  \
      ndfunc_t ndf = { _iter_##fBlasFnc, NO_LOOP, 3, 1, ain, aout };                                     \
      ret = na_ndloop3(&ndf, &opt, 3, a, b, c);                                                          \
    }                                                                                                    \
                                                                                                         \
    RB_GC_GUARD(a);                                                                                      \
    RB_GC_GUARD(b);                                                                                      \
    RB_GC_GUARD(c);                                                                                      \
                                                                                                         \
    return ret;                                                                                          \
  }

DEF_LINALG_OPTIONS(double)
DEF_LINALG_OPTIONS(float)
DEF_LINALG_OPTIONS(dcomplex)
DEF_LINALG_OPTIONS(scomplex)
DEF_LINALG_ITER_FUNC(double, dgemm)
DEF_LINALG_ITER_FUNC(float, sgemm)
DEF_LINALG_ITER_FUNC_COMPLEX(dcomplex, zgemm)
DEF_LINALG_ITER_FUNC_COMPLEX(scomplex, cgemm)
DEF_LINALG_FUNC(double, numo_cDFloat, dgemm)
DEF_LINALG_FUNC(float, numo_cSFloat, sgemm)
DEF_LINALG_FUNC(dcomplex, numo_cDComplex, zgemm)
DEF_LINALG_FUNC(scomplex, numo_cSComplex, cgemm)

#undef DEF_LINALG_OPTIONS
#undef DEF_LINALG_ITER_FUNC
#undef DEF_LINALG_ITER_FUNC_COMPLEX
#undef DEF_LINALG_FUNC

void define_linalg_blas_gemm(VALUE mBlas) {
  rb_define_module_function(mBlas, "dgemm", RUBY_METHOD_FUNC(_linalg_blas_dgemm), -1);
  rb_define_module_function(mBlas, "sgemm", RUBY_METHOD_FUNC(_linalg_blas_sgemm), -1);
  rb_define_module_function(mBlas, "zgemm", RUBY_METHOD_FUNC(_linalg_blas_zgemm), -1);
  rb_define_module_function(mBlas, "cgemm", RUBY_METHOD_FUNC(_linalg_blas_cgemm), -1);
}
