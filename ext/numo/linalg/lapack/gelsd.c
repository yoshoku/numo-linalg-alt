#include "gelsd.h"

struct _gelsd_option {
  int matrix_layout;
  double rcond;
};

#define DEF_LINALG_FUNC(tDType, tRtDType, tNAryClass, tRtNAryClass, fLapackFunc)               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                       \
    tRtDType* s = (tRtDType*)NDL_PTR(lp, 2);                                                   \
    int* rank = (int*)NDL_PTR(lp, 3);                                                          \
    int* info = (int*)NDL_PTR(lp, 4);                                                          \
    struct _gelsd_option* opt = (struct _gelsd_option*)(lp->opt_ptr);                          \
    const lapack_int m =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[1]                \
                                                          : NDL_SHAPE(lp, 0)[0]);              \
    const lapack_int nrhs = lp->args[1].ndim == 1 ? 1 : (lapack_int)NDL_SHAPE(lp, 1)[1];       \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldb = nrhs;                                                               \
    lapack_int r = 0;                                                                          \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, m, n, nrhs, a, lda, b, ldb, s, (tRtDType)(opt->rcond), &r            \
    );                                                                                         \
    *rank = (int)r;                                                                            \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE b_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                              \
    ID kw_table[2] = { rb_intern("matrix_layout"), rb_intern("rcond") };                       \
    VALUE kw_values[2] = { Qundef, Qundef };                                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                         \
    const int matrix_layout = kw_values[0] != Qundef && kw_values[0] != Qnil                   \
                                ? get_matrix_layout(kw_values[0])                              \
                                : LAPACK_ROW_MAJOR;                                            \
    const double rcond =                                                                       \
      kw_values[1] != Qundef && kw_values[1] != Qnil ? NUM2DBL(kw_values[1]) : -1.0;           \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                     \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                              \
      b_vnary = nary_dup(b_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    const int n_dims = NA_NDIM(a_nary);                                                        \
    if (n_dims != 2) {                                                                         \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    narray_t* b_nary = NULL;                                                                   \
    GetNArray(b_vnary, b_nary);                                                                \
    const int b_n_dims = NA_NDIM(b_nary);                                                      \
    if (b_n_dims != 1 && b_n_dims != 2) {                                                      \
      rb_raise(rb_eArgError, "input array b must be 1 or 2-dimensional");                      \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    const size_t m = NA_SHAPE(a_nary)[0];                                                      \
    const size_t n = NA_SHAPE(a_nary)[1];                                                      \
    size_t shape_s[1] = { m < n ? m : n };                                                     \
    ndfunc_arg_in_t ain[2] = { { tNAryClass, 2 }, { OVERWRITE, b_n_dims } };                   \
    ndfunc_arg_out_t aout[3] = { { tRtNAryClass, 1, shape_s },                                 \
                                 { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 3, ain, aout };            \
    struct _gelsd_option opt = { matrix_layout, rcond };                                       \
    VALUE ret = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                   \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    RB_GC_GUARD(b_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, double, numo_cDFloat, numo_cDFloat, dgelsd)
DEF_LINALG_FUNC(float, float, numo_cSFloat, numo_cSFloat, sgelsd)
DEF_LINALG_FUNC(lapack_complex_double, double, numo_cDComplex, numo_cDFloat, zgelsd)
DEF_LINALG_FUNC(lapack_complex_float, float, numo_cSComplex, numo_cSFloat, cgelsd)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gelsd(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgelsd", RUBY_METHOD_FUNC(_linalg_lapack_dgelsd), -1);
  rb_define_module_function(mLapack, "sgelsd", RUBY_METHOD_FUNC(_linalg_lapack_sgelsd), -1);
  rb_define_module_function(mLapack, "zgelsd", RUBY_METHOD_FUNC(_linalg_lapack_zgelsd), -1);
  rb_define_module_function(mLapack, "cgelsd", RUBY_METHOD_FUNC(_linalg_lapack_cgelsd), -1);
}
