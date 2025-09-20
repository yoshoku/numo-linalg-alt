#include "potrf.h"

struct _potrf_option {
  int matrix_layout;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryType, fLapackFnc)                                                     \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    int* info = (int*)NDL_PTR(lp, 1);                                                                      \
    struct _potrf_option* opt = (struct _potrf_option*)(lp->opt_ptr);                                      \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int lda = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                \
    const lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_layout, opt->uplo, n, a, lda);                   \
    *info = (int)i;                                                                                        \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                            \
    VALUE a_vnary = Qnil;                                                                                  \
    VALUE kw_args = Qnil;                                                                                  \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                    \
    ID kw_table[2] = { rb_intern("order"), rb_intern("uplo") };                                            \
    VALUE kw_values[2] = { Qundef, Qundef };                                                               \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                                     \
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR; \
    const char uplo = kw_values[1] != Qundef ? get_uplo(kw_values[1]) : 'U';                               \
                                                                                                           \
    if (CLASS_OF(a_vnary) != tNAryType) {                                                                  \
      a_vnary = rb_funcall(tNAryType, rb_intern("cast"), 1, a_vnary);                                      \
    }                                                                                                      \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                          \
      a_vnary = nary_dup(a_vnary);                                                                         \
    }                                                                                                      \
                                                                                                           \
    narray_t* a_nary = NULL;                                                                               \
    GetNArray(a_vnary, a_nary);                                                                            \
    if (NA_NDIM(a_nary) != 2) {                                                                            \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                       \
      return Qnil;                                                                                         \
    }                                                                                                      \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                                      \
      rb_raise(rb_eArgError, "input array a must be square");                                              \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                                         \
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };                                                     \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };                         \
    struct _potrf_option opt = { matrix_layout, uplo };                                                    \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                        \
    VALUE ret = rb_ary_new3(2, a_vnary, res);                                                              \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dpotrf)
DEF_LINALG_FUNC(float, numo_cSFloat, spotrf)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zpotrf)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cpotrf)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_potrf(VALUE mLapack) {
  rb_define_module_function(mLapack, "dpotrf", RUBY_METHOD_FUNC(_linalg_lapack_dpotrf), -1);
  rb_define_module_function(mLapack, "spotrf", RUBY_METHOD_FUNC(_linalg_lapack_spotrf), -1);
  rb_define_module_function(mLapack, "zpotrf", RUBY_METHOD_FUNC(_linalg_lapack_zpotrf), -1);
  rb_define_module_function(mLapack, "cpotrf", RUBY_METHOD_FUNC(_linalg_lapack_cpotrf), -1);
}
