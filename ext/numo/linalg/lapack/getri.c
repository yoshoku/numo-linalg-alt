#include "getri.h"

struct _getri_option {
  int matrix_layout;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                                   \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                                   \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    lapack_int* ipiv = (lapack_int*)NDL_PTR(lp, 1);                                                        \
    int* info = (int*)NDL_PTR(lp, 2);                                                                      \
    struct _getri_option* opt = (struct _getri_option*)(lp->opt_ptr);                                      \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int lda = n;                                                                              \
    const lapack_int i = LAPACKE_##fLapackFunc(opt->matrix_layout, n, a, lda, ipiv);                       \
    *info = (int)i;                                                                                        \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {                           \
    VALUE a_vnary = Qnil;                                                                                  \
    VALUE ipiv_vnary = Qnil;                                                                               \
    VALUE kw_args = Qnil;                                                                                  \
    rb_scan_args(argc, argv, "2:", &a_vnary, &ipiv_vnary, &kw_args);                                       \
    ID kw_table[1] = { rb_intern("order") };                                                               \
    VALUE kw_values[1] = { Qundef };                                                                       \
    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);                                                     \
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR; \
                                                                                                           \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                                 \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                                     \
    }                                                                                                      \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                          \
      a_vnary = nary_dup(a_vnary);                                                                         \
    }                                                                                                      \
    if (CLASS_OF(ipiv_vnary) != numo_cInt32) {                                                             \
      ipiv_vnary = rb_funcall(numo_cInt32, rb_intern("cast"), 1, ipiv_vnary);                              \
    }                                                                                                      \
    if (!RTEST(nary_check_contiguous(ipiv_vnary))) {                                                       \
      ipiv_vnary = nary_dup(ipiv_vnary);                                                                   \
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
    narray_t* ipiv_nary = NULL;                                                                            \
    GetNArray(ipiv_vnary, ipiv_nary);                                                                      \
    if (NA_NDIM(ipiv_nary) != 1) {                                                                         \
      rb_raise(rb_eArgError, "input array ipiv must be 1-dimensional");                                    \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { numo_cInt32, 1 } };                                     \
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };                                                     \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };                        \
    struct _getri_option opt = { matrix_layout };                                                          \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, ipiv_vnary);                                            \
                                                                                                           \
    VALUE ret = rb_ary_new3(2, a_vnary, res);                                                              \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
    RB_GC_GUARD(ipiv_vnary);                                                                               \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgetri)
DEF_LINALG_FUNC(float, numo_cSFloat, sgetri)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgetri)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgetri)

void define_linalg_lapack_getri(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgetri", RUBY_METHOD_FUNC(_linalg_lapack_dgetri), -1);
  rb_define_module_function(mLapack, "sgetri", RUBY_METHOD_FUNC(_linalg_lapack_sgetri), -1);
  rb_define_module_function(mLapack, "zgetri", RUBY_METHOD_FUNC(_linalg_lapack_zgetri), -1);
  rb_define_module_function(mLapack, "cgetri", RUBY_METHOD_FUNC(_linalg_lapack_cgetri), -1);
}
