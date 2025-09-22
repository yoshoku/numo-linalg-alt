#include "geqrf.h"

struct _geqrf_option {
  int matrix_layout;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFnc)                                                    \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    tDType* tau = (tDType*)NDL_PTR(lp, 1);                                                                 \
    int* info = (int*)NDL_PTR(lp, 2);                                                                      \
    struct _geqrf_option* opt = (struct _geqrf_option*)(lp->opt_ptr);                                      \
    const lapack_int m = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                  \
    const lapack_int lda = n;                                                                              \
    const lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_layout, m, n, a, lda, tau);                      \
    *info = (int)i;                                                                                        \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                            \
    VALUE a_vnary = Qnil;                                                                                  \
    VALUE kw_args = Qnil;                                                                                  \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                    \
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
                                                                                                           \
    narray_t* a_nary = NULL;                                                                               \
    GetNArray(a_vnary, a_nary);                                                                            \
    const int n_dims = NA_NDIM(a_nary);                                                                    \
    if (n_dims != 2) {                                                                                     \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                       \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    size_t m = NA_SHAPE(a_nary)[0];                                                                        \
    size_t n = NA_SHAPE(a_nary)[1];                                                                        \
    size_t shape[1] = { m < n ? m : n };                                                                   \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                                         \
    ndfunc_arg_out_t aout[2] = { { tNAryClass, 1, shape }, { numo_cInt32, 0 } };                           \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };                         \
    struct _geqrf_option opt = { matrix_layout };                                                          \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                        \
                                                                                                           \
    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);                                               \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
                                                                                                           \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgeqrf)
DEF_LINALG_FUNC(float, numo_cSFloat, sgeqrf)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgeqrf)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgeqrf)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_geqrf(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgeqrf", RUBY_METHOD_FUNC(_linalg_lapack_dgeqrf), -1);
  rb_define_module_function(mLapack, "sgeqrf", RUBY_METHOD_FUNC(_linalg_lapack_sgeqrf), -1);
  rb_define_module_function(mLapack, "zgeqrf", RUBY_METHOD_FUNC(_linalg_lapack_zgeqrf), -1);
  rb_define_module_function(mLapack, "cgeqrf", RUBY_METHOD_FUNC(_linalg_lapack_cgeqrf), -1);
}
