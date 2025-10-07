#include "gerqf.h"

struct _gerqf_option {
  int matrix_layout;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                                   \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                                   \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    tDType* tau = (tDType*)NDL_PTR(lp, 1);                                                                 \
    int* info = (int*)NDL_PTR(lp, 2);                                                                      \
    struct _gerqf_option* opt = (struct _gerqf_option*)(lp->opt_ptr);                                      \
    const lapack_int m = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                  \
    const lapack_int lda = n;                                                                              \
    const lapack_int i = LAPACKE_##fLapackFunc(opt->matrix_layout, m, n, a, lda, tau);                     \
    *info = (int)i;                                                                                        \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {                           \
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
    narray_t* a_nary;                                                                                      \
    GetNArray(a_vnary, a_nary);                                                                            \
    if (NA_NDIM(a_nary) != 2) {                                                                            \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                       \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    const size_t m = NA_SHAPE(a_nary)[0];                                                                  \
    const size_t n = NA_SHAPE(a_nary)[1];                                                                  \
    size_t shape[1] = { m < n ? m : n };                                                                   \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                                         \
    ndfunc_arg_out_t aout[2] = { { tNAryClass, 1, shape }, { numo_cInt32, 0 } };                           \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };                        \
    struct _gerqf_option opt = { matrix_layout };                                                          \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                        \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgerqf)
DEF_LINALG_FUNC(float, numo_cSFloat, sgerqf)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgerqf)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgerqf)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gerqf(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgerqf", RUBY_METHOD_FUNC(_linalg_lapack_dgerqf), -1);
  rb_define_module_function(mLapack, "sgerqf", RUBY_METHOD_FUNC(_linalg_lapack_sgerqf), -1);
  rb_define_module_function(mLapack, "zgerqf", RUBY_METHOD_FUNC(_linalg_lapack_zgerqf), -1);
  rb_define_module_function(mLapack, "cgerqf", RUBY_METHOD_FUNC(_linalg_lapack_cgerqf), -1);
}
