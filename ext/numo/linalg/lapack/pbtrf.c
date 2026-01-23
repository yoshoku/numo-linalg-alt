#include "pbtrf.h"

struct _pbtrf_option {
  int matrix_layout;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* ab = (tDType*)NDL_PTR(lp, 0);                                                      \
    int* info = (int*)NDL_PTR(lp, 1);                                                          \
    struct _pbtrf_option* opt = (struct _pbtrf_option*)(lp->opt_ptr);                          \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                      \
    const lapack_int kd = (lapack_int)NDL_SHAPE(lp, 0)[0] - 1;                                 \
    const lapack_int ldab = (lapack_int)NDL_SHAPE(lp, 0)[1];                                   \
    const lapack_int i =                                                                       \
      LAPACKE_##fLapackFunc(opt->matrix_layout, opt->uplo, n, kd, ab, ldab);                   \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE ab_vnary = Qnil;                                                                     \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &ab_vnary, &kw_args);                                       \
    ID kw_table[2] = { rb_intern("order"), rb_intern("uplo") };                                \
    VALUE kw_values[2] = { Qundef, Qundef };                                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char uplo = kw_values[1] != Qundef ? get_uplo(kw_values[1]) : 'U';                   \
                                                                                               \
    if (CLASS_OF(ab_vnary) != tNAryClass) {                                                    \
      ab_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, ab_vnary);                       \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(ab_vnary))) {                                             \
      ab_vnary = nary_dup(ab_vnary);                                                           \
    }                                                                                          \
                                                                                               \
    narray_t* ab_nary = NULL;                                                                  \
    GetNArray(ab_vnary, ab_nary);                                                              \
    if (NA_NDIM(ab_nary) != 2) {                                                               \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };            \
    struct _pbtrf_option opt = { matrix_layout, uplo };                                        \
    VALUE res = na_ndloop3(&ndf, &opt, 1, ab_vnary);                                           \
    VALUE ret = rb_ary_new3(2, ab_vnary, res);                                                 \
                                                                                               \
    RB_GC_GUARD(ab_vnary);                                                                     \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dpbtrf)
DEF_LINALG_FUNC(float, numo_cSFloat, spbtrf)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zpbtrf)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cpbtrf)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_pbtrf(VALUE mLapack) {
  rb_define_module_function(mLapack, "dpbtrf", _linalg_lapack_dpbtrf, -1);
  rb_define_module_function(mLapack, "spbtrf", _linalg_lapack_spbtrf, -1);
  rb_define_module_function(mLapack, "zpbtrf", _linalg_lapack_zpbtrf, -1);
  rb_define_module_function(mLapack, "cpbtrf", _linalg_lapack_cpbtrf, -1);
}
