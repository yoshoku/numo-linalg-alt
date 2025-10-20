#include "gehrd.h"

struct _gehrd_option {
  int matrix_layout;
  int ilo;
  int ihi;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* tau = (tDType*)NDL_PTR(lp, 1);                                                     \
    int* info = (int*)NDL_PTR(lp, 2);                                                          \
    struct _gehrd_option* opt = (struct _gehrd_option*)(lp->opt_ptr);                          \
    const lapack_int ilo = opt->ilo;                                                           \
    const lapack_int ihi = opt->ihi;                                                           \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    lapack_int i = LAPACKE_##fLapackFunc(opt->matrix_layout, n, ilo, ihi, a, lda, tau);        \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("ilo"), rb_intern("ihi"), rb_intern("order") };               \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 2, 1, kw_values);                                         \
    const int ilo = NUM2INT(kw_values[0]);                                                     \
    const int ihi = NUM2INT(kw_values[1]);                                                     \
    const int matrix_layout =                                                                  \
      kw_values[2] != Qundef ? get_matrix_layout(kw_values[2]) : LAPACK_ROW_MAJOR;             \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    const int n_dims = NA_NDIM(a_nary);                                                        \
    if (n_dims != 2) {                                                                         \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t n = matrix_layout == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];  \
    size_t shape_tau[1] = { n - 1 };                                                           \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[2] = { { tNAryClass, 1, shape_tau }, { numo_cInt32, 0 } };           \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };            \
    struct _gehrd_option opt = { matrix_layout, ilo, ihi };                                    \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);                                   \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgehrd)
DEF_LINALG_FUNC(float, numo_cSFloat, sgehrd)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgehrd)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgehrd)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gehrd(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgehrd", _linalg_lapack_dgehrd, -1);
  rb_define_module_function(mLapack, "sgehrd", _linalg_lapack_sgehrd, -1);
  rb_define_module_function(mLapack, "zgehrd", _linalg_lapack_zgehrd, -1);
  rb_define_module_function(mLapack, "cgehrd", _linalg_lapack_cgehrd, -1);
}
