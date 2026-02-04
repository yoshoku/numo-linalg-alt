#include "gbsv.h"

struct _gbsv_option {
  int matrix_layout;
  int kl;
  int ku;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* ab = (tDType*)NDL_PTR(lp, 0);                                                      \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                       \
    int* ipiv = (int*)NDL_PTR(lp, 2);                                                          \
    int* info = (int*)NDL_PTR(lp, 3);                                                          \
    struct _gbsv_option* opt = (struct _gbsv_option*)(lp->opt_ptr);                            \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 1)[0];                                      \
    const lapack_int nhrs = lp->args[1].ndim == 1 ? 1 : (lapack_int)NDL_SHAPE(lp, 1)[1];       \
    const lapack_int ldab = 2 * opt->kl + opt->ku + 1;                                         \
    const lapack_int ldb = nhrs;                                                               \
    const lapack_int i = LAPACKE_##fLapackFunc(                                                \
      opt->matrix_layout, n, opt->kl, opt->ku, nhrs, ab, ldab, ipiv, b, ldb                    \
    );                                                                                         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE ab_vnary = Qnil;                                                                     \
    VALUE b_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &ab_vnary, &b_vnary, &kw_args);                             \
    ID kw_table[3] = { rb_intern("kl"), rb_intern("ku"), rb_intern("order") };                 \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 2, 1, kw_values);                                         \
    const int kl = NUM2INT(kw_values[0]);                                                      \
    const int ku = NUM2INT(kw_values[1]);                                                      \
    const int matrix_layout =                                                                  \
      kw_values[2] != Qundef ? get_matrix_layout(kw_values[2]) : LAPACK_ROW_MAJOR;             \
                                                                                               \
    if (CLASS_OF(ab_vnary) != tNAryClass) {                                                    \
      ab_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, ab_vnary);                       \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(ab_vnary))) {                                             \
      ab_vnary = nary_dup(ab_vnary);                                                           \
    }                                                                                          \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                     \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                              \
      b_vnary = nary_dup(b_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* ab_nary = NULL;                                                                  \
    narray_t* b_nary = NULL;                                                                   \
    GetNArray(ab_vnary, ab_nary);                                                              \
    GetNArray(b_vnary, b_nary);                                                                \
    const int ab_n_dims = NA_NDIM(ab_nary);                                                    \
    const int b_n_dims = NA_NDIM(b_nary);                                                      \
    if (ab_n_dims != 2) {                                                                      \
      rb_raise(rb_eArgError, "input array ab must be 2-dimensional");                          \
      return Qnil;                                                                             \
    }                                                                                          \
    if (b_n_dims != 1 && b_n_dims != 2) {                                                      \
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");                     \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    const size_t n = NA_SHAPE(b_nary)[0];                                                      \
    size_t shape[1] = { n };                                                                   \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, b_n_dims } };                    \
    ndfunc_arg_out_t aout[2] = { { numo_cInt32, 1, shape }, { numo_cInt32, 0 } };              \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 2, ain, aout };            \
    struct _gbsv_option opt = { matrix_layout, kl, ku };                                       \
    VALUE res = na_ndloop3(&ndf, &opt, 2, ab_vnary, b_vnary);                                  \
    VALUE ret = rb_ary_concat(rb_assoc_new(ab_vnary, b_vnary), res);                           \
                                                                                               \
    RB_GC_GUARD(ab_vnary);                                                                     \
    RB_GC_GUARD(b_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgbsv)
DEF_LINALG_FUNC(float, numo_cSFloat, sgbsv)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgbsv)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgbsv)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gbsv(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgbsv", _linalg_lapack_dgbsv, -1);
  rb_define_module_function(mLapack, "sgbsv", _linalg_lapack_sgbsv, -1);
  rb_define_module_function(mLapack, "zgbsv", _linalg_lapack_zgbsv, -1);
  rb_define_module_function(mLapack, "cgbsv", _linalg_lapack_cgbsv, -1);
}
