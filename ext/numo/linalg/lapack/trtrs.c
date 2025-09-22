#include "trtrs.h"

struct _trtrs_option {
  int matrix_layout;
  char uplo;
  char trans;
  char diag;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFnc)                                                                       \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                                      \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                                                      \
    int* info = (int*)NDL_PTR(lp, 2);                                                                                         \
    struct _trtrs_option* opt = (struct _trtrs_option*)(lp->opt_ptr);                                                         \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                                     \
    const lapack_int nrhs = lp->args[1].ndim == 1 ? 1 : (lapack_int)NDL_SHAPE(lp, 1)[1];                                      \
    const lapack_int lda = n;                                                                                                 \
    const lapack_int ldb = nrhs;                                                                                              \
    const lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_layout, opt->uplo, opt->trans, opt->diag, n, nrhs, a, lda, b, ldb); \
    *info = (int)i;                                                                                                           \
  }                                                                                                                           \
                                                                                                                              \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                                               \
    VALUE a_vnary = Qnil;                                                                                                     \
    VALUE b_vnary = Qnil;                                                                                                     \
    VALUE kw_args = Qnil;                                                                                                     \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                                                             \
    ID kw_table[4] = { rb_intern("order"), rb_intern("uplo"), rb_intern("trans"), rb_intern("diag") };                        \
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };                                                                  \
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);                                                                        \
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;                    \
    const char uplo = kw_values[1] != Qundef ? get_uplo(kw_values[1]) : 'U';                                                  \
    const char trans = kw_values[2] != Qundef ? NUM2CHR(kw_values[2]) : 'N';                                                  \
    const char diag = kw_values[3] != Qundef ? NUM2CHR(kw_values[3]) : 'N';                                                   \
                                                                                                                              \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                                                    \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                                                        \
    }                                                                                                                         \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                                             \
      a_vnary = nary_dup(a_vnary);                                                                                            \
    }                                                                                                                         \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                                                    \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                                                        \
    }                                                                                                                         \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                                                             \
      b_vnary = nary_dup(b_vnary);                                                                                            \
    }                                                                                                                         \
                                                                                                                              \
    narray_t* a_nary = NULL;                                                                                                  \
    GetNArray(a_vnary, a_nary);                                                                                               \
    if (NA_NDIM(a_nary) != 2) {                                                                                               \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                                          \
      return Qnil;                                                                                                            \
    }                                                                                                                         \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                                                         \
      rb_raise(rb_eArgError, "input array a must be square");                                                                 \
      return Qnil;                                                                                                            \
    }                                                                                                                         \
                                                                                                                              \
    narray_t* b_nary = NULL;                                                                                                  \
    GetNArray(b_vnary, b_nary);                                                                                               \
    const int b_n_dims = NA_NDIM(b_nary);                                                                                     \
    if (b_n_dims != 1 && b_n_dims != 2) {                                                                                     \
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");                                                    \
      return Qnil;                                                                                                            \
    }                                                                                                                         \
                                                                                                                              \
    lapack_int n = (lapack_int)NA_SHAPE(a_nary)[0];                                                                           \
    lapack_int nb = (lapack_int)NA_SHAPE(b_nary)[0];                                                                          \
    if (n != nb) {                                                                                                            \
      rb_raise(nary_eShapeError, "shape1[0](=%d) != shape2[0](=%d)", n, nb);                                                  \
    }                                                                                                                         \
                                                                                                                              \
    ndfunc_arg_in_t ain[2] = { { tNAryClass, 2 }, { OVERWRITE, b_n_dims } };                                                  \
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };                                                                        \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };                                            \
    struct _trtrs_option opt = { matrix_layout, uplo, trans, diag };                                                          \
    VALUE info = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                                                 \
    VALUE ret = rb_ary_new3(2, b_vnary, info);                                                                                \
                                                                                                                              \
    RB_GC_GUARD(a_vnary);                                                                                                     \
    RB_GC_GUARD(b_vnary);                                                                                                     \
    return ret;                                                                                                               \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dtrtrs)
DEF_LINALG_FUNC(float, numo_cSFloat, strtrs)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, ztrtrs)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, ctrtrs)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_trtrs(VALUE mLapack) {
  rb_define_module_function(mLapack, "dtrtrs", RUBY_METHOD_FUNC(_linalg_lapack_dtrtrs), -1);
  rb_define_module_function(mLapack, "strtrs", RUBY_METHOD_FUNC(_linalg_lapack_strtrs), -1);
  rb_define_module_function(mLapack, "ztrtrs", RUBY_METHOD_FUNC(_linalg_lapack_ztrtrs), -1);
  rb_define_module_function(mLapack, "ctrtrs", RUBY_METHOD_FUNC(_linalg_lapack_ctrtrs), -1);
}
