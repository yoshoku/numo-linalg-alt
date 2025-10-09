#include "potrs.h"

struct _potrs_option {
  int matrix_layout;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                 \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                 \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                 \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                 \
    int* info = (int*)NDL_PTR(lp, 2);                                                    \
    struct _potrs_option* opt = (struct _potrs_option*)(lp->opt_ptr);                    \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                \
    const lapack_int nrhs = lp->args[1].ndim == 1 ? 1 : (lapack_int)NDL_SHAPE(lp, 1)[1]; \
    const lapack_int lda = n;                                                            \
    const lapack_int ldb = nrhs;                                                         \
    const lapack_int i =                                                                 \
      LAPACKE_##fLapackFunc(opt->matrix_layout, opt->uplo, n, nrhs, a, lda, b, ldb);     \
    *info = (int)i;                                                                      \
  }                                                                                      \
                                                                                         \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {         \
    VALUE a_vnary = Qnil;                                                                \
    VALUE b_vnary = Qnil;                                                                \
    VALUE kw_args = Qnil;                                                                \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                        \
    ID kw_table[2] = { rb_intern("order"), rb_intern("uplo") };                          \
    VALUE kw_values[2] = { Qundef, Qundef };                                             \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                   \
    const int matrix_layout =                                                            \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;       \
    const char uplo = kw_values[1] != Qundef ? get_uplo(kw_values[1]) : 'U';             \
                                                                                         \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                               \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                   \
    }                                                                                    \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                        \
      a_vnary = nary_dup(a_vnary);                                                       \
    }                                                                                    \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                               \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                   \
    }                                                                                    \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                        \
      b_vnary = nary_dup(b_vnary);                                                       \
    }                                                                                    \
                                                                                         \
    narray_t* a_nary = NULL;                                                             \
    GetNArray(a_vnary, a_nary);                                                          \
    if (NA_NDIM(a_nary) != 2) {                                                          \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                     \
      return Qnil;                                                                       \
    }                                                                                    \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                    \
      rb_raise(rb_eArgError, "input array a must be square");                            \
      return Qnil;                                                                       \
    }                                                                                    \
    narray_t* b_nary = NULL;                                                             \
    GetNArray(b_vnary, b_nary);                                                          \
    const int b_n_dims = NA_NDIM(b_nary);                                                \
    if (b_n_dims != 1 && b_n_dims != 2) {                                                \
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");               \
      return Qnil;                                                                       \
    }                                                                                    \
                                                                                         \
    lapack_int n = (lapack_int)NA_SHAPE(a_nary)[0];                                      \
    lapack_int nb = (lapack_int)NA_SHAPE(b_nary)[0];                                     \
    if (n != nb) {                                                                       \
      rb_raise(nary_eShapeError, "shape1[0](=%d) != shape2[0](=%d)", n, nb);             \
    }                                                                                    \
                                                                                         \
    ndfunc_arg_in_t ain[2] = { { tNAryClass, 2 }, { OVERWRITE, b_n_dims } };             \
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };                                   \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };      \
    struct _potrs_option opt = { matrix_layout, uplo };                                  \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                             \
    VALUE ret = rb_ary_new3(2, b_vnary, res);                                            \
                                                                                         \
    RB_GC_GUARD(a_vnary);                                                                \
    RB_GC_GUARD(b_vnary);                                                                \
    return ret;                                                                          \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dpotrs)
DEF_LINALG_FUNC(float, numo_cSFloat, spotrs)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zpotrs)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cpotrs)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_potrs(VALUE mLapack) {
  rb_define_module_function(mLapack, "dpotrs", RUBY_METHOD_FUNC(_linalg_lapack_dpotrs), -1);
  rb_define_module_function(mLapack, "spotrs", RUBY_METHOD_FUNC(_linalg_lapack_spotrs), -1);
  rb_define_module_function(mLapack, "zpotrs", RUBY_METHOD_FUNC(_linalg_lapack_zpotrs), -1);
  rb_define_module_function(mLapack, "cpotrs", RUBY_METHOD_FUNC(_linalg_lapack_cpotrs), -1);
}
