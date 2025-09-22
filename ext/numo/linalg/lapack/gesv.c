#include "gesv.h"

struct _gesv_option {
  int matrix_layout;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                                   \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                                   \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                                   \
    int* ipiv = (int*)NDL_PTR(lp, 2);                                                                      \
    int* info = (int*)NDL_PTR(lp, 3);                                                                      \
    struct _gesv_option* opt = (struct _gesv_option*)(lp->opt_ptr);                                        \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int nhrs = lp->args[1].ndim == 1 ? 1 : (lapack_int)NDL_SHAPE(lp, 1)[1];                   \
    const lapack_int lda = n;                                                                              \
    const lapack_int ldb = nhrs;                                                                           \
    const lapack_int i = LAPACKE_##fLapackFunc(opt->matrix_layout, n, nhrs, a, lda, ipiv, b, ldb);         \
    *info = (int)i;                                                                                        \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {                           \
    VALUE a_vnary = Qnil;                                                                                  \
    VALUE b_vnary = Qnil;                                                                                  \
    VALUE kw_args = Qnil;                                                                                  \
                                                                                                           \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                                          \
                                                                                                           \
    ID kw_table[1] = { rb_intern("order") };                                                               \
    VALUE kw_values[1] = { Qundef };                                                                       \
                                                                                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);                                                     \
                                                                                                           \
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR; \
                                                                                                           \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                                 \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                                     \
    }                                                                                                      \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                          \
      a_vnary = nary_dup(a_vnary);                                                                         \
    }                                                                                                      \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                                 \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                                     \
    }                                                                                                      \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                                          \
      b_vnary = nary_dup(b_vnary);                                                                         \
    }                                                                                                      \
                                                                                                           \
    narray_t* a_nary = NULL;                                                                               \
    narray_t* b_nary = NULL;                                                                               \
    GetNArray(a_vnary, a_nary);                                                                            \
    GetNArray(b_vnary, b_nary);                                                                            \
    const int a_n_dims = NA_NDIM(a_nary);                                                                  \
    const int b_n_dims = NA_NDIM(b_nary);                                                                  \
    if (a_n_dims != 2) {                                                                                   \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                       \
      return Qnil;                                                                                         \
    }                                                                                                      \
    if (b_n_dims != 1 && b_n_dims != 2) {                                                                  \
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");                                 \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    lapack_int n = (lapack_int)NA_SHAPE(a_nary)[0];                                                        \
    lapack_int nb = (lapack_int)(b_n_dims == 1 ? NA_SHAPE(b_nary)[0] : NA_SHAPE(b_nary)[0]);               \
    if (n != nb) {                                                                                         \
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", n, nb);                               \
    }                                                                                                      \
                                                                                                           \
    lapack_int nhrs = b_n_dims == 1 ? 1 : (lapack_int)NA_SHAPE(b_nary)[1];                                 \
    size_t shape[2] = { (size_t)n, (size_t)nhrs };                                                         \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, b_n_dims } };                                \
    ndfunc_arg_out_t aout[2] = { { numo_cInt32, 1, shape }, { numo_cInt32, 0 } };                          \
                                                                                                           \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 2, ain, aout };                        \
    struct _gesv_option opt = { matrix_layout };                                                           \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                               \
                                                                                                           \
    VALUE ret = rb_ary_concat(rb_assoc_new(a_vnary, b_vnary), res);                                        \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
    RB_GC_GUARD(b_vnary);                                                                                  \
                                                                                                           \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgesv)
DEF_LINALG_FUNC(float, numo_cSFloat, sgesv)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zgesv)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, cgesv)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gesv(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgesv", RUBY_METHOD_FUNC(_linalg_lapack_dgesv), -1);
  rb_define_module_function(mLapack, "sgesv", RUBY_METHOD_FUNC(_linalg_lapack_sgesv), -1);
  rb_define_module_function(mLapack, "zgesv", RUBY_METHOD_FUNC(_linalg_lapack_zgesv), -1);
  rb_define_module_function(mLapack, "cgesv", RUBY_METHOD_FUNC(_linalg_lapack_cgesv), -1);
}
