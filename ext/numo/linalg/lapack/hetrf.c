#include "hetrf.h"

struct _hetrf_option {
  int matrix_layout;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    lapack_int* ipiv = (lapack_int*)NDL_PTR(lp, 1);                                            \
    int* info = (int*)NDL_PTR(lp, 2);                                                          \
    struct _hetrf_option* opt = (struct _hetrf_option*)(lp->opt_ptr);                          \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                      \
    const lapack_int lda = n;                                                                  \
    const lapack_int i =                                                                       \
      LAPACKE_##fLapackFunc(opt->matrix_layout, opt->uplo, n, a, lda, ipiv);                   \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_tables[2] = { rb_intern("matrix_layout"), rb_intern("uplo") };                       \
    VALUE kw_values[2] = { Qundef, Qundef };                                                   \
    rb_get_kwargs(kw_args, kw_tables, 0, 2, kw_values);                                        \
    const int matrix_layout = kw_values[0] != Qundef && kw_values[0] != Qnil                   \
                                ? get_matrix_layout(kw_values[0])                              \
                                : LAPACK_ROW_MAJOR;                                            \
    const char uplo =                                                                          \
      kw_values[1] != Qundef && kw_values[1] != Qnil ? get_uplo(kw_values[1]) : 'U';           \
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
    if (NA_NDIM(a_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                          \
      rb_raise(rb_eArgError, "input array a must be square");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    const size_t n = NA_SHAPE(a_nary)[0];                                                      \
    size_t shape[1] = { n };                                                                   \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[2] = { { numo_cInt32, 1, shape }, { numo_cInt32, 0 } };              \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };            \
    struct _hetrf_option opt = { matrix_layout, uplo };                                        \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return res;                                                                                \
  }

DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zhetrf)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, chetrf)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_hetrf(VALUE mLapack) {
  rb_define_module_function(mLapack, "zhetrf", RUBY_METHOD_FUNC(_linalg_lapack_zhetrf), -1);
  rb_define_module_function(mLapack, "chetrf", RUBY_METHOD_FUNC(_linalg_lapack_chetrf), -1);
}
