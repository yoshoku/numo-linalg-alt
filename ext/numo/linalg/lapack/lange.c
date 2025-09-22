#include "lange.h"

struct _lange_option {
  int matrix_layout;
  char norm;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFnc)                                                    \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                   \
    tDType* d = (tDType*)NDL_PTR(lp, 1);                                                                   \
    struct _lange_option* opt = (struct _lange_option*)(lp->opt_ptr);                                      \
    const lapack_int m = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                  \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                  \
    const lapack_int lda = n;                                                                              \
    *d = LAPACKE_##fLapackFnc(opt->matrix_layout, opt->norm, m, n, a, lda);                                \
  }                                                                                                        \
                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                            \
    VALUE a_vnary = Qnil;                                                                                  \
    VALUE kw_args = Qnil;                                                                                  \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                    \
    ID kw_table[2] = { rb_intern("order"), rb_intern("norm") };                                            \
    VALUE kw_values[2] = { Qundef, Qundef };                                                               \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                                     \
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR; \
    const char norm = kw_values[1] != Qundef ? NUM2CHR(kw_values[1]) : 'F';                                \
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
    if (NA_NDIM(a_nary) != 2) {                                                                            \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                       \
      return Qnil;                                                                                         \
    }                                                                                                      \
                                                                                                           \
    ndfunc_arg_in_t ain[1] = { { tNAryClass, 2 } };                                                        \
    size_t shape_out[1] = { 1 };                                                                           \
    ndfunc_arg_out_t aout[1] = { { tNAryClass, 0, shape_out } };                                           \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };                         \
    struct _lange_option opt = { matrix_layout, norm };                                                    \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                        \
                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                  \
    return ret;                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dlange)
DEF_LINALG_FUNC(float, numo_cSFloat, slange)
DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, zlange)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, clange)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_lange(VALUE mLapack) {
  rb_define_module_function(mLapack, "dlange", RUBY_METHOD_FUNC(_linalg_lapack_dlange), -1);
  rb_define_module_function(mLapack, "slange", RUBY_METHOD_FUNC(_linalg_lapack_slange), -1);
  rb_define_module_function(mLapack, "zlange", RUBY_METHOD_FUNC(_linalg_lapack_zlange), -1);
  rb_define_module_function(mLapack, "clange", RUBY_METHOD_FUNC(_linalg_lapack_clange), -1);
}
