#include "gebal.h"

struct _gebal_option {
  int matrix_layout;
  char job;
};

char _get_job(VALUE val) {
  const char job = NUM2CHR(val);
  if (job != 'N' && job != 'P' && job != 'S' && job != 'B') {
    rb_raise(rb_eArgError, "job must be 'N', 'P', 'S', or 'B'");
  }
  return job;
}

#define DEF_LINALG_FUNC(tDType, tRtDType, tNAryClass, tRtNAryClass, fLapackFunc)               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    int* ilo = (int*)NDL_PTR(lp, 1);                                                           \
    int* ihi = (int*)NDL_PTR(lp, 2);                                                           \
    tRtDType* scale = (tRtDType*)NDL_PTR(lp, 3);                                               \
    int* info = (int*)NDL_PTR(lp, 4);                                                          \
    struct _gebal_option* opt = (struct _gebal_option*)(lp->opt_ptr);                          \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    lapack_int i =                                                                             \
      LAPACKE_##fLapackFunc(opt->matrix_layout, opt->job, n, a, lda, ilo, ihi, scale);         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[2] = { rb_intern("order"), rb_intern("job") };                                 \
    VALUE kw_values[2] = { Qundef, Qundef };                                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char job = kw_values[1] != Qundef ? _get_job(kw_values[1]) : 'B';                    \
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
    size_t shape_scale[1] = { n };                                                             \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[4] = { { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 },                                           \
                                 { tRtNAryClass, 1, shape_scale },                             \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };            \
    struct _gebal_option opt = { matrix_layout, job };                                         \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);                                   \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, double, numo_cDFloat, numo_cDFloat, dgebal)
DEF_LINALG_FUNC(float, float, numo_cSFloat, numo_cSFloat, sgebal)
DEF_LINALG_FUNC(lapack_complex_double, double, numo_cDComplex, numo_cDFloat, zgebal)
DEF_LINALG_FUNC(lapack_complex_float, float, numo_cSComplex, numo_cSFloat, cgebal)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gebal(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgebal", _linalg_lapack_dgebal, -1);
  rb_define_module_function(mLapack, "sgebal", _linalg_lapack_sgebal, -1);
  rb_define_module_function(mLapack, "zgebal", _linalg_lapack_zgebal, -1);
  rb_define_module_function(mLapack, "cgebal", _linalg_lapack_cgebal, -1);
}
