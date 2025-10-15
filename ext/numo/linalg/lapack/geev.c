#include "geev.h"

struct _geev_option {
  int matrix_layout;
  char jobvl;
  char jobvr;
};

char _get_jobvl(VALUE val) {
  const char jobvl = NUM2CHR(val);
  if (jobvl != 'N' && jobvl != 'V') {
    rb_raise(rb_eArgError, "jobvl must be 'N' or 'V'");
  }
  return jobvl;
}

char _get_jobvr(VALUE val) {
  const char jobvr = NUM2CHR(val);
  if (jobvr != 'N' && jobvr != 'V') {
    rb_raise(rb_eArgError, "jobvr must be 'N' or 'V'");
  }
  return jobvr;
}

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* wr = (tDType*)NDL_PTR(lp, 1);                                                      \
    tDType* wi = (tDType*)NDL_PTR(lp, 2);                                                      \
    tDType* vl = (tDType*)NDL_PTR(lp, 3);                                                      \
    tDType* vr = (tDType*)NDL_PTR(lp, 4);                                                      \
    int* info = (int*)NDL_PTR(lp, 5);                                                          \
    struct _geev_option* opt = (struct _geev_option*)(lp->opt_ptr);                            \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldvl = (opt->jobvl == 'N') ? 1 : n;                                       \
    const lapack_int ldvr = (opt->jobvr == 'N') ? 1 : n;                                       \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvl, opt->jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr        \
    );                                                                                         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("order"), rb_intern("jobvl"), rb_intern("jobvr") };           \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvl = kw_values[1] != Qundef ? _get_jobvl(kw_values[1]) : 'V';                \
    const char jobvr = kw_values[2] != Qundef ? _get_jobvr(kw_values[2]) : 'V';                \
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
    size_t shape_wr[1] = { n };                                                                \
    size_t shape_wi[1] = { n };                                                                \
    size_t shape_vl[2] = { n, (jobvl == 'N') ? 1 : n };                                        \
    size_t shape_vr[2] = { n, (jobvr == 'N') ? 1 : n };                                        \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[5] = { { tNAryClass, 1, shape_wr },                                  \
                                 { tNAryClass, 1, shape_wi },                                  \
                                 { tNAryClass, 2, shape_vl },                                  \
                                 { tNAryClass, 2, shape_vr },                                  \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 5, ain, aout };            \
    struct _geev_option opt = { matrix_layout, jobvl, jobvr };                                 \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

#define DEF_LINALG_FUNC_COMPLEX(tDType, tNAryClass, fLapackFunc)                               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* w = (tDType*)NDL_PTR(lp, 1);                                                       \
    tDType* vl = (tDType*)NDL_PTR(lp, 2);                                                      \
    tDType* vr = (tDType*)NDL_PTR(lp, 3);                                                      \
    int* info = (int*)NDL_PTR(lp, 4);                                                          \
    struct _geev_option* opt = (struct _geev_option*)(lp->opt_ptr);                            \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldvl = (opt->jobvl == 'N') ? 1 : n;                                       \
    const lapack_int ldvr = (opt->jobvr == 'N') ? 1 : n;                                       \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvl, opt->jobvr, n, a, lda, w, vl, ldvl, vr, ldvr             \
    );                                                                                         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("order"), rb_intern("jobvl"), rb_intern("jobvr") };           \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvl = kw_values[1] != Qundef ? _get_jobvl(kw_values[1]) : 'V';                \
    const char jobvr = kw_values[2] != Qundef ? _get_jobvr(kw_values[2]) : 'V';                \
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
    size_t shape_w[1] = { n };                                                                 \
    size_t shape_vl[2] = { n, (jobvl == 'N') ? 1 : n };                                        \
    size_t shape_vr[2] = { n, (jobvr == 'N') ? 1 : n };                                        \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[4] = { { tNAryClass, 1, shape_w },                                   \
                                 { tNAryClass, 2, shape_vl },                                  \
                                 { tNAryClass, 2, shape_vr },                                  \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };            \
    struct _geev_option opt = { matrix_layout, jobvl, jobvr };                                 \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dgeev)
DEF_LINALG_FUNC(float, numo_cSFloat, sgeev)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_double, numo_cDComplex, zgeev)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_float, numo_cSComplex, cgeev)

#undef DEF_LINALG_FUNC
#undef DEF_LINALG_FUNC_COMPLEX

void define_linalg_lapack_geev(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgeev", RUBY_METHOD_FUNC(_linalg_lapack_dgeev), -1);
  rb_define_module_function(mLapack, "sgeev", RUBY_METHOD_FUNC(_linalg_lapack_sgeev), -1);
  rb_define_module_function(mLapack, "zgeev", RUBY_METHOD_FUNC(_linalg_lapack_zgeev), -1);
  rb_define_module_function(mLapack, "cgeev", RUBY_METHOD_FUNC(_linalg_lapack_cgeev), -1);
}
