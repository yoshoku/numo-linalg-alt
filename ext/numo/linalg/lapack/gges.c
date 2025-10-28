#include "gges.h"

#define DEF_GGES_OPTION(fLapackFunc, tSelectFunc)                                              \
  struct _gges_option_##fLapackFunc {                                                          \
    int matrix_layout;                                                                         \
    char jobvsl;                                                                               \
    char jobvsr;                                                                               \
    char sort;                                                                                 \
    tSelectFunc select;                                                                        \
  };

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)(NDL_PTR(lp, 0));                                                     \
    tDType* b = (tDType*)(NDL_PTR(lp, 1));                                                     \
    tDType* alpha_r = (tDType*)(NDL_PTR(lp, 2));                                               \
    tDType* alpha_i = (tDType*)(NDL_PTR(lp, 3));                                               \
    tDType* beta = (tDType*)(NDL_PTR(lp, 4));                                                  \
    tDType* vsl = (tDType*)(NDL_PTR(lp, 5));                                                   \
    tDType* vsr = (tDType*)(NDL_PTR(lp, 6));                                                   \
    int* sdim = (int*)(NDL_PTR(lp, 7));                                                        \
    int* info = (int*)(NDL_PTR(lp, 8));                                                        \
    struct _gges_option_##fLapackFunc* opt =                                                   \
      (struct _gges_option_##fLapackFunc*)(lp->opt_ptr);                                       \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldb = n;                                                                  \
    const lapack_int ldvsl = (opt->jobvsl == 'N') ? 1 : n;                                     \
    const lapack_int ldvsr = (opt->jobvsr == 'N') ? 1 : n;                                     \
    lapack_int s = 0;                                                                          \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvsl, opt->jobvsr, opt->sort, opt->select, n, a, lda, b, ldb, \
      &s, alpha_r, alpha_i, beta, vsl, ldvsl, vsr, ldvsr                                       \
    );                                                                                         \
    *sdim = (int)s;                                                                            \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE b_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                              \
    ID kw_table[4] = { rb_intern("order"), rb_intern("jobvsl"), rb_intern("jobvsr"),           \
                       rb_intern("sort") };                                                    \
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvsl = kw_values[1] != Qundef ? get_job(kw_values[1], "jobvsl") : 'V';        \
    const char jobvsr = kw_values[2] != Qundef ? get_job(kw_values[2], "jobvsr") : 'V';        \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                     \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                              \
      b_vnary = nary_dup(b_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    if (NA_NDIM(a_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional array");                     \
      return Qnil;                                                                             \
    }                                                                                          \
    narray_t* b_nary = NULL;                                                                   \
    GetNArray(b_vnary, b_nary);                                                                \
    if (NA_NDIM(b_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array b must be 2-dimensional array");                     \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t n = matrix_layout == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];  \
    size_t shape_alphar[1] = { n };                                                            \
    size_t shape_alphai[1] = { n };                                                            \
    size_t shape_beta[1] = { n };                                                              \
    size_t shape_vsl[2] = { n, jobvsl == 'N' ? 1 : n };                                        \
    size_t shape_vsr[2] = { n, jobvsr == 'N' ? 1 : n };                                        \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };                           \
    ndfunc_arg_out_t aout[7] = { { tNAryClass, 1, shape_alphar },                              \
                                 { tNAryClass, 1, shape_alphai },                              \
                                 { tNAryClass, 1, shape_beta },                                \
                                 { tNAryClass, 2, shape_vsl },                                 \
                                 { tNAryClass, 2, shape_vsr },                                 \
                                 { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 7, ain, aout };            \
    struct _gges_option_##fLapackFunc opt = { matrix_layout, jobvsl, jobvsr, 'N', NULL };      \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                   \
    VALUE ret = rb_ary_concat(rb_ary_new3(2, a_vnary, b_vnary), res);                          \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    RB_GC_GUARD(b_vnary);                                                                      \
    return ret;                                                                                \
  }

#define DEF_LINALG_FUNC_COMPLEX(tDType, tNAryClass, fLapackFunc)                               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)(NDL_PTR(lp, 0));                                                     \
    tDType* b = (tDType*)(NDL_PTR(lp, 1));                                                     \
    tDType* alpha = (tDType*)(NDL_PTR(lp, 2));                                                 \
    tDType* beta = (tDType*)(NDL_PTR(lp, 3));                                                  \
    tDType* vsl = (tDType*)(NDL_PTR(lp, 4));                                                   \
    tDType* vsr = (tDType*)(NDL_PTR(lp, 5));                                                   \
    int* sdim = (int*)(NDL_PTR(lp, 6));                                                        \
    int* info = (int*)(NDL_PTR(lp, 7));                                                        \
    struct _gges_option_##fLapackFunc* opt =                                                   \
      (struct _gges_option_##fLapackFunc*)(lp->opt_ptr);                                       \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldb = n;                                                                  \
    const lapack_int ldvsl = (opt->jobvsl == 'N') ? 1 : n;                                     \
    const lapack_int ldvsr = (opt->jobvsr == 'N') ? 1 : n;                                     \
    lapack_int s = 0;                                                                          \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvsl, opt->jobvsr, opt->sort, opt->select, n, a, lda, b, ldb, \
      &s, alpha, beta, vsl, ldvsl, vsr, ldvsr                                                  \
    );                                                                                         \
    *sdim = (int)s;                                                                            \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE b_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                              \
    ID kw_table[4] = { rb_intern("order"), rb_intern("jobvsl"), rb_intern("jobvsr"),           \
                       rb_intern("sort") };                                                    \
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvsl = kw_values[1] != Qundef ? get_job(kw_values[1], "jobvsl") : 'V';        \
    const char jobvsr = kw_values[2] != Qundef ? get_job(kw_values[2], "jobvsr") : 'V';        \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                     \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                              \
      b_vnary = nary_dup(b_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    if (NA_NDIM(a_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional array");                     \
      return Qnil;                                                                             \
    }                                                                                          \
    narray_t* b_nary = NULL;                                                                   \
    GetNArray(b_vnary, b_nary);                                                                \
    if (NA_NDIM(b_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array b must be 2-dimensional array");                     \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t n = matrix_layout == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];  \
    size_t shape_alpha[1] = { n };                                                             \
    size_t shape_beta[1] = { n };                                                              \
    size_t shape_vsl[2] = { n, jobvsl == 'N' ? 1 : n };                                        \
    size_t shape_vsr[2] = { n, jobvsr == 'N' ? 1 : n };                                        \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };                           \
    ndfunc_arg_out_t aout[6] = { { tNAryClass, 1, shape_alpha },                               \
                                 { tNAryClass, 1, shape_beta },                                \
                                 { tNAryClass, 2, shape_vsl },                                 \
                                 { tNAryClass, 2, shape_vsr },                                 \
                                 { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 6, ain, aout };            \
    struct _gges_option_##fLapackFunc opt = { matrix_layout, jobvsl, jobvsr, 'N', NULL };      \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                   \
    VALUE ret = rb_ary_concat(rb_ary_new3(2, a_vnary, b_vnary), res);                          \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    RB_GC_GUARD(b_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_GGES_OPTION(dgges, LAPACK_D_SELECT3)
DEF_GGES_OPTION(sgges, LAPACK_S_SELECT3)
DEF_GGES_OPTION(zgges, LAPACK_Z_SELECT2)
DEF_GGES_OPTION(cgges, LAPACK_C_SELECT2)

DEF_LINALG_FUNC(double, numo_cDFloat, dgges)
DEF_LINALG_FUNC(float, numo_cSFloat, sgges)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_double, numo_cDComplex, zgges)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_float, numo_cSComplex, cgges)

#undef DEF_GGES_OPTION
#undef DEF_LINALG_FUNC
#undef DEF_LINALG_FUNC_COMPLEX

void define_linalg_lapack_gges(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgges", _linalg_lapack_dgges, -1);
  rb_define_module_function(mLapack, "sgges", _linalg_lapack_sgges, -1);
  rb_define_module_function(mLapack, "zgges", _linalg_lapack_zgges, -1);
  rb_define_module_function(mLapack, "cgges", _linalg_lapack_cgges, -1);
}
