#include "gesdd.h"

struct _gesdd_option {
  int matrix_order;
  char jobz;
};

#define DEF_LINALG_FUNC(tDType, tRtType, tNAryType, tRtNAryType, fLapackFnc)                                                               \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                                                   \
    tRtType* s = (tRtType*)NDL_PTR(lp, 1);                                                                                                 \
    tDType* u = (tDType*)NDL_PTR(lp, 2);                                                                                                   \
    tDType* vt = (tDType*)NDL_PTR(lp, 3);                                                                                                  \
    int* info = (int*)NDL_PTR(lp, 4);                                                                                                      \
    struct _gesdd_option* opt = (struct _gesdd_option*)(lp->opt_ptr);                                                                      \
                                                                                                                                           \
    const lapack_int m = (lapack_int)(opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0] : NDL_SHAPE(lp, 0)[1]);                  \
    const lapack_int n = (lapack_int)(opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[1] : NDL_SHAPE(lp, 0)[0]);                  \
    const lapack_int min_mn = m < n ? m : n;                                                                                               \
    const lapack_int lda = n;                                                                                                              \
    const lapack_int ldu = opt->jobz == 'S' ? min_mn : m;                                                                                  \
    const lapack_int ldvt = opt->jobz == 'S' ? min_mn : n;                                                                                 \
                                                                                                                                           \
    lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_order, opt->jobz, m, n, a, lda, s, u, ldu, vt, ldvt);                                  \
    *info = (int)i;                                                                                                                        \
  }                                                                                                                                        \
                                                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                                                            \
    VALUE a_vnary = Qnil;                                                                                                                  \
    VALUE kw_args = Qnil;                                                                                                                  \
                                                                                                                                           \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                                                    \
                                                                                                                                           \
    ID kw_table[2] = { rb_intern("jobz"), rb_intern("order") };                                                                            \
    VALUE kw_values[2] = { Qundef, Qundef };                                                                                               \
                                                                                                                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);                                                                                     \
                                                                                                                                           \
    const char jobz = kw_values[0] == Qundef ? 'A' : StringValueCStr(kw_values[0])[0];                                                     \
    const char order = kw_values[1] == Qundef ? 'R' : StringValueCStr(kw_values[1])[0];                                                    \
                                                                                                                                           \
    if (CLASS_OF(a_vnary) != tNAryType) {                                                                                                  \
      rb_raise(rb_eTypeError, "type of input array is invalid for overwriting");                                                           \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                                                          \
      a_vnary = nary_dup(a_vnary);                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    narray_t* a_nary = NULL;                                                                                                               \
    GetNArray(a_vnary, a_nary);                                                                                                            \
    const int n_dims = NA_NDIM(a_nary);                                                                                                    \
    if (n_dims != 2) {                                                                                                                     \
      rb_raise(rb_eArgError, "input array must be 2-dimensional");                                                                         \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    const int matrix_order = order == 'C' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;                                                           \
    const size_t m = matrix_order == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];                                         \
    const size_t n = matrix_order == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[1] : NA_SHAPE(a_nary)[0];                                         \
                                                                                                                                           \
    const size_t min_mn = m < n ? m : n;                                                                                                   \
    size_t shape_s[1] = { min_mn };                                                                                                        \
    size_t shape_u[2] = { m, m };                                                                                                          \
    size_t shape_vt[2] = { n, n };                                                                                                         \
                                                                                                                                           \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                                                                         \
    ndfunc_arg_out_t aout[4] = { { tRtNAryType, 1, shape_s }, { tNAryType, 2, shape_u }, { tNAryType, 2, shape_vt }, { numo_cInt32, 0 } }; \
                                                                                                                                           \
    switch (jobz) {                                                                                                                        \
    case 'A':                                                                                                                              \
      break;                                                                                                                               \
    case 'S':                                                                                                                              \
      shape_u[matrix_order == LAPACK_ROW_MAJOR ? 1 : 0] = min_mn;                                                                          \
      shape_vt[matrix_order == LAPACK_ROW_MAJOR ? 0 : 1] = min_mn;                                                                         \
      break;                                                                                                                               \
    case 'O':                                                                                                                              \
      break;                                                                                                                               \
    case 'N':                                                                                                                              \
      aout[1].dim = 0;                                                                                                                     \
      aout[2].dim = 0;                                                                                                                     \
      break;                                                                                                                               \
    default:                                                                                                                               \
      rb_raise(rb_eArgError, "jobz must be one of 'A', 'S', 'O', or 'N'");                                                                 \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };                                                         \
    struct _gesdd_option opt = { matrix_order, jobz };                                                                                     \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                                                        \
                                                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                                                  \
    return ret;                                                                                                                            \
  }

DEF_LINALG_FUNC(double, double, numo_cDFloat, numo_cDFloat, dgesdd)
DEF_LINALG_FUNC(float, float, numo_cSFloat, numo_cSFloat, sgesdd)
DEF_LINALG_FUNC(lapack_complex_double, double, numo_cDComplex, numo_cDFloat, zgesdd)
DEF_LINALG_FUNC(lapack_complex_float, float, numo_cSComplex, numo_cSFloat, cgesdd)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gesdd(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgesdd", RUBY_METHOD_FUNC(_linalg_lapack_dgesdd), -1);
  rb_define_module_function(mLapack, "sgesdd", RUBY_METHOD_FUNC(_linalg_lapack_sgesdd), -1);
  rb_define_module_function(mLapack, "zgesdd", RUBY_METHOD_FUNC(_linalg_lapack_zgesdd), -1);
  rb_define_module_function(mLapack, "cgesdd", RUBY_METHOD_FUNC(_linalg_lapack_cgesdd), -1);
}
