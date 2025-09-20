#include "gesvd.h"

struct _gesvd_option {
  int matrix_order;
  char jobu;
  char jobvt;
};

#define DEF_LINALG_FUNC(tDType, tRtType, tNArrType, tRtNArrType, fLapackFnc)                                                               \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                                                   \
    tRtType* s = (tRtType*)NDL_PTR(lp, 1);                                                                                                 \
    tDType* u = (tDType*)NDL_PTR(lp, 2);                                                                                                   \
    tDType* vt = (tDType*)NDL_PTR(lp, 3);                                                                                                  \
    int* info = (int*)NDL_PTR(lp, 4);                                                                                                      \
    struct _gesvd_option* opt = (struct _gesvd_option*)(lp->opt_ptr);                                                                      \
                                                                                                                                           \
    const size_t m = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0] : NDL_SHAPE(lp, 0)[1];                                    \
    const size_t n = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[1] : NDL_SHAPE(lp, 0)[0];                                    \
    const size_t min_mn = m < n ? m : n;                                                                                                   \
    const lapack_int lda = n;                                                                                                              \
    const lapack_int ldu = opt->jobu == 'A' ? m : min_mn;                                                                                  \
    const lapack_int ldvt = n;                                                                                                             \
                                                                                                                                           \
    tRtType* superb = (tRtType*)ruby_xmalloc(min_mn * sizeof(tRtType));                                                                    \
                                                                                                                                           \
    lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_order, opt->jobu, opt->jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);              \
    *info = (int)i;                                                                                                                        \
                                                                                                                                           \
    ruby_xfree(superb);                                                                                                                    \
  }                                                                                                                                        \
                                                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                                                            \
    VALUE a_vnary = Qnil;                                                                                                                  \
    VALUE kw_args = Qnil;                                                                                                                  \
                                                                                                                                           \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                                                    \
                                                                                                                                           \
    ID kw_table[3] = { rb_intern("jobu"), rb_intern("jobvt"), rb_intern("order") };                                                        \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                                                                       \
                                                                                                                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                                                                     \
                                                                                                                                           \
    const char jobu = kw_values[0] == Qundef ? 'A' : StringValueCStr(kw_values[0])[0];                                                     \
    const char jobvt = kw_values[1] == Qundef ? 'A' : StringValueCStr(kw_values[1])[0];                                                    \
    const char order = kw_values[2] == Qundef ? 'R' : StringValueCStr(kw_values[2])[0];                                                    \
                                                                                                                                           \
    if (jobu == 'O' && jobvt == 'O') {                                                                                                     \
      rb_raise(rb_eArgError, "jobu and jobvt cannot be both 'O'");                                                                         \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
    if (CLASS_OF(a_vnary) != tNArrType) {                                                                                                  \
      rb_raise(rb_eTypeError, "type of input array is invalid for overwriting");                                                           \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    if (CLASS_OF(a_vnary) != tNArrType) {                                                                                                  \
      a_vnary = rb_funcall(tNArrType, rb_intern("cast"), 1, a_vnary);                                                                      \
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
    ndfunc_arg_out_t aout[4] = { { tRtNArrType, 1, shape_s }, { tNArrType, 2, shape_u }, { tNArrType, 2, shape_vt }, { numo_cInt32, 0 } }; \
                                                                                                                                           \
    switch (jobu) {                                                                                                                        \
    case 'A':                                                                                                                              \
      break;                                                                                                                               \
    case 'S':                                                                                                                              \
      shape_u[matrix_order == LAPACK_ROW_MAJOR ? 1 : 0] = min_mn;                                                                          \
      break;                                                                                                                               \
    case 'O':                                                                                                                              \
    case 'N':                                                                                                                              \
      aout[1].dim = 0;                                                                                                                     \
      break;                                                                                                                               \
    default:                                                                                                                               \
      rb_raise(rb_eArgError, "jobu must be 'A', 'S', 'O', or 'N'");                                                                        \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    switch (jobvt) {                                                                                                                       \
    case 'A':                                                                                                                              \
      break;                                                                                                                               \
    case 'S':                                                                                                                              \
      shape_vt[matrix_order == LAPACK_ROW_MAJOR ? 0 : 1] = min_mn;                                                                         \
      break;                                                                                                                               \
    case 'O':                                                                                                                              \
    case 'N':                                                                                                                              \
      aout[2].dim = 0;                                                                                                                     \
      break;                                                                                                                               \
    default:                                                                                                                               \
      rb_raise(rb_eArgError, "jobvt must be 'A', 'S', 'O', or 'N'");                                                                       \
      return Qnil;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };                                                         \
    struct _gesvd_option opt = { matrix_order, jobu, jobvt };                                                                              \
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                                                        \
                                                                                                                                           \
    switch (jobu) {                                                                                                                        \
    case 'O':                                                                                                                              \
      rb_ary_store(ret, 1, a_vnary);                                                                                                       \
      break;                                                                                                                               \
    case 'N':                                                                                                                              \
      rb_ary_store(ret, 1, Qnil);                                                                                                          \
      break;                                                                                                                               \
    }                                                                                                                                      \
                                                                                                                                           \
    switch (jobvt) {                                                                                                                       \
    case 'O':                                                                                                                              \
      rb_ary_store(ret, 2, a_vnary);                                                                                                       \
      break;                                                                                                                               \
    case 'N':                                                                                                                              \
      rb_ary_store(ret, 2, Qnil);                                                                                                          \
      break;                                                                                                                               \
    }                                                                                                                                      \
                                                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                                                  \
    return ret;                                                                                                                            \
  }

DEF_LINALG_FUNC(double, double, numo_cDFloat, numo_cDFloat, dgesvd)
DEF_LINALG_FUNC(float, float, numo_cSFloat, numo_cSFloat, sgesvd)
DEF_LINALG_FUNC(lapack_complex_double, double, numo_cDComplex, numo_cDFloat, zgesvd)
DEF_LINALG_FUNC(lapack_complex_float, float, numo_cSComplex, numo_cSFloat, cgesvd)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_gesvd(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgesvd", RUBY_METHOD_FUNC(_linalg_lapack_dgesvd), -1);
  rb_define_module_function(mLapack, "sgesvd", RUBY_METHOD_FUNC(_linalg_lapack_sgesvd), -1);
  rb_define_module_function(mLapack, "zgesvd", RUBY_METHOD_FUNC(_linalg_lapack_zgesvd), -1);
  rb_define_module_function(mLapack, "cgesvd", RUBY_METHOD_FUNC(_linalg_lapack_cgesvd), -1);
}
