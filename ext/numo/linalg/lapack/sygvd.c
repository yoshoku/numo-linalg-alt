#include "sygvd.h"

struct _sygvd_option {
  int matrix_layout;
  lapack_int itype;
  char jobz;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFnc)                                                                    \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                                    \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                                   \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                                                   \
    tDType* w = (tDType*)NDL_PTR(lp, 2);                                                                                   \
    int* info = (int*)NDL_PTR(lp, 3);                                                                                      \
    struct _sygvd_option* opt = (struct _sygvd_option*)(lp->opt_ptr);                                                      \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                                  \
    const lapack_int lda = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                                \
    const lapack_int ldb = (lapack_int)NDL_SHAPE(lp, 1)[0];                                                                \
    const lapack_int i = LAPACKE_##fLapackFnc(opt->matrix_layout, opt->itype, opt->jobz, opt->uplo, n, a, lda, b, ldb, w); \
    *info = (int)i;                                                                                                        \
  }                                                                                                                        \
                                                                                                                           \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                                            \
    VALUE a_vnary = Qnil;                                                                                                  \
    VALUE b_vnary = Qnil;                                                                                                  \
    VALUE kw_args = Qnil;                                                                                                  \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                                                          \
    ID kw_table[4] = { rb_intern("itype"), rb_intern("jobz"), rb_intern("uplo"), rb_intern("order") };                     \
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };                                                               \
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);                                                                     \
    const lapack_int itype = kw_values[0] != Qundef ? get_itype(kw_values[0]) : 1;                                         \
    const char jobz = kw_values[1] != Qundef ? get_jobz(kw_values[1]) : 'V';                                               \
    const char uplo = kw_values[2] != Qundef ? get_uplo(kw_values[2]) : 'U';                                               \
    const int matrix_layout = kw_values[3] != Qundef ? get_matrix_layout(kw_values[3]) : LAPACK_ROW_MAJOR;                 \
                                                                                                                           \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                                                 \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                                                     \
    }                                                                                                                      \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                                          \
      a_vnary = nary_dup(a_vnary);                                                                                         \
    }                                                                                                                      \
    if (CLASS_OF(b_vnary) != tNAryClass) {                                                                                 \
      b_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, b_vnary);                                                     \
    }                                                                                                                      \
    if (!RTEST(nary_check_contiguous(b_vnary))) {                                                                          \
      b_vnary = nary_dup(b_vnary);                                                                                         \
    }                                                                                                                      \
                                                                                                                           \
    narray_t* a_nary = NULL;                                                                                               \
    GetNArray(a_vnary, a_nary);                                                                                            \
    if (NA_NDIM(a_nary) != 2) {                                                                                            \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                                       \
      return Qnil;                                                                                                         \
    }                                                                                                                      \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                                                      \
      rb_raise(rb_eArgError, "input array a must be square");                                                              \
      return Qnil;                                                                                                         \
    }                                                                                                                      \
    narray_t* b_nary = NULL;                                                                                               \
    GetNArray(b_vnary, b_nary);                                                                                            \
    if (NA_NDIM(b_nary) != 2) {                                                                                            \
      rb_raise(rb_eArgError, "input array b must be 2-dimensional");                                                       \
      return Qnil;                                                                                                         \
    }                                                                                                                      \
    if (NA_SHAPE(b_nary)[0] != NA_SHAPE(b_nary)[1]) {                                                                      \
      rb_raise(rb_eArgError, "input array b must be square");                                                              \
      return Qnil;                                                                                                         \
    }                                                                                                                      \
                                                                                                                           \
    const size_t n = NA_SHAPE(a_nary)[1];                                                                                  \
    size_t shape[1] = { n };                                                                                               \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };                                                       \
    ndfunc_arg_out_t aout[2] = { { tNAryClass, 1, shape }, { numo_cInt32, 0 } };                                           \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 2, 2, ain, aout };                                         \
    struct _sygvd_option opt = { matrix_layout, itype, jobz, uplo };                                                       \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                                               \
    VALUE ret = rb_ary_new3(4, a_vnary, b_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1));                              \
                                                                                                                           \
    RB_GC_GUARD(a_vnary);                                                                                                  \
    RB_GC_GUARD(b_vnary);                                                                                                  \
    return ret;                                                                                                            \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dsygvd)
DEF_LINALG_FUNC(float, numo_cSFloat, ssygvd)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_sygvd(VALUE mLapack) {
  rb_define_module_function(mLapack, "dsygvd", RUBY_METHOD_FUNC(_linalg_lapack_dsygvd), -1);
  rb_define_module_function(mLapack, "ssygvd", RUBY_METHOD_FUNC(_linalg_lapack_ssygvd), -1);
}
