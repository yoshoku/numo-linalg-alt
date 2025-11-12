#include "heev.h"

struct _heev_option {
  int matrix_layout;
  char jobz;
  char uplo;
};

#define DEF_LINALG_FUNC(tDType, tNAryClass, tRtDType, tRtNAryClass, fLapackFunc)               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tRtDType* w = (tRtDType*)NDL_PTR(lp, 1);                                                   \
    int* info = (int*)NDL_PTR(lp, 2);                                                          \
    struct _heev_option* opt = (struct _heev_option*)(lp->opt_ptr);                            \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                      \
    const lapack_int lda = (lapack_int)NDL_SHAPE(lp, 0)[0];                                    \
    const lapack_int i =                                                                       \
      LAPACKE_##fLapackFunc(opt->matrix_layout, opt->jobz, opt->uplo, n, a, lda, w);           \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("jobz"), rb_intern("uplo"), rb_intern("order") };             \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                         \
    const char jobz = kw_values[0] != Qundef ? get_job(kw_values[0], "jobz") : 'V';            \
    const char uplo = kw_values[1] != Qundef ? get_uplo(kw_values[1]) : 'U';                   \
    const int matrix_layout =                                                                  \
      kw_values[2] != Qundef ? get_matrix_layout(kw_values[2]) : LAPACK_ROW_MAJOR;             \
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
    const size_t n = NA_SHAPE(a_nary)[1];                                                      \
    size_t shape[1] = { n };                                                                   \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[2] = { { tRtNAryClass, 1, shape }, { numo_cInt32, 0 } };             \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };            \
    struct _heev_option opt = { matrix_layout, jobz, uplo };                                   \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
    VALUE ret = rb_ary_new3(3, a_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1));           \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, double, numo_cDFloat, zheev)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, float, numo_cSFloat, cheev)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_heev(VALUE mLapack) {
  rb_define_module_function(mLapack, "zheev", _linalg_lapack_zheev, -1);
  rb_define_module_function(mLapack, "cheev", _linalg_lapack_cheev, -1);
}
