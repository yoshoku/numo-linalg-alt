#include "heevr.h"

#define DEF_LINALG_FUNC(tDType, tRtType, tNAryType, tRtNAryType, fLapackFnc)                                                                                         \
  struct _heevr_option_##tRtType {                                                                                                                                   \
    int matrix_layout;                                                                                                                                               \
    char jobz;                                                                                                                                                       \
    char range;                                                                                                                                                      \
    char uplo;                                                                                                                                                       \
    tRtType vl;                                                                                                                                                      \
    tRtType vu;                                                                                                                                                      \
    lapack_int il;                                                                                                                                                   \
    lapack_int iu;                                                                                                                                                   \
  };                                                                                                                                                                 \
                                                                                                                                                                     \
  static void _iter_##fLapackFnc(na_loop_t* const lp) {                                                                                                              \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                                                                                             \
    int* m = (int*)NDL_PTR(lp, 1);                                                                                                                                   \
    tRtType* w = (tRtType*)NDL_PTR(lp, 2);                                                                                                                           \
    tDType* z = (tDType*)NDL_PTR(lp, 3);                                                                                                                             \
    int* isuppz = (int*)NDL_PTR(lp, 4);                                                                                                                              \
    int* info = (int*)NDL_PTR(lp, 5);                                                                                                                                \
    struct _heevr_option_##tRtType* opt = (struct _heevr_option_##tRtType*)(lp->opt_ptr);                                                                            \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                                                                                            \
    const lapack_int lda = (lapack_int)NDL_SHAPE(lp, 0)[0];                                                                                                          \
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;                                                                                            \
    const tRtType abstol = 0.0;                                                                                                                                      \
    const lapack_int i = LAPACKE_##fLapackFnc(                                                                                                                       \
      opt->matrix_layout, opt->jobz, opt->range, opt->uplo, n, a, lda,                                                                                               \
      opt->vl, opt->vu, opt->il, opt->iu, abstol, m, w, z, ldz, isuppz);                                                                                             \
    *info = (int)i;                                                                                                                                                  \
  }                                                                                                                                                                  \
                                                                                                                                                                     \
  static VALUE _linalg_lapack_##fLapackFnc(int argc, VALUE* argv, VALUE self) {                                                                                      \
    VALUE a_vnary = Qnil;                                                                                                                                            \
    VALUE kw_args = Qnil;                                                                                                                                            \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                                                                                              \
    ID kw_table[8] = { rb_intern("jobz"), rb_intern("range"), rb_intern("uplo"),                                                                                     \
                       rb_intern("vl"), rb_intern("vu"), rb_intern("il"), rb_intern("iu"), rb_intern("order") };                                                     \
    VALUE kw_values[8] = { Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef };                                                                         \
    rb_get_kwargs(kw_args, kw_table, 0, 8, kw_values);                                                                                                               \
    const char jobz = kw_values[0] != Qundef ? get_jobz(kw_values[0]) : 'V';                                                                                         \
    const char range = kw_values[1] != Qundef ? get_range(kw_values[1]) : 'A';                                                                                       \
    const char uplo = kw_values[2] != Qundef ? get_uplo(kw_values[2]) : 'U';                                                                                         \
    const tRtType vl = kw_values[3] != Qundef ? NUM2DBL(kw_values[3]) : 0.0;                                                                                         \
    const tRtType vu = kw_values[4] != Qundef ? NUM2DBL(kw_values[4]) : 0.0;                                                                                         \
    const lapack_int il = kw_values[5] != Qundef ? NUM2INT(kw_values[5]) : 0;                                                                                        \
    const lapack_int iu = kw_values[6] != Qundef ? NUM2INT(kw_values[6]) : 0;                                                                                        \
    const int matrix_layout = kw_values[7] != Qundef ? get_matrix_layout(kw_values[7]) : LAPACK_ROW_MAJOR;                                                           \
                                                                                                                                                                     \
    if (CLASS_OF(a_vnary) != tNAryType) {                                                                                                                            \
      a_vnary = rb_funcall(tNAryType, rb_intern("cast"), 1, a_vnary);                                                                                                \
    }                                                                                                                                                                \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                                                                                                    \
      a_vnary = nary_dup(a_vnary);                                                                                                                                   \
    }                                                                                                                                                                \
                                                                                                                                                                     \
    narray_t* a_nary = NULL;                                                                                                                                         \
    GetNArray(a_vnary, a_nary);                                                                                                                                      \
    if (NA_NDIM(a_nary) != 2) {                                                                                                                                      \
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                                                                                                 \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                                                                                                \
      rb_raise(rb_eArgError, "input array a must be square");                                                                                                        \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
                                                                                                                                                                     \
    if (range == 'V' && vu <= vl) {                                                                                                                                  \
      rb_raise(rb_eArgError, "vu must be greater than vl");                                                                                                          \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
                                                                                                                                                                     \
    const size_t n = NA_SHAPE(a_nary)[1];                                                                                                                            \
    if (range == 'I' && (il < 1 || il > n)) {                                                                                                                        \
      rb_raise(rb_eArgError, "il must satisfy 1 <= il <= n");                                                                                                        \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
    if (range == 'I' && (iu < 1 || iu > n)) {                                                                                                                        \
      rb_raise(rb_eArgError, "iu must satisfy 1 <= iu <= n");                                                                                                        \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
    if (range == 'I' && iu < il) {                                                                                                                                   \
      rb_raise(rb_eArgError, "iu must be greater than or equal to il");                                                                                              \
      return Qnil;                                                                                                                                                   \
    }                                                                                                                                                                \
                                                                                                                                                                     \
    size_t m = range != 'I' ? n : (size_t)(iu - il + 1);                                                                                                             \
    size_t w_shape[1] = { m };                                                                                                                                       \
    size_t z_shape[2] = { n, m };                                                                                                                                    \
    size_t isuppz_shape[1] = { 2 * m };                                                                                                                              \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                                                                                                   \
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 }, { tRtNAryType, 1, w_shape }, { tNAryType, 2, z_shape }, { numo_cInt32, 1, isuppz_shape }, { numo_cInt32, 0 } }; \
    ndfunc_t ndf = { _iter_##fLapackFnc, NO_LOOP | NDF_EXTRACT, 1, 5, ain, aout };                                                                                   \
    struct _heevr_option_##tRtType opt = { matrix_layout, jobz, range, uplo, vl, vu, il, iu };                                                                       \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                                                                                                  \
    VALUE ret = rb_ary_new3(6, a_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1), rb_ary_entry(res, 2),                                                            \
                            rb_ary_entry(res, 3), rb_ary_entry(res, 4));                                                                                             \
                                                                                                                                                                     \
    RB_GC_GUARD(a_vnary);                                                                                                                                            \
    return ret;                                                                                                                                                      \
  }

DEF_LINALG_FUNC(lapack_complex_double, double, numo_cDComplex, numo_cDFloat, zheevr)
DEF_LINALG_FUNC(lapack_complex_float, float, numo_cSComplex, numo_cSFloat, cheevr)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_heevr(VALUE mLapack) {
  rb_define_module_function(mLapack, "zheevr", RUBY_METHOD_FUNC(_linalg_lapack_zheevr), -1);
  rb_define_module_function(mLapack, "cheevr", RUBY_METHOD_FUNC(_linalg_lapack_cheevr), -1);
}
