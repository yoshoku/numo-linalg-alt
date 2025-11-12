#include "hegvx.h"

#define DEF_LINALG_FUNC(tDType, tNAryClass, tRtDType, tRtNAryClass, fLapackFunc)               \
  struct _hegvx_option_##tRtDType {                                                            \
    int matrix_layout;                                                                         \
    lapack_int itype;                                                                          \
    char jobz;                                                                                 \
    char range;                                                                                \
    char uplo;                                                                                 \
    tRtDType vl;                                                                               \
    tRtDType vu;                                                                               \
    lapack_int il;                                                                             \
    lapack_int iu;                                                                             \
  };                                                                                           \
                                                                                               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* b = (tDType*)NDL_PTR(lp, 1);                                                       \
    int* m = (int*)NDL_PTR(lp, 2);                                                             \
    tRtDType* w = (tRtDType*)NDL_PTR(lp, 3);                                                   \
    tDType* z = (tDType*)NDL_PTR(lp, 4);                                                       \
    int* ifail = (int*)NDL_PTR(lp, 5);                                                         \
    int* info = (int*)NDL_PTR(lp, 6);                                                          \
    struct _hegvx_option_##tRtDType* opt = (struct _hegvx_option_##tRtDType*)(lp->opt_ptr);    \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[1];                                      \
    const lapack_int lda = (lapack_int)NDL_SHAPE(lp, 0)[0];                                    \
    const lapack_int ldb = (lapack_int)NDL_SHAPE(lp, 1)[0];                                    \
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;                      \
    const tRtDType abstol = 0.0;                                                               \
    const lapack_int i = LAPACKE_##fLapackFunc(                                                \
      opt->matrix_layout, opt->itype, opt->jobz, opt->range, opt->uplo, n, a, lda, b, ldb,     \
      opt->vl, opt->vu, opt->il, opt->iu, abstol, m, w, z, ldz, ifail                          \
    );                                                                                         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE b_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);                              \
    ID kw_table[9] = { rb_intern("itype"), rb_intern("jobz"), rb_intern("range"),              \
                       rb_intern("uplo"),  rb_intern("vl"),   rb_intern("vu"),                 \
                       rb_intern("il"),    rb_intern("iu"),   rb_intern("order") };            \
    VALUE kw_values[9] = { Qundef, Qundef, Qundef, Qundef, Qundef,                             \
                           Qundef, Qundef, Qundef, Qundef };                                   \
    rb_get_kwargs(kw_args, kw_table, 0, 9, kw_values);                                         \
    const lapack_int itype = kw_values[0] != Qundef ? get_itype(kw_values[0]) : 1;             \
    const char jobz = kw_values[1] != Qundef ? get_job(kw_values[1], "jobz") : 'V';            \
    const char range = kw_values[2] != Qundef ? get_range(kw_values[2]) : 'A';                 \
    const char uplo = kw_values[3] != Qundef ? get_uplo(kw_values[3]) : 'U';                   \
    const tRtDType vl = kw_values[4] != Qundef ? NUM2DBL(kw_values[4]) : 0.0;                  \
    const tRtDType vu = kw_values[5] != Qundef ? NUM2DBL(kw_values[5]) : 0.0;                  \
    const lapack_int il = kw_values[6] != Qundef ? NUM2INT(kw_values[6]) : 0;                  \
    const lapack_int iu = kw_values[7] != Qundef ? NUM2INT(kw_values[7]) : 0;                  \
    const int matrix_layout =                                                                  \
      kw_values[8] != Qundef ? get_matrix_layout(kw_values[8]) : LAPACK_ROW_MAJOR;             \
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
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {                                          \
      rb_raise(rb_eArgError, "input array a must be square");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
    narray_t* b_nary = NULL;                                                                   \
    GetNArray(b_vnary, b_nary);                                                                \
    if (NA_NDIM(b_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array b must be 2-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (NA_SHAPE(b_nary)[0] != NA_SHAPE(b_nary)[1]) {                                          \
      rb_raise(rb_eArgError, "input array b must be square");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    if (range == 'V' && vu <= vl) {                                                            \
      rb_raise(rb_eArgError, "vu must be greater than vl");                                    \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    const size_t n = NA_SHAPE(a_nary)[1];                                                      \
    if (range == 'I' && (il < 1 || il > (lapack_int)n)) {                                      \
      rb_raise(rb_eArgError, "il must satisfy 1 <= il <= n");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
    if (range == 'I' && (iu < 1 || iu > (lapack_int)n)) {                                      \
      rb_raise(rb_eArgError, "iu must satisfy 1 <= iu <= n");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
    if (range == 'I' && iu < il) {                                                             \
      rb_raise(rb_eArgError, "il must be less than or equal to iu");                           \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t m = range != 'I' ? n : (size_t)(iu - il + 1);                                       \
    size_t w_shape[1] = { m };                                                                 \
    size_t z_shape[2] = { n, m };                                                              \
    size_t ifail_shape[1] = { n };                                                             \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };                           \
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 },                                           \
                                 { tRtNAryClass, 1, w_shape },                                 \
                                 { tNAryClass, 2, z_shape },                                   \
                                 { numo_cInt32, 1, ifail_shape },                              \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 5, ain, aout };            \
    struct _hegvx_option_##tRtDType opt = {                                                    \
      matrix_layout, itype, jobz, range, uplo, vl, vu, il, iu                                  \
    };                                                                                         \
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);                                   \
    VALUE ret = rb_ary_new3(                                                                   \
      7, a_vnary, b_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1), rb_ary_entry(res, 2),   \
      rb_ary_entry(res, 3), rb_ary_entry(res, 4)                                               \
    );                                                                                         \
                                                                                               \
    RB_GC_GUARD(a_vnary);                                                                      \
    RB_GC_GUARD(b_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(lapack_complex_double, numo_cDComplex, double, numo_cDFloat, zhegvx)
DEF_LINALG_FUNC(lapack_complex_float, numo_cSComplex, float, numo_cSFloat, chegvx)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_hegvx(VALUE mLapack) {
  rb_define_module_function(mLapack, "zhegvx", _linalg_lapack_zhegvx, -1);
  rb_define_module_function(mLapack, "chegvx", _linalg_lapack_chegvx, -1);
}
