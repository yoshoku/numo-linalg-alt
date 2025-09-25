# frozen_string_literal: true

require 'test_helper'

class TestLinalgLapack < Minitest::Test # rubocop:disable Metrics/ClassLength
  def setup
    Numo::NArray.srand(53_196)
  end

  def test_lapack_dgeqrf_dorgqr
    ma = 3
    na = 2
    a = Numo::DFloat.new(ma, na).rand
    qr, tau, = Numo::Linalg::Lapack.dgeqrf(a.dup)
    r = qr.triu
    qq = Numo::DFloat.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::Linalg::Lapack.dorgqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::DFloat.new(mb, nb).rand
    qr, tau, = Numo::Linalg::Lapack.dgeqrf(b.dup)
    r = qr.triu
    q, = Numo::Linalg::Lapack.dorgqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-7)
  end

  def test_lapack_sgeqrf_sorgqr
    ma = 3
    na = 2
    a = Numo::SFloat.new(ma, na).rand
    qr, tau, = Numo::Linalg::Lapack.sgeqrf(a.dup)
    r = qr.triu
    qq = Numo::SFloat.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::Linalg::Lapack.sorgqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::SFloat.new(mb, nb).rand
    qr, tau, = Numo::Linalg::Lapack.sgeqrf(b.dup)
    r = qr.triu
    q, = Numo::Linalg::Lapack.sorgqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_b, :<, 1e-5)
  end

  def test_lapack_zgeqrf_zungqr
    ma = 3
    na = 2
    a = Numo::DComplex.new(ma, na).rand
    qr, tau, = Numo::Linalg::Lapack.zgeqrf(a.dup)
    r = qr.triu
    qq = Numo::DComplex.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::Linalg::Lapack.zungqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::DComplex.new(mb, nb).rand
    qr, tau, = Numo::Linalg::Lapack.zgeqrf(b.dup)
    r = qr.triu
    q, = Numo::Linalg::Lapack.zungqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-7)
  end

  def test_lapack_cgeqrf_cungqr
    ma = 3
    na = 2
    a = Numo::SComplex.new(ma, na).rand
    qr, tau, = Numo::Linalg::Lapack.cgeqrf(a.dup)
    r = qr.triu
    qq = Numo::SComplex.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::Linalg::Lapack.cungqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::SComplex.new(mb, nb).rand
    qr, tau, = Numo::Linalg::Lapack.cgeqrf(b.dup)
    r = qr.triu
    q, = Numo::Linalg::Lapack.cungqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_b, :<, 1e-5)
  end

  def test_lapack_dgeev
    a = Numo::DFloat.new(5, 5).rand - 0.5
    wr, wi, vl, vr, info = Numo::Linalg::Lapack.dgeev(a.dup, jobvl: 'V', jobvr: 'V')
    w = wr + (wi * 1.0i)
    img_ids = wi.gt(0).where
    u = Numo::DComplex.cast(vl)
    u[true, img_ids].imag = vl[true, img_ids + 1]
    u[true, img_ids + 1] = u[true, img_ids].conj
    vt = Numo::DComplex.cast(vr)
    vt[true, img_ids].imag = vr[true, img_ids + 1]
    vt[true, img_ids + 1] = vt[true, img_ids].conj
    error_u = (u.transpose.conj.dot(a) - w.diag.dot(u.transpose.conj)).abs.max
    error_vt = (a.dot(vt) - vt.dot(w.diag)).abs.max

    assert_equal(0, info)
    assert_operator(error_u, :<, 1e-7)
    assert_operator(error_vt, :<, 1e-7)
  end

  def test_lapack_sgeev
    a = Numo::SFloat.new(5, 5).rand - 0.5
    wr, wi, vl, vr, info = Numo::Linalg::Lapack.sgeev(a.dup, jobvl: 'V', jobvr: 'V')
    w = wr + (wi * 1.0i)
    img_ids = wi.gt(0).where
    u = Numo::SComplex.cast(vl)
    u[true, img_ids].imag = vl[true, img_ids + 1]
    u[true, img_ids + 1] = u[true, img_ids].conj
    vt = Numo::SComplex.cast(vr)
    vt[true, img_ids].imag = vr[true, img_ids + 1]
    vt[true, img_ids + 1] = vt[true, img_ids].conj
    error_u = (u.transpose.conj.dot(a) - w.diag.dot(u.transpose.conj)).abs.max
    error_vt = (a.dot(vt) - vt.dot(w.diag)).abs.max

    assert_equal(0, info)
    assert_operator(error_u, :<, 1e-5)
    assert_operator(error_vt, :<, 1e-5)
  end

  def test_lapack_zgeev
    a = Numo::DComplex.new(5, 5).rand - (0.5 + 0.2i)
    w, vl, vr, info = Numo::Linalg::Lapack.zgeev(a.dup, jobvl: 'V', jobvr: 'V')
    error_vl = (vl.transpose.conj.dot(a) - w.diag.dot(vl.transpose.conj)).abs.max
    error_vr = (a.dot(vr) - vr.dot(w.diag)).abs.max

    assert_equal(0, info)
    assert_operator(error_vl, :<, 1e-7)
    assert_operator(error_vr, :<, 1e-7)
  end

  def test_lapack_cgeev
    a = Numo::SComplex.new(5, 5).rand - (0.5 + 0.2i)
    w, vl, vr, info = Numo::Linalg::Lapack.cgeev(a.dup, jobvl: 'V', jobvr: 'V')
    error_vl = (vl.transpose.conj.dot(a) - w.diag.dot(vl.transpose.conj)).abs.max
    error_vr = (a.dot(vr) - vr.dot(w.diag)).abs.max

    assert_equal(0, info)
    assert_operator(error_vl, :<, 1e-5)
    assert_operator(error_vr, :<, 1e-5)
  end

  def test_lapack_dgesv
    a = Numo::DFloat.new(5, 5).rand
    b = Numo::DFloat.new(5).rand
    c = Numo::DFloat.new(5, 5).rand
    d = Numo::DFloat.new(5, 3).rand
    ret_ab = Numo::Linalg::Lapack.dgesv(a.dup, b.dup)
    ret_cd = Numo::Linalg::Lapack.dgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert_operator(error_ab, :<, 1e-7)
    assert_operator(error_cd, :<, 1e-7)
  end

  def test_lapack_sgesv
    a = Numo::SFloat.new(3, 3).rand
    b = Numo::SFloat.new(3).rand
    c = Numo::SFloat.new(3, 3).rand
    d = Numo::SFloat.new(3, 5).rand
    ret_ab = Numo::Linalg::Lapack.sgesv(a.dup, b.dup)
    ret_cd = Numo::Linalg::Lapack.sgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert_operator(error_ab, :<, 1e-5)
    assert_operator(error_cd, :<, 1e-5)
  end

  def test_lapack_zgesv
    a = Numo::DComplex.new(5, 5).rand
    b = Numo::DComplex.new(5).rand
    c = Numo::DComplex.new(5, 5).rand
    d = Numo::DComplex.new(5, 3).rand
    ret_ab = Numo::Linalg::Lapack.zgesv(a.dup, b.dup)
    ret_cd = Numo::Linalg::Lapack.zgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert_operator(error_ab, :<, 1e-7)
    assert_operator(error_cd, :<, 1e-7)
  end

  def test_lapack_cgesv
    a = Numo::SComplex.new(3, 3).rand
    b = Numo::SComplex.new(3).rand
    c = Numo::SComplex.new(3, 3).rand
    d = Numo::SComplex.new(3, 5).rand
    ret_ab = Numo::Linalg::Lapack.cgesv(a.dup, b.dup)
    ret_cd = Numo::Linalg::Lapack.cgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert_operator(error_ab, :<, 1e-5)
    assert_operator(error_cd, :<, 1e-5)
  end

  def test_lapack_dgesvd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.dgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgesvd
    x = Numo::SFloat.new(5, 3).rand.dot(Numo::SFloat.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.sgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgesvd
    x = Numo::DComplex.new(5, 3).rand.dot(Numo::DComplex.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.zgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgesvd
    x = Numo::SComplex.new(5, 3).rand.dot(Numo::SComplex.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.cgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dgesdd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.dgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgesdd
    x = Numo::SFloat.new(5, 3).rand.dot(Numo::SFloat.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.sgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgesdd
    x = Numo::DComplex.new(5, 3).rand.dot(Numo::DComplex.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.zgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgesdd
    x = Numo::SComplex.new(5, 3).rand.dot(Numo::SComplex.new(3, 2).rand)
    s, u, vt, = Numo::Linalg::Lapack.cgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dgetrf
    nr = 3
    nc = 2
    a = Numo::DFloat.new(nr, nc).rand
    lu, piv, = Numo::Linalg::Lapack.dgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::DFloat.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgetrf
    nr = 3
    nc = 5
    a = Numo::SFloat.new(nr, nc).rand
    lu, piv, = Numo::Linalg::Lapack.sgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }[0...nr, 0...nr]
    u = lu.triu[0...nr, 0...nc]
    pm = Numo::SFloat.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgetrf
    nr = 3
    nc = 2
    a = Numo::DComplex.new(nr, nc).rand
    lu, piv, = Numo::Linalg::Lapack.zgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::DComplex.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgetrf
    nr = 3
    nc = 2
    a = Numo::SComplex.new(nr, nc).rand
    lu, piv, = Numo::Linalg::Lapack.cgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::SComplex.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dgetri
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.dgetrf(a.dup)
    a_inv, = Numo::Linalg::Lapack.dgetri(lu, piv)
    error = (Numo::DFloat.eye(n) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgetri
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.sgetrf(a.dup)
    a_inv, = Numo::Linalg::Lapack.sgetri(lu, piv)
    error = (Numo::SFloat.eye(n) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgetri
    n = 3
    a = Numo::DComplex.new(n, n).rand
    lu, piv, = Numo::Linalg::Lapack.zgetrf(a.dup)
    a_inv, = Numo::Linalg::Lapack.zgetri(lu, piv)
    error = (Numo::DComplex.eye(n) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgetri
    n = 3
    a = Numo::SComplex.new(n, n).rand
    lu, piv, = Numo::Linalg::Lapack.cgetrf(a.dup)
    a_inv, = Numo::Linalg::Lapack.cgetri(lu, piv)
    error = (Numo::SComplex.eye(n) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dgetrs
    n = 5
    a = Numo::DFloat.new(n, n).rand
    b = Numo::DFloat.new(n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.dgetrf(a.dup)
    x, info = Numo::Linalg::Lapack.dgetrs(lu, piv, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgetrs
    n = 5
    a = Numo::SFloat.new(n, n).rand
    b = Numo::SFloat.new(n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.sgetrf(a.dup)
    x, info = Numo::Linalg::Lapack.sgetrs(lu, piv, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgetrs
    n = 5
    a = Numo::DComplex.new(n, n).rand
    b = Numo::DComplex.new(n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.zgetrf(a.dup)
    x, info = Numo::Linalg::Lapack.zgetrs(lu, piv, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgetrs
    n = 5
    a = Numo::SComplex.new(n, n).rand
    b = Numo::SComplex.new(n).rand - 0.5
    lu, piv, = Numo::Linalg::Lapack.cgetrf(a.dup)
    x, info = Numo::Linalg::Lapack.cgetrs(lu, piv, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dtrtrs
    n = 5
    a = Numo::DFloat.new(n, n).rand.triu
    b = Numo::DFloat.new(n).rand - 0.5
    x, info = Numo::Linalg::Lapack.dtrtrs(a, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_strtrs
    n = 5
    a = Numo::SFloat.new(n, n).rand.tril
    b = Numo::SFloat.new(n).rand - 0.5
    x, info = Numo::Linalg::Lapack.strtrs(a, b.dup, uplo: 'L')
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_ztrtrs
    n = 5
    a = Numo::DComplex.new(n, n).rand.triu
    b = Numo::DComplex.new(n).rand - 0.5
    x, info = Numo::Linalg::Lapack.ztrtrs(a, b.dup)
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_ctrtrs
    n = 5
    a = Numo::SComplex.new(n, n).rand.tril
    b = Numo::SComplex.new(n).rand - 0.5
    x, info = Numo::Linalg::Lapack.ctrtrs(a, b.dup, uplo: 'L')
    error = (b - a.dot(x)).abs.max

    assert_equal(0, info)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dpotrf
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    b = a.transpose.dot(a)
    c, _info = Numo::Linalg::Lapack.dpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.dot(cu)).abs.max

    assert_operator(error, :<, 1e-7)

    c, _info = Numo::Linalg::Lapack.dpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_spotrf
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    b = a.transpose.dot(a)
    c, _info = Numo::Linalg::Lapack.spotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.dot(cu)).abs.max

    assert_operator(error, :<, 1e-5)

    c, _info = Numo::Linalg::Lapack.spotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zpotrf
    n = 3
    a = Numo::DComplex.new(n, n).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    c, _info = Numo::Linalg::Lapack.zpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.conjugate.dot(cu)).abs.max

    assert_operator(error, :<, 1e-7)

    c, _info = Numo::Linalg::Lapack.zpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cpotrf
    n = 3
    a = Numo::SComplex.new(n, n).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    c, _info = Numo::Linalg::Lapack.cpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.conjugate.dot(cu)).abs.max

    assert_operator(error, :<, 1e-5)

    c, _info = Numo::Linalg::Lapack.cpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dpotrs
    n = 5
    a = Numo::DFloat.new(n, n).rand
    c = a.transpose.dot(a)
    b = Numo::DFloat.new(n).rand - 0.5
    f, = Numo::Linalg::Lapack.dpotrf(c.dup)
    x, _info = Numo::Linalg::Lapack.dpotrs(f, b.dup)
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_spotrs
    n = 5
    a = Numo::DFloat.new(n, n).rand
    c = a.transpose.dot(a)
    b = Numo::DFloat.new(n).rand - 0.5
    f, = Numo::Linalg::Lapack.spotrf(c.dup, uplo: 'L')
    x, _info = Numo::Linalg::Lapack.spotrs(f, b.dup, uplo: 'L')
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zpotrs
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::DComplex.new(n).rand - 0.5
    f, = Numo::Linalg::Lapack.zpotrf(c.dup, uplo: 'L')
    x, _info = Numo::Linalg::Lapack.zpotrs(f, b.dup, uplo: 'L')
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cpotrs
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::SComplex.new(n).rand - 0.5
    f, = Numo::Linalg::Lapack.cpotrf(c.dup)
    x, _info = Numo::Linalg::Lapack.cpotrs(f, b.dup)
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsyev
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::Linalg::Lapack.dsyev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_ssyev
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::Linalg::Lapack.ssyev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zheev
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::Linalg::Lapack.zheev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cheev
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::Linalg::Lapack.cheev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsyevd
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::Linalg::Lapack.dsyevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_ssyevd
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::Linalg::Lapack.ssyevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zheevd
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::Linalg::Lapack.zheevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cheevd
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::Linalg::Lapack.cheevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsyevr
    m = 3
    n = 5
    a = Numo::DFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    _c, _mm, w, v, _isuppz, _info = Numo::Linalg::Lapack.dsyevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _c, mi, w, v, _isuppz, _info = Numo::Linalg::Lapack.dsyevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _c, mv, w, v, _isuppz, _info = Numo::Linalg::Lapack.dsyevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_i, :<, 1e-7)
    # assert(error_v < 1e-7)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_ssyevr
    m = 3
    n = 5
    a = Numo::SFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    _c, _mm, w, v, _isuppz, _info = Numo::Linalg::Lapack.ssyevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _c, mi, w, v, _isuppz, _info = Numo::Linalg::Lapack.ssyevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _c, mv, w, v, _isuppz, _info = Numo::Linalg::Lapack.ssyevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_i, :<, 1e-5)
    # assert(error_v < 1e-5)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_zheevr
    m = 3
    n = 5
    a = Numo::DComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    _c, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.zheevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _c, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.zheevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _c, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.zheevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_i, :<, 1e-7)
    # assert(error_v < 1e-7)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_cheevr
    m = 3
    n = 5
    a = Numo::SComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    _c, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.cheevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _c, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.cheevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _c, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.chegvx(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_i, :<, 1e-5)
    # assert(error_v < 1e-5)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_dsygv
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::DFloat.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.dsygv(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_ssygv
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::SFloat.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.ssygv(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zhegv
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::DFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.zhegv(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_chegv
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::SFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.chegv(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsygvd
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::DFloat.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.dsygvd(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_ssygvd
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::SFloat.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.ssygvd(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zhegvd
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::DFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.zhegvd(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_chegvd
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::SFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::Linalg::Lapack.chegvd(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsygvx
    m = 3
    n = 5
    a = Numo::DFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    b = Numo::DFloat.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.dsygvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.dsygvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.dsygvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_i, :<, 1e-7)
    # assert(error_v < 1e-7)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_ssygvx
    m = 3
    n = 5
    a = Numo::SFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    b = Numo::SFloat.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.ssygvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.ssygvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.ssygvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_i, :<, 1e-5)
    # assert(error_v < 1e-5)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_zhegvx
    m = 3
    n = 5
    a = Numo::DComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::DComplex.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.zhegvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.zhegvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.zhegvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_i, :<, 1e-7)
    # assert(error_v < 1e-7)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_chegvx
    m = 3
    n = 5
    a = Numo::SComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::SComplex.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::Linalg::Lapack.chegvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::Linalg::Lapack.chegvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::Linalg::Lapack.chegvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert_operator(error_a, :<, 1e-5)
    assert_operator(error_i, :<, 1e-5)
    # assert(error_v < 1e-5)
    assert_operator(mi, :<, n)
    # assert(mv < n)
  end

  def test_lapack_dlange
    a = Numo::DFloat.new(4, 3).rand - 0.5
    norm = Numo::Linalg::Lapack.dlange(a)
    error_f = (norm - Math.sqrt(a.dot(a.transpose).trace)).abs
    norm = Numo::Linalg::Lapack.dlange(a, norm: 'M')
    error_m = (norm - a.abs.max).abs

    assert_operator(error_f, :<, 1e-7)
    assert_operator(error_m, :<, 1e-7)
  end

  def test_lapack_slange
    a = Numo::SFloat.new(4, 3).rand - 0.5
    norm = Numo::Linalg::Lapack.dlange(a)
    error_f = (norm - Math.sqrt(a.dot(a.transpose).trace)).abs
    norm = Numo::Linalg::Lapack.dlange(a, norm: 'M')
    error_m = (norm - a.abs.max).abs

    assert_operator(error_f, :<, 1e-5)
    assert_operator(error_m, :<, 1e-5)
  end

  def test_lapack_zlange
    a = Numo::DComplex.new(4, 3).rand - 0.5
    norm = Numo::Linalg::Lapack.zlange(a)
    error_f = (norm - Math.sqrt(a.dot(a.transpose.conjugate).trace)).abs
    norm = Numo::Linalg::Lapack.zlange(a, norm: 'M')
    error_m = (norm - a.abs.max).abs

    assert_operator(error_f, :<, 1e-7)
    assert_operator(error_m, :<, 1e-7)
  end

  def test_lapack_clange
    a = Numo::SComplex.new(4, 3).rand - 0.5
    norm = Numo::Linalg::Lapack.clange(a)
    error_f = (norm - Math.sqrt(a.dot(a.transpose.conjugate).trace)).abs
    norm = Numo::Linalg::Lapack.clange(a, norm: 'M')
    error_m = (norm - a.abs.max).abs

    assert_operator(error_f, :<, 1e-5)
    assert_operator(error_m, :<, 1e-5)
  end

  def test_lapack_dgelsd
    a = Numo::DFloat.new(3, 8).rand
    b = Numo::DFloat.new(3).rand
    x = Numo::DFloat.zeros(a.shape[1])
    x[0...b.size] = b
    s, rank, info = Numo::Linalg::Lapack.dgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-7)

    b = Numo::DFloat.new(3, 2).rand
    x = Numo::DFloat.zeros(a.shape[1], b.shape[1])
    x[0...b.shape[0], 0...b.shape[1]] = b
    s, rank, info = Numo::Linalg::Lapack.dgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_sgelsd
    a = Numo::SFloat.new(3, 8).rand
    b = Numo::SFloat.new(3).rand
    x = Numo::SFloat.zeros(a.shape[1])
    x[0...b.size] = b
    s, rank, info = Numo::Linalg::Lapack.sgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_zgelsd
    a = Numo::DComplex.new(3, 8).rand
    b = Numo::DComplex.new(3).rand
    x = Numo::DComplex.zeros(a.shape[1])
    x[0...b.size] = b
    s, rank, info = Numo::Linalg::Lapack.zgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-7)

    b = Numo::DComplex.new(3, 2).rand
    x = Numo::DComplex.zeros(a.shape[1], b.shape[1])
    x[0...b.shape[0], 0...b.shape[1]] = b
    s, rank, info = Numo::Linalg::Lapack.zgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-7)
  end

  def test_lapack_cgelsd
    a = Numo::SComplex.new(3, 8).rand
    b = Numo::SComplex.new(3).rand
    x = Numo::SComplex.zeros(a.shape[1])
    x[0...b.size] = b
    s, rank, info = Numo::Linalg::Lapack.cgelsd(a.dup, x)
    error = (b - a.dot(x)).abs.max

    assert_equal(1, s.ndim)
    assert_equal(3, s.size)
    assert_equal(0, info)
    assert_equal(3, rank)
    assert_operator(error, :<, 1e-5)
  end

  def test_lapack_dsytrf
    n = 10
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)

    # --- uplo: 'U'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.dsytrf(lud)
    u, d = sytrf_permutation_u(lud, ipiv)
    error = (a - u.dot(d).dot(u.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(0, info)

    # --- uplo: 'L'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.dsytrf(lud, uplo: 'L')
    l, d = sytrf_permutation_l(lud, ipiv)
    error = (a - l.dot(d).dot(l.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(0, info)
  end

  def test_lapack_ssytrf
    n = 10
    a = Numo::SFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)

    # --- uplo: 'U'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.ssytrf(lud)
    u, d = sytrf_permutation_u(lud, ipiv)
    error = (a - u.dot(d).dot(u.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
    assert_equal(0, info)

    # --- uplo: 'L'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.ssytrf(lud, uplo: 'L')
    l, d = sytrf_permutation_l(lud, ipiv)
    error = (a - l.dot(d).dot(l.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
    assert_equal(0, info)
  end

  def test_lapack_zsytrf
    n = 10
    a = Numo::DComplex.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)

    # --- uplo: 'U'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.zsytrf(lud)
    u, d = sytrf_permutation_u(lud, ipiv)
    error = (a - u.dot(d).dot(u.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(0, info)

    # --- uplo: 'L'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.zsytrf(lud, uplo: 'L')
    l, d = sytrf_permutation_l(lud, ipiv)
    error = (a - l.dot(d).dot(l.transpose)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(0, info)
  end

  def test_lapack_csytrf
    n = 10
    a = Numo::SComplex.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)

    # --- uplo: 'U'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.csytrf(lud)
    u, d = sytrf_permutation_u(lud, ipiv)
    error = (a - u.dot(d).dot(u.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
    assert_equal(0, info)

    # --- uplo: 'L'
    lud = a.dup
    ipiv, info = Numo::Linalg::Lapack.csytrf(lud, uplo: 'L')
    l, d = sytrf_permutation_l(lud, ipiv)
    error = (a - l.dot(d).dot(l.transpose)).abs.max

    assert_operator(error, :<, 1e-5)
    assert_equal(0, info)
  end

  def sytrf_permutation_u(lud, ipiv)
    n = lud.shape[0]
    u = lud.triu.tap { |m| m[m.diag_indices] = 1 }
    d = lud.class.zeros(n, n)
    # If IPIV(k) > 0, then rows and columns k and IPIV(k) were interchanged
    # and D(k,k) is a 1-by-1 diagonal block.
    # IF UPLO = 'U' and If IPIV(k) = IPIV(k-1) < 0, then
    # rows and columns k-1 and -IPIV(k) were interchanged
    # and D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
    skip = false
    n.times do |k|
      d[k, k] = lud[k, k]
      if ipiv[k].positive?
        i = ipiv[k] - 1
        u[[i, k], 0..k] = u[[k, i], 0..k].dup
      elsif k.positive? && ipiv[k].negative? && ipiv[k] == ipiv[k - 1] && !skip
        i = -ipiv[k] - 1
        d[k - 1, k] = lud[k - 1, k]
        d[k, k - 1] = d[k - 1, k]
        u[k - 1, k] = 0
        u[[i, k - 1], 0..k] = u[[k - 1, i], 0..k].dup
        skip = true
        next
      end
      skip = false if skip
    end
    [u, d]
  end

  def sytrf_permutation_l(lud, ipiv)
    n = lud.shape[0]
    l = lud.tril.tap { |m| m[m.diag_indices] = 1 }
    d = lud.class.zeros(n, n)
    # If UPLO = 'L' and IPIV(k) = IPIV(k+1) < 0, then
    # rows and columns k+1 and -IPIV(k) were interchanged
    # and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
    skip = false
    (n - 1).downto(0) do |k|
      d[k, k] = lud[k, k]
      if ipiv[k].positive?
        i = ipiv[k] - 1
        l[[i, k], k...n] = l[[k, i], k...n].dup
      elsif k < n - 1 && ipiv[k].negative? && ipiv[k] == ipiv[k + 1] && !skip
        i = -ipiv[k] - 1
        d[k + 1, k] = lud[k + 1, k]
        d[k, k + 1] = d[k + 1, k]
        l[k + 1, k] = 0
        l[[i, k + 1], k...n] = l[[k + 1, i], k...n].dup
        skip = true
        next
      end
      skip = false if skip
    end
    [l, d]
  end
end
