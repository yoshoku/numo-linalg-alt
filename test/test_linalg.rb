# frozen_string_literal: true

require 'test_helper'

class TestLinalg < Minitest::Test # rubocop:disable Metrics/ClassLength
  def setup
    Numo::NArray.srand(53_196)
  end

  def test_that_it_has_a_version_number
    refute_nil ::Numo::Linalg::VERSION
  end

  def test_blas_char
    assert_equal 'd', Numo::Linalg.blas_char([true, false])
    assert_equal 'd', Numo::Linalg.blas_char([1, 2])
    assert_equal 'd', Numo::Linalg.blas_char([1.1, 2.2])
    assert_equal 'z', Numo::Linalg.blas_char([Complex(1, 2), 3])
    assert_equal 'd', Numo::Linalg.blas_char(Numo::NArray[1, 2])
    assert_equal 's', Numo::Linalg.blas_char(Numo::SFloat[1.1, 2.2])
    assert_equal 'd', Numo::Linalg.blas_char(Numo::DFloat[1.1, 2.2])
    assert_equal 'c', Numo::Linalg.blas_char(Numo::SComplex[1.1, 2.2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DComplex[1.1, 2.2])
    assert_equal 'd', Numo::Linalg.blas_char(Numo::SFloat[1, 2], Numo::DFloat[1, 2])
    assert_equal 'c', Numo::Linalg.blas_char(Numo::SFloat[1, 2], Numo::SComplex[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::SFloat[1, 2], Numo::DComplex[1, 2])
    assert_equal 'd', Numo::Linalg.blas_char(Numo::DFloat[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DFloat[1, 2], Numo::SComplex[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DFloat[1, 2], Numo::DComplex[1, 2])
    assert_equal 'c', Numo::Linalg.blas_char(Numo::SComplex[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::SComplex[1, 2], Numo::DFloat[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::SComplex[1, 2], Numo::DComplex[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DComplex[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DComplex[1, 2], Numo::DFloat[1, 2])
    assert_equal 'z', Numo::Linalg.blas_char(Numo::DComplex[1, 2], Numo::SComplex[1, 2])
    assert_raises TypeError, 'invalid data type for BLAS/LAPACK' do
      Numo::Linalg.blas_char(['1', 2, 3])
    end
  end

  def test_eigh
    m = 3
    n = 5
    x = Numo::DFloat.new(m, n).rand - 0.5
    a = x.transpose.dot(x)
    v, w = Numo::Linalg.eigh(a)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    v, w = Numo::Linalg.eigh(a, vals_range: (n - m)...n)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    x = Numo::DComplex.new(m, n).rand - 0.5
    a = x.transpose.conjugate.dot(x)
    v, w = Numo::Linalg.eigh(a, turbo: true)
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    v, w = Numo::Linalg.eigh(a, vals_range: [n - m, n - 1])
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    x = Numo::DFloat.new(m, n).rand - 0.5
    y = Numo::DFloat.new(m, n).rand - 0.5
    a = x.transpose.dot(x)
    b = y.transpose.dot(y) + (n * Numo::DFloat.eye(n))
    v, w = Numo::Linalg.eigh(a, b)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    v, w = Numo::Linalg.eigh(a, b, vals_range: (n - m)...n)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    x = Numo::DComplex.new(m, n).rand - 0.5
    y = Numo::DComplex.new(m, n).rand - 0.5
    a = x.transpose.conjugate.dot(x)
    b = y.transpose.conjugate.dot(y) + (n * Numo::DComplex.eye(n))
    v, w = Numo::Linalg.eigh(a, b, turbo: true)
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.conjugate.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    v, w = Numo::Linalg.eigh(a, b, vals_range: [n - m, n - 1])
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.conjugate.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)
  end

  def test_norm
    # empty array
    assert_equal(0, Numo::Linalg.norm([]))
    assert_equal(0, Numo::Linalg.norm(Numo::DFloat[]))

    # vector
    a = Numo::DFloat[3, -4]
    b = Numo::DFloat[1, 0, 2, 0, 3]

    assert_equal(5, Numo::Linalg.norm(a))
    assert_equal(5, Numo::Linalg.norm(a, 2))
    assert_equal(7, Numo::Linalg.norm(a, 1))
    assert_equal(3, Numo::Linalg.norm(b, 0))
    assert_in_delta(2.4, Numo::Linalg.norm(a, -2))
    assert_equal(4, Numo::Linalg.norm(a, Float::INFINITY))
    assert_equal(3, Numo::Linalg.norm(a, -Float::INFINITY))
    assert_equal(4, Numo::Linalg.norm(a, 'inf'))
    assert_equal(3, Numo::Linalg.norm(a, '-inf'))
    assert_equal(Numo::DFloat[5], Numo::Linalg.norm(a, keepdims: true))
    assert_equal(Numo::DFloat[5], Numo::Linalg.norm(a, 2, keepdims: true))
    assert_equal(5, Numo::Linalg.norm(a, axis: 0))
    assert_match(/axis is out of range/, assert_raises(ArgumentError) { Numo::Linalg.norm(a, axis: 1) }.message)
    assert_match(/invalid axis/, assert_raises(ArgumentError) { Numo::Linalg.norm(a, axis: '1') }.message)

    # matrix
    a = Numo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]

    assert_equal(10, Numo::Linalg.norm(a))
    assert_equal(10, Numo::Linalg.norm(a, 'fro'))
    assert_operator((Numo::Linalg.norm(a, 'nuc') - 12.3643).abs, :<, 1e-4)
    assert_operator((Numo::Linalg.norm(a, 2) - 9.6144).abs, :<, 1e-4)
    assert_equal(11, Numo::Linalg.norm(a, 1))
    assert_match(/invalid ord/, assert_raises(ArgumentError) { Numo::Linalg.norm(a, 0) }.message)
    assert_equal(3, Numo::Linalg.norm(a, -1))
    assert_operator((Numo::Linalg.norm(a, -2) - 2.7498).abs, :<, 1e-4)
    assert_equal(15, Numo::Linalg.norm(a, Float::INFINITY))
    assert_equal(7, Numo::Linalg.norm(a, -Float::INFINITY))
    assert_equal(15, Numo::Linalg.norm(a, 'inf'))
    assert_equal(7, Numo::Linalg.norm(a, '-inf'))
    assert_equal(Numo::DFloat[[10]], Numo::Linalg.norm(a, keepdims: true))
    assert_equal(Numo::DFloat[5, 3, 11, 3], Numo::Linalg.norm(a, 1, axis: 0))
    assert_equal(Numo::DFloat[7, 15], Numo::Linalg.norm(a, 1, axis: 1))
    assert_equal(Numo::DFloat[[5, 3, 11, 3]], Numo::Linalg.norm(a, 1, axis: 0, keepdims: true))
    assert_equal(Numo::DFloat[[7], [15]], Numo::Linalg.norm(a, 1, axis: 1, keepdims: true))
    assert_equal(10, Numo::Linalg.norm(a, 'fro', axis: [0, 1]))
    assert_equal(11, Numo::Linalg.norm(a, 1, axis: [0, 1]))
    assert_equal(Numo::DFloat[[15]], Numo::Linalg.norm(a, Float::INFINITY, axis: [0, 1], keepdims: true))

    # tensor
    a = Numo::DFloat[[[2, 3, 1], [1, 2, 4]], [[2, 2, 3], [3, 2, 4]]]

    assert_equal(9, Numo::Linalg.norm(a))
    assert_equal(Numo::DFloat[[[9]]], Numo::Linalg.norm(a, keepdims: true))
  end

  def test_cholesky
    a = Numo::DFloat.new(3, 3).rand - 0.5
    b = a.transpose.dot(a)
    u = Numo::Linalg.cholesky(b)
    error = (b - u.transpose.dot(u)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::SComplex.new(3, 3).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    l = Numo::Linalg.cholesky(b, uplo: 'L')
    error = (b - l.dot(l.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_cho_solve
    a = Numo::DFloat.new(3, 3).rand - 0.5
    c = a.transpose.dot(a)
    u = Numo::Linalg.cholesky(c)
    b = Numo::DFloat.new(3, 2).rand - 0.5
    x = Numo::Linalg.cho_solve(u, b)
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_det
    a = Numo::DFloat[[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    error = (Numo::Linalg.det(a) - 3).abs

    assert_operator(error, :<, 1e-7)
  end

  def test_inv
    a = Numo::DFloat.new(3, 3).rand - 0.5
    a_inv = Numo::Linalg.inv(a)
    error = (Numo::DFloat.eye(3) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_pinv
    a = Numo::DComplex.new(5, 3).rand - 0.5
    a_inv = Numo::Linalg.pinv(a)
    error = (Numo::DComplex.eye(3) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_polar
    a = Numo::DComplex.new(3, 3).rand - (0.5 + 0.2i)
    u, pmat = Numo::Linalg.polar(a)
    error_r = (a - u.dot(pmat)).abs.max
    u, pmat = Numo::Linalg.polar(a, side: 'left')
    error_l = (a - pmat.dot(u)).abs.max

    assert_operator(error_r, :<, 1e-7)
    assert_operator(error_l, :<, 1e-7)

    a = Numo::DFloat.new(2, 3).rand - 0.5
    u, pmat = Numo::Linalg.polar(a)
    error_r = (a - u.dot(pmat)).abs.max
    error = (u.dot(u.transpose) - Numo::DFloat.eye(2)).abs.sum

    assert_operator(error_r, :<, 1e-7)
    assert_operator(error, :<, 1e-7)
  end

  def test_qr
    ma = 5
    na = 3
    a = Numo::DFloat.new(ma, na).rand - 0.5
    q, r = Numo::Linalg.qr(a, mode: 'economic')
    error_a = (a - q.dot(r)).abs.max

    mb = 3
    nb = 5
    b = Numo::DFloat.new(mb, nb).rand - 0.5
    q, r = Numo::Linalg.qr(b, mode: 'economic')
    error_b = (b - q.dot(r)).abs.max

    mc = 5
    nc = 3
    c = Numo::DComplex.new(mc, nc).rand - 0.5
    q, r = Numo::Linalg.qr(c, mode: 'economic')
    error_c = (c - q.dot(r)).abs.max

    md = 3
    nd = 5
    d = Numo::DComplex.new(md, nd).rand - 0.5
    q, r = Numo::Linalg.qr(d, mode: 'economic')
    error_d = (d - q.dot(r)).abs.max

    q, r = Numo::Linalg.qr(a, mode: 'reduce')

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-7)
    assert_operator(error_c, :<, 1e-7)
    assert_operator(error_d, :<, 1e-7)
    assert_equal(q.shape, [ma, ma])
    assert_equal(r.shape, [ma, na])
  end

  def test_rq
    ma = 5
    na = 3
    a = Numo::DFloat.new(ma, na).rand - 0.5
    r, q = Numo::Linalg.rq(a)
    error_a = (a - r.dot(q)).abs.max

    assert_equal([ma, na], r.shape)
    assert_equal([na, na], q.shape)
    assert_operator(error_a, :<, 1e-7)

    mb = 3
    nb = 5
    b = Numo::DFloat.new(mb, nb).rand - 0.5
    r, q = Numo::Linalg.rq(b)
    error_b = (b - r.dot(q)).abs.max

    assert_equal([mb, nb], r.shape)
    assert_equal([nb, nb], q.shape)
    assert_operator(error_b, :<, 1e-7)

    mc = 5
    nc = 3
    c = Numo::DFloat.new(mc, nc).rand - 0.5
    r, q = Numo::Linalg.rq(c, mode: 'economic')
    error_c = (c - r.dot(q)).abs.max

    assert_equal([mc, nc], r.shape)
    assert_equal([nc, nc], q.shape)
    assert_operator(error_c, :<, 1e-7)

    md = 3
    nd = 5
    d = Numo::DFloat.new(md, nd).rand - 0.5
    r, q = Numo::Linalg.rq(d, mode: 'economic')
    error_d = (d - r.dot(q)).abs.max

    assert_equal([md, md], r.shape)
    assert_equal([md, nd], q.shape)
    assert_operator(error_d, :<, 1e-7)

    r = Numo::Linalg.rq(a, mode: 'r')

    assert_equal([ma, na], r.shape)

    r = Numo::Linalg.rq(b, mode: 'r')

    assert_equal([mb, nb], r.shape)
  end

  def test_qz
    a = Numo::DComplex.new(3, 3).rand - (0.5 + 0.2i)
    b = Numo::DComplex.new(3, 3).rand - (0.5 + 0.2i)
    aa, bb, q, z = Numo::Linalg.qz(a, b)

    error_a = (a - q.dot(aa).dot(z.transpose.conjugate)).abs.max
    error_b = (b - q.dot(bb).dot(z.transpose.conjugate)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-7)
  end

  def test_schur
    a = Numo::DComplex.new(3, 3).rand - (0.5 + 0.2i)
    t, z, = Numo::Linalg.schur(a)

    error = (z.dot(t).dot(z.transpose.conjugate) - a).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_solve
    a = Numo::DComplex.new(3, 3).rand
    b = Numo::SFloat.new(3).rand
    x = Numo::Linalg.solve(a, b)
    error_ab = (b - a.dot(x)).abs.max

    assert_operator(error_ab, :<, 1e-7)
  end

  def test_solve_triangular
    a = Numo::SFloat.new(3, 3).rand.triu
    b = Numo::DComplex.new(3).rand
    x = Numo::Linalg.solve_triangular(a, b)
    error_ab = (b - a.dot(x)).abs.max

    assert_operator(error_ab, :<, 1e-7)
  end

  def test_svd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::Linalg.svd(x, driver: 'sdd', job: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_dot
    a = Numo::DFloat.new(3).rand
    b = Numo::SFloat.new(3).rand
    c = Numo::DFloat.new(3, 2).rand
    d = Numo::SFloat.new(2, 3).rand
    error_ab = (a.dot(b) - Numo::Linalg.dot(a, b)).abs
    error_ac = (a.dot(c) - Numo::Linalg.dot(a, c)).abs.max
    error_cb = (c.transpose.dot(b) - Numo::Linalg.dot(c.transpose, b)).abs.max
    error_cd = (c.dot(d) - Numo::Linalg.dot(c, d)).abs.max

    assert_operator(error_ab, :<, 1e-7)
    assert_operator(error_ac, :<, 1e-7)
    assert_operator(error_cb, :<, 1e-7)
    assert_operator(error_cd, :<, 1e-7)
  end

  def test_matmul
    a = Numo::DFloat[[1, 0], [0, 1]]
    b = Numo::DFloat[[4, 1], [2, 2]]
    error_ab = (Numo::DFloat[[4, 1], [2, 2]] - Numo::Linalg.matmul(a, b)).abs.max

    assert_operator(error_ab, :<, 1e-7)
    assert_match(/must be 2-d/, assert_raises(ArgumentError) do
      Numo::Linalg.matmul(Numo::DFloat[1, 2], Numo::DFloat[3, 4])
    end.message)
  end

  def test_matrix_power
    a = Numo::DFloat[[1, 2], [3, 4]]
    error_a2 = (Numo::DFloat[[7, 10], [15, 22]] - Numo::Linalg.matrix_power(a, 2)).abs.max
    error_a3 = (Numo::DFloat[[37, 54], [81, 118]] - Numo::Linalg.matrix_power(a, 3)).abs.max
    error_a_neg2 = (Numo::DFloat[[5.5, -2.5], [-3.75, 1.75]] - Numo::Linalg.matrix_power(a, -2)).abs.max
    error_a_neg3 = (Numo::DFloat[[-14.75, 6.75], [10.125, -4.625]] - Numo::Linalg.matrix_power(a, -3)).abs.max

    assert_operator(error_a2, :<, 1e-7)
    assert_operator(error_a3, :<, 1e-7)
    assert_operator(error_a_neg2, :<, 1e-7)
    assert_operator(error_a_neg3, :<, 1e-7)
    assert_match(/must be 2-d/, assert_raises(Numo::NArray::ShapeError) do
      Numo::Linalg.matrix_power(Numo::DFloat[1, 2, 3], 2)
    end.message)
    assert_match(/must be square/, assert_raises(Numo::NArray::ShapeError) do
      Numo::Linalg.matrix_power(Numo::DFloat[[1, 2, 3], [4, 5, 6]], 2)
    end.message)
    assert_match(/must be an integer/, assert_raises(ArgumentError) do
      Numo::Linalg.matrix_power(Numo::DFloat[[1, 2], [3, 4]], 2.5)
    end.message)
  end

  def test_svdvals
    a = Numo::SFloat[[1, 2, 3], [2, 4, 6], [-1, 1, -1]]
    s = Numo::Linalg.svdvals(a)
    error = (Numo::SFloat[8.38434, 1.64402, 0] - s).abs.sum.fdiv(3)

    assert_operator(error, :<, 1e-5)
  end

  def test_orth
    a = Numo::DFloat[[1, 2, 3], [2, 4, 6], [-1, 1, -1]]
    u = Numo::Linalg.orth(a)
    error = (u.transpose.dot(u) - Numo::DFloat.eye(u.shape[1])).abs.max

    assert_operator(error, :<, 1e-7)
    assert_match(/must be 2-d/, assert_raises(Numo::NArray::ShapeError) do
      Numo::Linalg.orth(Numo::DFloat[1, 2, 3])
    end.message)
  end

  def test_null_space
    a = Numo::DFloat.new(3, 4).seq + 1
    n = Numo::Linalg.null_space(a)
    error = (n.transpose.dot(n) - Numo::DFloat.eye(n.shape[1])).abs.max

    assert_equal(4, n.shape[0])
    assert_equal(2, n.shape[1])
    assert_operator(error, :<, 1e-7)
  end

  def test_lu
    a = Numo::DFloat.new(5, 3).rand - 0.5
    pm, l, u = Numo::Linalg.lu(a)
    error = (pm.dot(a) - l.dot(u)).abs.max
    l, u = Numo::Linalg.lu(a, permute_l: true)
    error_perm = (a - l.dot(u)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_operator(error_perm, :<, 1e-7)
    assert_match(/must be 2-d/, assert_raises(Numo::NArray::ShapeError) do
      Numo::Linalg.orth(Numo::DFloat[1, 2, 3])
    end.message)
  end

  def test_lu_fact
    a = Numo::DFloat.new(5, 3).rand - 0.5
    lu, piv = Numo::Linalg.lu_fact(a)
    l = lu.tril.tap { |m| m[m.diag_indices(0)] = 1.0 }
    u = lu.triu[0...3, 0...3]
    pm = Numo::DFloat.eye(a.shape[0]).tap do |m|
      piv.to_a.each_with_index { |i, j| m[[i - 1, j], true] = m[[j, i - 1], true].dup }
    end
    error = (pm.dot(a) - l.dot(u)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_lu_solve
    a = Numo::DFloat.new(3, 3).rand - 0.5
    b = Numo::DFloat.new(3, 2).rand - 0.5
    lu, piv = Numo::Linalg.lu_fact(a)
    x = Numo::Linalg.lu_solve(lu, piv, b)
    error = (b - a.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_cho_fact
    a = Numo::DFloat.new(3, 3).rand - 0.5
    b = a.transpose.dot(a)
    u = Numo::Linalg.cho_fact(b).triu
    error = (b - u.transpose.dot(u)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::SComplex.new(3, 3).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    l = Numo::Linalg.cho_fact(b, uplo: 'L').tril
    error = (b - l.dot(l.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_orthogonal_procrustes
    a = Numo::DComplex.new(2, 3).rand - (0.5 + 0.2i)
    b = a.fliplr.dup
    r, scale = Numo::Linalg.orthogonal_procrustes(a, b)
    error = (b - a.dot(r)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Float, scale)
    assert_kind_of(Numo::DComplex, r)
    assert_equal(3, r.shape[0])
    assert_equal(3, r.shape[1])
  end

  def test_matrix_balance
    a = Numo::DFloat[[1, 0, 0], [1, 2, 0], [1, 2, 3]]

    b, h = Numo::Linalg.matrix_balance(a)
    error = (b - Numo::Linalg.inv(h).dot(a).dot(h)).abs.max

    assert_kind_of(Numo::DFloat, b)
    assert_kind_of(Numo::DFloat, h)
    assert_operator(error, :<, 1e-7)

    _, sc, pm = Numo::Linalg.matrix_balance(a, permute: true, scale: true, separate: true)

    assert_kind_of(Numo::DFloat, sc)
    assert_equal(Numo::DFloat[1, 1, 1], sc)
    assert_kind_of(Numo::Int32, pm)
    assert_equal(Numo::Int32[2, 1, 0], pm)

    _, _, pm = Numo::Linalg.matrix_balance(a, permute: false, scale: true, separate: true)

    assert_equal(Numo::Int32[0, 1, 2], pm)
  end

  def test_eig
    a = Numo::DComplex.new(3, 3).rand - (0.5 + 0.2i)
    w, vl, vr = Numo::Linalg.eig(a)
    error = (a.dot(vr) - vr.dot(w.diag)).abs.max

    assert_nil(vl)
    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat.new(5, 5).rand - 0.5
    w, vl, vr = Numo::Linalg.eig(a)
    error = (a.dot(vr) - vr.dot(w.diag)).abs.max

    assert_nil(vl)
    assert_operator(error, :<, 1e-7)

    a = Numo::SFloat.new(5, 5).rand - 0.5
    w, vl, vr = Numo::Linalg.eig(a, left: true, right: false)
    error = (vl.transpose.conjugate.dot(a) - w.diag.dot(vl.transpose.conjugate)).abs.max

    assert_nil(vr)
    assert_operator(error, :<, 1e-5)
  end

  def test_eigvals
    a = Numo::DFloat.new(5, 5).rand - 0.5
    w1, = Numo::Linalg.eig(a)
    w2 = Numo::Linalg.eigvals(a)
    error = (w1 - w2).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_eigvalsh
    m = 3
    n = 5
    x = Numo::DFloat.new(m, n).rand - 0.5
    a = x.dot(x.transpose)
    v = Numo::Linalg.eigvalsh(a)
    r = Numo::Linalg.eigh(a)[0]

    assert_operator((v - r).abs.max, :<, 1e-7)
  end

  def test_ldl
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a + a.transpose)
    u, d, perm = Numo::Linalg.ldl(a)
    error_u = (a - u.dot(d).dot(u.transpose)).abs.max
    sum_lower = u[perm, true].tril.sum

    assert_operator(error_u, :<, 1e-7)
    assert_equal(n, sum_lower)

    l, d, perm = Numo::Linalg.ldl(a, uplo: 'L')
    error_l = (a - l.dot(d).dot(l.transpose)).abs.max
    sum_upper = l[perm, true].triu.sum

    assert_operator(error_l, :<, 1e-7)
    assert_equal(n, sum_upper)

    b = Numo::DFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    a += b * Complex::I
    u, d, = Numo::Linalg.ldl(a)
    error = (a - u.dot(d).dot(u.transpose.conj)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_cond
    a = Numo::DFloat[[1, 2], [3, 4]]
    error_l2 = (Numo::Linalg.cond(a) - 14.933034).abs
    error_m2 = (Numo::Linalg.cond(a, -2) - 1.fdiv(14.933034)).abs
    error_fro = (Numo::Linalg.cond(a, 'fro') - 14.999999).abs
    error_l1 = (Numo::Linalg.cond(a, 1) - 20.999999).abs
    error_inf = (Numo::Linalg.cond(a, 'inf') - 20.999999).abs

    assert_operator(error_l2, :<, 1e-5)
    assert_operator(error_m2, :<, 1e-5)
    assert_operator(error_fro, :<, 1e-5)
    assert_operator(error_l1, :<, 1e-5)
    assert_operator(error_inf, :<, 1e-5)
  end

  def test_slogdet
    a = Numo::DFloat[[1, 2], [3, 4]]
    sign, logdet = Numo::Linalg.slogdet(a)
    error_sign = (sign - -1.0).abs
    error_logdet = (logdet - 0.693147).abs

    assert_operator(error_sign, :<, 1e-5)
    assert_operator(error_logdet, :<, 1e-5)

    b = Numo::DComplex[[1 - 2i, 2 + 1i], [3 + 4i, 4 - 3i]]
    sign, logdet = Numo::Linalg.slogdet(b)
    error_sign = (sign - (-0.178885 - 0.983869i)).abs
    error_logdet = (logdet - 3.107304).abs

    assert_operator(error_sign, :<, 1e-5)
    assert_operator(error_logdet, :<, 1e-5)
  end

  def test_matrix_rank
    m = Numo::DFloat.new(6, 2).rand - 0.5
    n = Numo::DFloat.new(2, 3).rand - 0.5
    a = m.dot(n)

    assert_equal(2, Numo::Linalg.matrix_rank(a))

    m = Numo::DComplex.new(6, 2).rand - 0.5
    n = Numo::DComplex.new(2, 3).rand - 0.5
    a = m.dot(n)

    assert_equal(2, Numo::Linalg.matrix_rank(a))
  end

  def test_lstsq
    # --- vec
    a = Numo::DFloat.new(5, 8).rand - 0.5
    b = Numo::DFloat.new(5).rand - 0.5
    x, r, rank, s = Numo::Linalg.lstsq(a, b)
    error = (b - a.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(1, x.ndim)
    assert_equal(8, x.size)
    assert_empty(r)
    assert_equal(5, rank)
    assert_equal(5, s.size)

    # --- mat
    a = Numo::DFloat.new(5, 8).rand - 0.5
    b = Numo::DFloat.new(5, 2).rand - 0.5
    x, r, rank, s = Numo::Linalg.lstsq(a, b)
    error = (b - a.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(2, x.ndim)
    assert_equal(8, x.shape[0])
    assert_equal(2, x.shape[1])
    assert_empty(r)
    assert_equal(5, rank)
    assert_equal(5, s.size)

    # --- m > n
    # --- vec
    a = Numo::DFloat.new(5, 3).rand - 0.5
    b = Numo::DFloat.new(5).rand - 0.5
    x, r, rank, s = Numo::Linalg.lstsq(a, b)
    diff = b - a.dot(x)
    error = r - diff.dot(diff)

    assert_operator(error, :<, 1e-7)
    assert_equal(1, x.ndim)
    assert_equal(3, x.size)
    assert_kind_of(Float, r)
    assert_equal(3, rank)
    assert_equal(3, s.size)

    # --- mat
    a = Numo::DFloat.new(5, 3).rand - 0.5
    b = Numo::DFloat.new(5, 2).rand - 0.5
    x, r, rank, s = Numo::Linalg.lstsq(a, b)
    diff = b - a.dot(x)
    error = (r - (diff * diff).sum(axis: 0)).abs.max

    assert_operator(error, :<, 1e-7)
    assert_equal(2, x.ndim)
    assert_equal(3, x.shape[0])
    assert_equal(2, x.shape[1])
    assert_kind_of(Numo::DFloat, r)
    assert_equal(1, r.ndim)
    assert_equal(2, r.size)
    assert_equal(3, rank)
    assert_equal(3, s.size)
  end

  def test_expm
    error = (Numo::Linalg.expm(Numo::DFloat.zeros(3, 3)) - Numo::DFloat.eye(3)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat[1, 2, 3]
    expected = Numo::NMath.exp(a).diag
    error = (expected - Numo::Linalg.expm(a.diag)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat[[1, 2], [0, 1]]
    e_one = Math.exp(1)
    expected = Numo::DFloat[[e_one, 2 * e_one], [0, e_one]]
    error = (expected - Numo::Linalg.expm(a)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat[[-156, 78], [-189, 95]]
    expected = Numo::DFloat[[-5.16765957, 4.29962817], [-10.41832981, 8.6683234]]
    error = (expected - Numo::Linalg.expm(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_sinm
    a = Numo::DFloat[[1, 2], [-2, 1]]
    a_sinm = Numo::Linalg.sinm(a)
    error = (Numo::DFloat[[3.1657785, 1.959601], [-1.959601, 3.1657785]] - a_sinm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DFloat, a_sinm)

    a = Numo::DFloat[[1, 2], [-3, 2]]
    error = (Numo::DFloat[[5.4512198, 0.3218189], [-0.4827283, 5.6121293]] - Numo::Linalg.sinm(a)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat[[1, 2], [3, 2]] + (Numo::DFloat[[2, 3], [2, 1]] * Complex::I)
    expected = (Numo::DFloat[[-9.2624065, -7.9639489],
                             [-11.4045322, -12.7029898]] +
                (Numo::DFloat[[-11.1732702, -10.5383063],
                              [-6.66461, -7.2995739]] * Complex::I))
    a_sinm = Numo::Linalg.sinm(a)
    error = (expected - a_sinm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DComplex, a_sinm)
  end

  def test_cosm
    a = Numo::DFloat[[1, 2], [-2, 1]]
    a_cosm = Numo::Linalg.cosm(a)
    error = (Numo::DFloat[[2.032723, -3.0518978], [3.0518978, 2.032723]] - a_cosm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DFloat, a_cosm)

    a = Numo::DFloat[[1, 2], [-3, 2]]
    error = (Numo::DFloat[[1.5268037, -4.5381036], [6.8071554, -0.742248]] - Numo::Linalg.cosm(a)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::DFloat[[1, 2], [3, 2]] + (Numo::DFloat[[2, 3], [2, 1]] * Complex::I)
    expected = (Numo::DFloat[[-10.6722586, -11.5059886],
                             [-7.1775936, -6.3438635]] +
                (Numo::DFloat[[7.9637316, 8.9526293],
                              [12.6893458, 11.70044808]] * Complex::I))
    a_cosm = Numo::Linalg.cosm(a)
    error = (expected - a_cosm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DComplex, a_cosm)
  end

  def test_tanm
    a = Numo::DFloat[[1, 2], [-2, 1]]
    a_tanm = Numo::Linalg.tanm(a)
    error = (Numo::DFloat[[0.0338128, 1.0147936], [-1.0147936, 0.0338128]] - a_tanm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DFloat, a_tanm)

    a = Numo::DFloat[[1, 2], [3, 2]] + (Numo::DFloat[[2, 3], [2, 1]] * Complex::I)
    expected = (Numo::DFloat[[-0.3439464, -0.0721939],
                             [0.3446103,  0.0728577]] +
               (Numo::DFloat[[-0.0146713, 1.0692520],
                             [1.0147687, -0.0691546]] * Complex::I))
    a_tanm = Numo::Linalg.tanm(a)
    error = (expected - a_tanm).abs.max

    assert_operator(error, :<, 1e-7)
    assert_kind_of(Numo::DComplex, a_tanm)
  end

  def test_lu_inv
    a = Numo::DFloat.new(5, 5).rand - 0.5
    lu, ipiv = Numo::Linalg.lu_fact(a)
    a_inv = Numo::Linalg.lu_inv(lu, ipiv)
    error = (Numo::DFloat.eye(5) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_cho_inv
    a = Numo::DFloat.new(5, 5).rand - 0.5
    b = a.transpose.dot(a)
    u = Numo::Linalg.cho_fact(b)
    tri_b_inv = Numo::Linalg.cho_inv(u)
    tri_b_inv = tri_b_inv.triu
    b_inv = tri_b_inv + tri_b_inv.transpose - tri_b_inv.diagonal.diag
    error = (Numo::DFloat.eye(5) - b_inv.dot(b)).abs.max

    assert_operator(error, :<, 1e-7)
  end
end
