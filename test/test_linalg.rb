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
    assert_match(/must be square/, assert_raises(ArgumentError) do
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
end
