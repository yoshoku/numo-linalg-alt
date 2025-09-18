# frozen_string_literal: true

require 'test_helper'

class TestLinalgBlas < Minitest::Test # rubocop:disable Metrics/ClassLength
  def setup
    Numo::NArray.srand(Minitest.seed)
  end

  def test_blas_ddot
    x = Numo::DFloat[1, 2, 3]
    y = Numo::DFloat[4, 5, 6]

    assert_equal 32, Numo::Linalg::Blas.ddot(x, y)
  end

  def test_blas_sdot
    x = Numo::SFloat[1, 2, 3]
    y = Numo::SFloat[4, 5, 6]

    assert_equal 32, Numo::Linalg::Blas.sdot(x, y)
  end

  def test_blas_zdotu
    x = Numo::DComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)]
    y = Numo::DComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]

    assert_equal Complex(18, 43), Numo::Linalg::Blas.zdotu(x, y)
  end

  def test_blas_cdotu
    x = Numo::SComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)]
    y = Numo::SComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]

    assert_equal Complex(18, 43), Numo::Linalg::Blas.cdotu(x, y)
  end

  def test_blas_dgemm
    a = Numo::DFloat[[1, 2, 3], [4, 5, 6]]
    b = Numo::DFloat[[7, 8, 9], [3, 2, 1]]

    assert_equal Numo::DFloat[[19, 16, 13], [29, 26, 23], [39, 36, 33]], Numo::Linalg::Blas.dgemm(a, b, transa: 'T')
    assert_equal Numo::DFloat[[50, 10], [122, 28]], Numo::Linalg::Blas.dgemm(a, b, transb: 'T')

    x = (Numo::DFloat.new(3, 2).rand * 10).floor
    y = (Numo::DFloat.new(2, 5).rand * 10).floor

    assert_equal x.dot(y), Numo::Linalg::Blas.dgemm(x, y)
  end

  def test_blas_sgemm
    a = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    b = Numo::SFloat[[7, 8, 9], [3, 2, 1]]

    assert_equal Numo::SFloat[[19, 16, 13], [29, 26, 23], [39, 36, 33]], Numo::Linalg::Blas.sgemm(a, b, transa: 'T')
    assert_equal Numo::SFloat[[50, 10], [122, 28]], Numo::Linalg::Blas.sgemm(a, b, transb: 'T')
  end

  def test_blas_zgemm
    a = (Numo::DComplex.new(2, 3).rand * 10).floor
    b = (Numo::DComplex.new(2, 3).rand * 10).floor

    assert_equal a.dot(b.transpose), Numo::Linalg::Blas.zgemm(a, b, transb: 'T')
    assert_equal a.transpose.dot(b), Numo::Linalg::Blas.zgemm(a, b, transa: 'T')
  end

  def test_blas_cgemm
    a = (Numo::SComplex.new(2, 3).rand * 10).floor
    b = (Numo::SComplex.new(2, 3).rand * 10).floor

    assert_equal a.dot(b.transpose), Numo::Linalg::Blas.cgemm(a, b, transb: 'T')
    assert_equal a.transpose.dot(b), Numo::Linalg::Blas.cgemm(a, b, transa: 'T')
  end

  def test_blas_dgemv
    a = Numo::DFloat[[1, 2, 3], [4, 5, 6]]
    x = Numo::DFloat[7, 8, 9]
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::Linalg::Blas.dgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::Linalg::Blas.dgemv(b, x, trans: 'T')
  end

  def test_blas_sgemv
    a = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    x = Numo::SFloat[7, 8, 9]
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::Linalg::Blas.sgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::Linalg::Blas.sgemv(b, x, trans: 'T')
  end

  def test_blas_zgemv
    a = (Numo::DComplex.new(2, 3).rand * 10).floor
    x = (Numo::DComplex.new(3).rand * 10).floor
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::Linalg::Blas.zgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::Linalg::Blas.zgemv(b, x, trans: 'T')
  end

  def test_blas_cgemv
    a = (Numo::SComplex.new(2, 3).rand * 10).floor
    x = (Numo::SComplex.new(3).rand * 10).floor
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::Linalg::Blas.cgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::Linalg::Blas.cgemv(b, x, trans: 'T')
  end

  def test_blas_dnrm2
    a = Numo::DFloat.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a)) - Numo::Linalg::Blas.dnrm2(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_blas_snrm2
    a = Numo::SFloat.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a)) - Numo::Linalg::Blas.snrm2(a)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_blas_dznrm2
    a = Numo::DComplex.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::Linalg::Blas.dznrm2(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_blas_scnrm2
    a = Numo::SComplex.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::Linalg::Blas.scnrm2(a)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_blas_aliases
    a = Numo::DComplex.new(3).rand
    b = Numo::SComplex.new(3).rand
    error_a = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::Linalg::Blas.znrm2(a)).abs.max
    error_b = (Numo::NMath.sqrt(b.dot(b.conjugate)) - Numo::Linalg::Blas.cnrm2(b)).abs.max

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-5)
  end

  def test_blas_call
    assert_equal 32, Numo::Linalg::Blas.call(:dot, Numo::DFloat[1, 2, 3], Numo::DFloat[4, 5, 6])
    assert_equal 32, Numo::Linalg::Blas.call(:dot, Numo::SFloat[1, 2, 3], Numo::SFloat[4, 5, 6])
    assert_equal Complex(18, 43), Numo::Linalg::Blas.call(
      :dotu,
      Numo::DComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)],
      Numo::DComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]
    )
    assert_equal Complex(18, 43), Numo::Linalg::Blas.call(
      :dotu,
      Numo::SComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)],
      Numo::SComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]
    )
  end
end
