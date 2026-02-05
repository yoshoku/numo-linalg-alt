# frozen_string_literal: true

require 'numo/narray/alt'

require_relative 'linalg/version'
# On RHEL-based Linux distributions, native extensions are installed in a separate
# directory from Ruby scripts, so use require to load them.
require 'numo/linalg/linalg'

# Ruby/Numo (NUmerical MOdules)
module Numo
  # Numo::Linalg Alternative (numo-linalg-alt) is an alternative to Numo::Linalg.
  module Linalg # rubocop:disable Metrics/ModuleLength
    # Exception class for errors occurred in LAPACK functions.
    class LapackError < StandardError; end

    module_function

    # Computes the eigenvalues and eigenvectors of a symmetric / Hermitian matrix
    # by solving an ordinary or generalized eigenvalue problem.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   x = Numo::DFloat.new(5, 3).rand - 0.5
    #   c = x.dot(x.transpose)
    #   vals, vecs = Numo::Linalg.eigh(c, vals_range: [2, 4])
    #
    #   pp vals
    #   # =>
    #   # Numo::DFloat#shape=[3]
    #   # [0.118795, 0.434252, 0.903245]
    #
    #   pp vecs
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.154178, 0.60661, -0.382961],
    #   #  [-0.349761, -0.141726, -0.513178],
    #   #  [0.739633, -0.468202, 0.105933],
    #   #  [0.0519655, -0.471436, -0.701507],
    #   #  [-0.551488, -0.412883, 0.294371]]
    #
    #   pp (x - vecs.dot(vals.diag).dot(vecs.transpose)).abs.max
    #   # => 3.3306690738754696e-16
    #
    # @param a [Numo::NArray] The n-by-n symmetric / Hermitian matrix.
    # @param b [Numo::NArray] The n-by-n symmetric / Hermitian matrix. If nil, identity matrix is assumed.
    # @param vals_only [Boolean] The flag indicating whether to return only eigenvalues.
    # @param vals_range [Range/Array]
    #   The range of indices of the eigenvalues (in ascending order) and corresponding eigenvectors to be returned.
    #   If nil, all eigenvalues and eigenvectors are computed.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param turbo [Bool] The flag indicating whether to use a divide and conquer algorithm. If vals_range is given, this flag is ignored.
    # @return [Array<Numo::NArray>] The eigenvalues and eigenvectors.
    def eigh(a, b = nil, vals_only: false, vals_range: nil, uplo: 'U', turbo: false) # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity, Lint/UnusedMethodArgument
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      b_given = !b.nil?
      raise Numo::NArray::ShapeError, 'input array b must be 2-dimensional' if b_given && b.ndim != 2
      raise Numo::NArray::ShapeError, 'input array b must be square' if b_given && b.shape[0] != b.shape[1]
      raise ArgumentError, "invalid array type: #{b.class}" if b_given && blas_char(b) == 'n'

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      jobz = vals_only ? 'N' : 'V'

      if b_given
        fnc = %w[d s].include?(bchr) ? "#{bchr}sygv" : "#{bchr}hegv"
        if vals_range.nil?
          fnc << 'd' if turbo
          vecs, _b, vals, _info = Numo::Linalg::Lapack.send(fnc.to_sym, a.dup, b.dup, jobz: jobz)
        else
          fnc << 'x'
          il = vals_range.first(1)[0] + 1
          iu = vals_range.last(1)[0] + 1
          _a, _b, _m, vals, vecs, _ifail, _info = Numo::Linalg::Lapack.send(
            fnc.to_sym, a.dup, b.dup, jobz: jobz, range: 'I', il: il, iu: iu
          )
        end
      else
        fnc = %w[d s].include?(bchr) ? "#{bchr}syev" : "#{bchr}heev"
        if vals_range.nil?
          fnc << 'd' if turbo
          vecs, vals, _info = Numo::Linalg::Lapack.send(fnc.to_sym, a.dup, jobz: jobz)
        else
          fnc << 'r'
          il = vals_range.first(1)[0] + 1
          iu = vals_range.last(1)[0] + 1
          _a, _m, vals, vecs, _isuppz, _info = Numo::Linalg::Lapack.send(
            fnc.to_sym, a.dup, jobz: jobz, range: 'I', il: il, iu: iu
          )
        end
      end

      vecs = nil if vals_only

      [vals, vecs]
    end

    # Computes the matrix or vector norm.
    #
    #   |  ord  |  matrix norm           | vector norm                 |
    #   | ----- | ---------------------- | --------------------------- |
    #   |  nil  | Frobenius norm         | 2-norm                      |
    #   | 'fro' | Frobenius norm         |  -                          |
    #   | 'nuc' | nuclear norm           |  -                          |
    #   | 'inf' | x.abs.sum(axis:-1).max | x.abs.max                   |
    #   |    0  |  -                     | (x.ne 0).sum                |
    #   |    1  | x.abs.sum(axis:-2).max | same as below               |
    #   |    2  | 2-norm (max sing_vals) | same as below               |
    #   | other |  -                     | (x.abs**ord).sum**(1.0/ord) |
    #
    # @example
    #   require 'numo/linalg'
    #
    #   # matrix norm
    #   x = Numo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]
    #   pp Numo::Linalg.norm(x)
    #   # => 10
    #
    #   # vector norm
    #   x = Numo::DFloat[3, -4]
    #   pp Numo::Linalg.norm(x)
    #   # => 5
    #
    # @param a [Numo::NArray] The matrix or vector (>= 1-dimensinal NArray)
    # @param ord [String/Numeric] The order of the norm.
    # @param axis [Integer/Array] The applied axes.
    # @param keepdims [Bool] The flag indicating whether to leave the normed axes in the result as dimensions with size one.
    # @return [Numo::NArray/Numeric] The norm of the matrix or vectors.
    def norm(a, ord = nil, axis: nil, keepdims: false) # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/PerceivedComplexity
      a = Numo::NArray.asarray(a) unless a.is_a?(Numo::NArray)

      return 0.0 if a.empty?

      # for compatibility with Numo::Linalg.norm
      if ord.is_a?(String)
        if ord == 'inf'
          ord = Float::INFINITY
        elsif ord == '-inf'
          ord = -Float::INFINITY
        end
      end

      if axis.nil?
        norm = case a.ndim
               when 1
                 Numo::Linalg::Blas.send(:"#{blas_char(a)}nrm2", a) if ord.nil? || ord == 2
               when 2
                 if ord.nil? || ord == 'fro'
                   Numo::Linalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: 'F')
                 elsif ord.is_a?(Numeric)
                   if ord == 1
                     Numo::Linalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: '1')
                   elsif !ord.infinite?.nil? && ord.infinite?.positive?
                     Numo::Linalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: 'I')
                   end
                 end
               else
                 if ord.nil?
                   b = a.flatten.dup
                   Numo::Linalg::Blas.send(:"#{blas_char(b)}nrm2", b)
                 end
               end
        unless norm.nil?
          norm = Numo::NArray.asarray(norm).reshape(*([1] * a.ndim)) if keepdims
          return norm
        end
      end

      if axis.nil?
        axis = Array.new(a.ndim) { |d| d }
      else
        case axis
        when Integer
          axis = [axis]
        when Array, Numo::NArray
          axis = axis.flatten.to_a
        else
          raise ArgumentError, "invalid axis: #{axis}"
        end
      end

      raise ArgumentError, "the number of dimensions of axis is inappropriate for the norm: #{axis.size}" unless [1, 2].include?(axis.size)
      raise ArgumentError, "axis is out of range: #{axis}" unless axis.all? { |ax| (-a.ndim...a.ndim).cover?(ax) }

      if axis.size == 1
        ord ||= 2
        raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(Numeric)

        ord_inf = ord.infinite?
        if ord_inf.nil?
          case ord
          when 0
            a.class.cast(a.ne(0)).sum(axis: axis, keepdims: keepdims)
          when 1
            a.abs.sum(axis: axis, keepdims: keepdims)
          else
            (a.abs**ord).sum(axis: axis, keepdims: keepdims)**1.fdiv(ord)
          end
        elsif ord_inf.positive?
          a.abs.max(axis: axis, keepdims: keepdims)
        else
          a.abs.min(axis: axis, keepdims: keepdims)
        end
      else
        ord ||= 'fro'
        raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(String) || ord.is_a?(Numeric)
        raise ArgumentError, "invalid axis: #{axis}" if axis.uniq.size == 1

        r_axis, c_axis = axis.map { |ax| ax.negative? ? ax + a.ndim : ax }

        norm = if ord.is_a?(String)
                 raise ArgumentError, "invalid ord: #{ord}" unless %w[fro nuc].include?(ord)

                 if ord == 'fro'
                   Numo::NMath.sqrt((a.abs**2).sum(axis: axis))
                 else
                   b = a.transpose(c_axis, r_axis).dup
                   gesvd = :"#{blas_char(b)}gesvd"
                   s, = Numo::Linalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                   s.sum(axis: -1)
                 end
               else
                 ord_inf = ord.infinite?
                 if ord_inf.nil?
                   case ord
                   when -2
                     b = a.transpose(c_axis, r_axis).dup
                     gesvd = :"#{blas_char(b)}gesvd"
                     s, = Numo::Linalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                     s.min(axis: -1)
                   when -1
                     c_axis -= 1 if c_axis > r_axis
                     a.abs.sum(axis: r_axis).min(axis: c_axis)
                   when 1
                     c_axis -= 1 if c_axis > r_axis
                     a.abs.sum(axis: r_axis).max(axis: c_axis)
                   when 2
                     b = a.transpose(c_axis, r_axis).dup
                     gesvd = :"#{blas_char(b)}gesvd"
                     s, = Numo::Linalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                     s.max(axis: -1)
                   else
                     raise ArgumentError, "invalid ord: #{ord}"
                   end
                 else
                   r_axis -= 1 if r_axis > c_axis
                   if ord_inf.positive?
                     a.abs.sum(axis: c_axis).max(axis: r_axis)
                   else
                     a.abs.sum(axis: c_axis).min(axis: r_axis)
                   end
                 end
               end
        if keepdims
          norm = Numo::NArray.asarray(norm) unless norm.is_a?(Numo::NArray)
          norm = norm.reshape(*([1] * a.ndim))
        end

        norm
      end
    end

    # Computes the Cholesky decomposition of a symmetric / Hermitian positive-definite matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   s = Numo::DFloat.new(3, 3).rand - 0.5
    #   a = s.transpose.dot(s)
    #   u = Numo::Linalg.cholesky(a)
    #
    #   pp u
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0.532006, 0.338183, -0.18036],
    #   #  [0, 0.325153, 0.011721],
    #   #  [0, 0, 0.436738]]
    #
    #   pp (a - u.transpose.dot(u)).abs.max
    #   # => 1.3877787807814457e-17
    #
    #   l = Numo::Linalg.cholesky(a, uplo: 'L')
    #
    #   pp l
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0.532006, 0, 0],
    #   #  [0.338183, 0.325153, 0],
    #   #  [-0.18036, 0.011721, 0.436738]]
    #
    #   pp (a - l.dot(l.transpose)).abs.max
    #   # => 1.3877787807814457e-17
    #
    # @param a [Numo::NArray] The n-by-n symmetric matrix.
    # @param uplo [String] Whether to compute the upper- or lower-triangular Cholesky factor ('U' or 'L').
    # @return [Numo::NArray] The upper- or lower-triangular Cholesky factor of a.
    def cholesky(a, uplo: 'U')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potrf"
      c, _info = Numo::Linalg::Lapack.send(fnc, a.dup, uplo: uplo)

      case uplo
      when 'U'
        c.triu
      when 'L'
        c.tril
      else
        raise ArgumentError, "invalid uplo: #{uplo}"
      end
    end

    # Computes the Cholesky decomposition of a banded symmetric / Hermitian positive-definite matrix.
    #
    # @param a [Numo::NArray] The banded matrix.
    # @param uplo [String] Is the matrix form upper or lower ('U' or 'L').
    # @return [Numo::NArray] The Cholesky factor of the banded matrix.
    def cholesky_banded(a, uplo: 'U')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}pbtrf"
      c, info = Numo::Linalg::Lapack.send(fnc, a.dup, uplo: uplo)
      raise LapackError, "the #{-info}-th argument of potrs had illegal value" if info.negative?

      if info.positive?
        raise LapackError, "the leading principal minor of order #{info} is not positive, " \
                           'and the factorization could not be completed.'
      end

      c
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` with the Cholesky factorization of `A`.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   s = Numo::DFloat.new(3, 3).rand - 0.5
    #   a = s.transpose.dot(s)
    #   u = Numo::Linalg.cholesky(a)
    #
    #   b = Numo::DFloat.new(3).rand
    #   x = Numo::Linalg.cho_solve(u, b)
    #
    #   puts (b - a.dot(x)).abs.max
    #   => 0.0
    #
    # @param a [Numo::NArray] The n-by-n cholesky factor.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param uplo [String] Whether to compute the upper- or lower-triangular Cholesky factor ('U' or 'L').
    # @return [Numo::NArray] The solution vector or matrix `X`.
    def cho_solve(a, b, uplo: 'U')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise Numo::NArray::ShapeError, "incompatible dimensions: a.shape[0] = #{a.shape[0]} != b.shape[0] = #{b.shape[0]}" if a.shape[0] != b.shape[0]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potrs"
      x, info = Numo::Linalg::Lapack.send(fnc, a, b.dup, uplo: uplo)
      raise LapackError, "the #{-info}-th argument of potrs had illegal value" if info.negative?

      x
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` with the Cholesky factorization of banded matrix `A`.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(4, 4).rand - 0.5
    #   a = a.dot(a.transpose)
    #   a = a.tril(1) * a.triu(-1)
    #   ab = Numo::DFloat.zeros(2, 4)
    #   ab[0, 1...] = a[a.diag_indices(1)]
    #   ab[1, true] = a[a.diag_indices]
    #   c = Numo::Linalg.cholesky_banded(ab)
    #   b = Numo::DFloat.new(4, 2).rand
    #   x = Numo::Linalg.cho_solve_banded(c, b)
    #   pp (b - a.dot(x)).abs.max
    #   # => 1.1102230246251565e-16
    #
    # @param ab [Numo::NArray] The m-by-n banded cholesky factor.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param uplo [String] Whether to compute the upper- or lower-triangular Cholesky factor ('U' or 'L').
    # @return [Numo::NArray] The solution vector or matrix `X`.
    def cho_solve_banded(ab, b, uplo: 'U')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if ab.ndim != 2
      raise Numo::NArray::ShapeError, "incompatible dimensions: ab.shape[1] = #{ab.shape[1]} != b.shape[0] = #{b.shape[0]}" if ab.shape[1] != b.shape[0]

      bchr = blas_char(ab, b)
      raise ArgumentError, "invalid array type: #{ab.class}" if bchr == 'n'

      fnc = :"#{bchr}pbtrs"
      x, info = Numo::Linalg::Lapack.send(fnc, ab, b.dup, uplo: uplo)
      raise LapackError, "the #{-info}-th argument of potrs had illegal value" if info.negative?

      x
    end

    # Computes the determinant of matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    #   pp (3.0 - Numo::Linalg.det(a)).abs
    #   # => 1.3322676295501878e-15
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Float/Complex] The determinant of `a`.
    def det(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = :"#{bchr}getrf"
      lu, piv, info = Numo::Linalg::Lapack.send(getrf, a.dup)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      # info > 0 means the factor U has a zero diagonal element and is singular.
      # In this case, the determinant is zero. The method should simply return 0.0.
      # Therefore, the error is not raised here.
      # raise 'the factor U is singular, ...' if info.positive?

      det_l = 1
      det_u = lu.diagonal.prod
      det_p = piv.map_with_index { |v, i| v == i + 1 ? 1 : -1 }.prod
      det_l * det_u * det_p
    end

    # Computes the inverse matrix of a square matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(5, 5).rand
    #
    #   inv_a = Numo::Linalg.inv(a)
    #
    #   pp (inv_a.dot(a) - Numo::DFloat.eye(5)).abs.max
    #   # => 7.019165976816745e-16
    #
    #   pp inv_a.dot(a).sum
    #   # => 5.0
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The inverse matrix of `a`.
    def inv(a, driver: 'getrf', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = :"#{bchr}getrf"
      getri = :"#{bchr}getri"

      lu, piv, info = Numo::Linalg::Lapack.send(getrf, a.dup)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      a_inv, info = Numo::Linalg::Lapack.send(getri, lu, piv)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?
      raise LapackError, 'The matrix is singular, and the inverse matrix could not be computed.' if info.positive?

      a_inv
    end

    # Computes the (Moore-Penrose) pseudo-inverse of a matrix using singular value decomposition.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(5, 3).rand
    #
    #   inv_a = Numo::Linalg.pinv(a)
    #
    #   pp (inv_a.dot(a) - Numo::DFloat.eye(3)).abs.max
    #   # => 1.1102230246251565e-15
    #
    #   pp inv_a.dot(a).sum
    #   # => 3.0
    #
    # @param a [Numo::NArray] The m-by-n matrix to be pseudo-inverted.
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @param rcond [Float] The threshold value for small singular values of `a`, default value is `a.shape.max * EPS`.
    # @return [Numo::NArray] The pseudo-inverse of `a`.
    def pinv(a, driver: 'svd', rcond: nil)
      s, u, vh = svd(a, driver: driver, job: 'S')
      rcond = a.shape.max * s.class::EPSILON if rcond.nil?
      rank = s.gt(rcond * s[0]).count

      u = u[true, 0...rank] / s[0...rank]
      u.dot(vh[0...rank, true]).conj.transpose.dup
    end

    # Computes the polar decomposition of a matrix.
    #
    # https://en.wikipedia.org/wiki/Polar_decomposition
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[0.5, 1, 2], [1.5, 3, 4]]
    #   u, p = Numo::Linalg.polar(a)
    #   pp u.dot(p)
    #   # =>
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0.5, 1, 2],
    #   #  [1.5, 3, 4]]
    #   pp u.dot(u.transpose)
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[1, -1.68043e-16],
    #   #  [-1.68043e-16, 1]]
    #
    #   u, p = Numo::Linalg.polar(a, side: 'left')
    #   pp p.dot(u)
    #   # =>
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0.5, 1, 2],
    #   #  [1.5, 3, 4]]
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @param side [String] The side of polar decomposition ('right' or 'left').
    # @return [Array<Numo::NArray>] The unitary matrix `U` and the positive-semidefinite Hermitian matrix `P`
    #   such that `A = U * P` if side='right', or `A = P * U` if side='left'.
    def polar(a, side: 'right')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArugumentError, "invalid side: #{side}" unless %w[left right].include?(side)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      s, w, vh = svd(a, driver: 'svd', job: 'S')
      u = w.dot(vh)
      p_mat = if side == 'right'
                vh.transpose.conj.dot(s.diag).dot(vh)
              else
                w.dot(s.diag).dot(w.transpose.conj)
              end
      [u, p_mat]
    end

    # Computes the QR decomposition of a matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   x = Numo::DFloat.new(5, 3).rand
    #
    #   q, r = Numo::Linalg.qr(x, mode: 'economic')
    #
    #   pp q
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[-0.0574417, 0.635216, 0.707116],
    #   #  [-0.187002, -0.073192, 0.422088],
    #   #  [-0.502239, 0.634088, -0.537489],
    #   #  [-0.0473292, 0.134867, -0.0223491],
    #   #  [-0.840979, -0.413385, 0.180096]]
    #
    #   pp r
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[-1.07508, -0.821334, -0.484586],
    #   #  [0, 0.513035, 0.451868],
    #   #  [0, 0, 0.678737]]
    #
    #   pp (q.dot(r) - x).abs.max
    #   # => 3.885780586188048e-16
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @param mode [String] The mode of decomposition.
    #   - "reduce"   -- returns both Q [m, m] and R [m, n],
    #   - "r"        -- returns only R,
    #   - "economic" -- returns both Q [m, n] and R [n, n],
    #   - "raw"      -- returns QR and TAU (LAPACK geqrf results).
    # @return [Numo::NArray] if mode='r'.
    # @return [Array<Numo::NArray>] if mode='reduce' or 'economic' or 'raw'.
    def qr(a, mode: 'reduce')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, "invalid mode: #{mode}" unless %w[reduce r economic raw].include?(mode)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      geqrf = :"#{bchr}geqrf"
      qr, tau, = Numo::Linalg::Lapack.send(geqrf, a.dup)

      return [qr, tau] if mode == 'raw'

      m, n = qr.shape
      r = m > n && %w[economic raw].include?(mode) ? qr[0...n, true].triu : qr.triu

      return r if mode == 'r'

      org_ung_qr = %w[d s].include?(bchr) ? :"#{bchr}orgqr" : :"#{bchr}ungqr"

      q = if m < n
            Numo::Linalg::Lapack.send(org_ung_qr, qr[true, 0...m], tau)[0]
          elsif mode == 'economic'
            Numo::Linalg::Lapack.send(org_ung_qr, qr, tau)[0]
          else
            qqr = a.class.zeros(m, m)
            qqr[0...m, 0...n] = qr
            Numo::Linalg::Lapack.send(org_ung_qr, qqr, tau)[0]
          end

      [q, r]
    end

    # Computes the RQ decomposition of a matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(2, 3).rand
    #   r, q = Numo::Linalg.rq(a)
    #   pp r
    #   # =>
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0, -0.381748, -0.79309],
    #   #  [0, 0, -0.41502]]
    #   pp q
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0.227957, 0.874475, -0.428169],
    #   #  [0.844617, -0.396377, -0.359872],
    #   #  [-0.484416, -0.279603, -0.828953]]
    #   puts (a - r.dot(q)).abs.max
    #   # => 5.551115123125783e-17
    #
    #   r, q = Numo::Linalg.rq(a, mode: 'economic')
    #   pp r
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[-0.381748, -0.79309],
    #   #  [0, -0.41502]]
    #   pp q
    #   # =>
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0.844617, -0.396377, -0.359872],
    #   #  [-0.484416, -0.279603, -0.828953]]
    #   puts (a - r.dot(q)).abs.max
    #   # => 5.551115123125783e-17
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @param mode [String] The mode of decomposition.
    #   - "full"     -- returns both R [m, n] and Q [n, n],
    #   - "r"        -- returns only R,
    #   - "economic" -- returns both R [m, k] and Q [k, n], where k = min(m, n).
    # @return [Array<Numo::NArray>/Numo::NArray]
    #   if mode='full' or 'economic', returns [R, Q].
    #   if mode='r', returns R.
    def rq(a, mode: 'full') # rubocop:disable  Metrics/AbcSize
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, "invalid mode: #{mode}" unless %w[full r economic].include?(mode)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}gerqf"
      rq, tau, info = Numo::Linalg::Lapack.send(fnc, a.dup)
      raise LapackError, "the #{-info}-th argument of #{fnc} had illegal value" if info.negative?

      m, n = rq.shape
      r = rq.triu(n - m).dup
      r = r[true, (n - m)...n].dup if mode == 'economic' && n > m

      return r if mode == 'r'

      fnc = %w[d s].include?(bchr) ? :"#{bchr}orgrq" : :"#{bchr}ungrq"
      tmp = if n < m
              rq[(m - n)...m, 0...n].dup
            elsif mode == 'economic'
              rq.dup
            else
              rq.class.zeros(n, n).tap { |mat| mat[(n - m)...n, true] = rq }
            end

      q, info = Numo::Linalg::Lapack.send(fnc, tmp, tau)
      raise LapackError, "the #{-info}-th argument of #{fnc} had illegal value" if info.negative?

      [r, q]
    end

    # Computes the QZ decomposition (generalized Schur decomposition) of a pair of square matrices.
    #
    # The QZ decomposition is given by `A = Q * AA * Z^H` and `B = Q * BB * Z^H`,
    # where `A` and `B` are the input matrices, `Q` and `Z` are unitary matrices,
    # and `AA` and `BB` are upper triangular matrices (or quasi-upper triangular matrices in real case).
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(5, 5).rand
    #   b = Numo::DFloat.new(5, 5).rand
    #
    #   aa, bb, q, z = Numo::Linalg.qz(a, b)
    #
    #   pp (a - q.dot(aa).dot(z.transpose)).abs.max
    #   # => 1.7763568394002505e-15
    #   pp (b - q.dot(bb).dot(z.transpose)).abs.max
    #   # => 1.1102230246251565e-15
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param b [Numo::NArray] The n-by-n square matrix.
    # @return [Array<Numo::NArray, Numo::NArray, Numo::NArray, Numo::NArray>]
    #   The matrices `AA`, `BB`, `Q`, and `Z` such that `A = Q * AA * Z^H` and `B = Q * BB * Z^H`.
    def qz(a, b) # rubocop:disable Metrics/AbcSize
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array b must be 2-dimensional' if b.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise Numo::NArray::ShapeError, 'input array b must be square' if b.shape[0] != b.shape[1]
      raise Numo::NArray::ShapeError, "incompatible dimensions: a.shape = #{a.shape} != b.shape = #{b.shape}" if a.shape != b.shape

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      fnc = :"#{bchr}gges"
      if %w[d s].include?(bchr)
        aa, bb, _ar, _ai, _beta, q, z, _sdim, info = Numo::Linalg::Lapack.send(fnc, a.dup, b.dup)
      else
        aa, bb, _alpha, _beta, q, z, _sdim, info = Numo::Linalg::Lapack.send(fnc, a.dup, b.dup)
      end

      n = a.shape[0]
      raise LapackError, "the #{-info}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'something other than QZ iteration failed.' if info == n + 1
      raise LapackError, "reordering failed in #{bchr}tgsen" if info == n + 3

      if info == n + 2
        raise LapackError, 'after reordering, roundoff changed values of some eigenvalues ' \
                           'so that leading eigenvalues in the Generalized Schur form no ' \
                           'longer satisfy the sorting condition.'
      end

      if info.positive? && info <= n
        warn('the QZ iteration failed. (a, b) are not in Schur form, ' \
             "but alpha[i] and beta[i] for i = #{info},...,n are correct.")
      end

      [aa, bb, q, z]
    end

    # Computes the Schur decomposition of a square matrix.
    # The Schur decomposition is given by `A = Z * T * Z^H`,
    # where `A` is the input matrix, `Z` is a unitary matrix,
    # and `T` is an upper triangular matrix (or quasi-upper triangular matrix in real case).
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    #   t, z, sdim = Numo::Linalg.schur(a)
    #   pp t
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[16.0104, 4.81155, 0.920982],
    #   #  [0, -1.91242, 0.0274406],
    #   #  [0, 0, -0.0979794]]
    #   pp z
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[-0.219668, -0.94667, 0.235716],
    #   #  [-0.527141, -0.0881306, -0.845195],
    #   #  [-0.820895, 0.309918, 0.479669]]
    #   pp sdim
    #   # => 0
    #   pp (a - z.dot(t).dot(z.transpose)).abs.max
    #   # => 1.0658141036401503e-14
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param sort [String/Nil] The option for sorting eigenvalues ('lhp', 'rhp', 'iuc', 'ouc', or nil).
    #   - 'lhp': eigenvalue.real < 0
    #   - 'rhp': eigenvalue.real >= 0
    #   - 'iuc': eigenvalue.abs <= 1
    #   - 'ouc': eigenvalue.abs > 1
    # @return [Array<Numo::NArray, Numo::NArray, Integer>] The Schur form `T`, the unitary matrix `Z`,
    #   and the number of eigenvalues for which the sorting condition is true.
    def schur(a, sort: nil)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, "invalid sort: #{sort}" unless sort.nil? || %w[lhp rhp iuc ouc].include?(sort)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}gees"
      if %w[d s].include?(bchr)
        b, _wr, _wi, v, sdim, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvs: 'V', sort: sort)
      else
        b, _w, v, sdim, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvs: 'V', sort: sort)
      end

      n = a.shape[0]
      raise LapackError, "the #{-info}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'the QR algorithm failed to compute all the eigenvalues.' if info.positive? && info <= n
      raise LapackError, 'the eigenvalues could not be reordered.' if info == n + 1

      if info == n + 2
        raise LapackError, 'after reordering, roundoff changed values of some eigenvalues ' \
                           'so that leading eigenvalues in the Schur form no longer satisfy ' \
                           'the sorting condition.'
      end

      [b, v, sdim]
    end

    # Computes the Hessenberg decomposition of a square matrix.
    # The Hessenberg decomposition is given by `A = Q * H * Q^H`,
    # where `A` is the input matrix, `Q` is a unitary matrix,
    # and `H` is an upper Hessenberg matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 0.5
    #   h, q = Numo::Linalg.hessenberg(a, calc_q: true)
    #
    #   pp h
    #   # => Numo::DFloat#shape=[3,3]
    #   # [[0.5, -1.7985, -0.124035],
    #   #  [-4.03113, 7.02308, 1.41538],
    #   #  [0, 0.415385, -0.0230769]]
    #   pp q
    #   # => Numo::DFloat#shape=[3,3]
    #   # [[1, 0, 0],
    #   #  [0, -0.496139, -0.868243],
    #   #  [0, -0.868243, 0.496139]]
    #   pp (a - q.dot(h).dot(q.transpose)).abs.max
    #   # => 1.7763568394002505e-15
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param calc_q [Boolean] The flag indicating whether to calculate the unitary matrix `Q`.
    # @return [Numo::NArray] if calc_q=false, the Hessenberg form `H`.
    # @return [Array<Numo::NArray, Numo::NArray>] if calc_q=true,
    #   the Hessenberg form `H` and the unitary matrix `Q`.
    def hessenberg(a, calc_q: false)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      func = :"#{bchr}gebal"
      b, ilo, ihi, _, info = Numo::Linalg::Lapack.send(func, a.dup)

      raise LapackError, "the #{-info}-th argument of #{func} had illegal value" if info.negative?

      func = :"#{bchr}gehrd"
      hq, tau, info = Numo::Linalg::Lapack.send(func, b, ilo: ilo, ihi: ihi)

      raise LapackError, "the #{-info}-th argument of #{func} had illegal value" if info.negative?

      h = hq.triu(-1)
      return h unless calc_q

      func = %w[d s].include?(bchr) ? :"#{bchr}orghr" : :"#{bchr}unghr"
      q, info = Numo::Linalg::Lapack.send(func, hq, tau, ilo: ilo, ihi: ihi)

      raise LapackError, "the #{-info}-th argument of #{func} had illegal value" if info.negative?

      [h, q]
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` from square matrix `A`.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 3).rand
    #   b = Numo::DFloat.eye(3)
    #
    #   x = Numo::Linalg.solve(a, b)
    #
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[-2.12332, 4.74868, 0.326773],
    #   #  [1.38043, -3.79074, 1.25355],
    #   #  [0.775187, 1.41032, -0.613774]]
    #
    #   pp (b - a.dot(x)).abs.max
    #   # => 2.1081041547796492e-16
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def solve(a, b, driver: 'gen', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      gesv = :"#{bchr}gesv"
      _lu, x, _ipiv, info = Numo::Linalg::Lapack.send(gesv, a.dup, b.dup)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      if info.positive?
        warn('the factorization has been completed, but the factor is singular, ' \
             'so the solution could not be computed.')
      end

      x
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` assuming `A` is a triangular matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 3).rand.triu
    #   b = Numo::DFloat.eye(3)
    #
    #   x = Numo::Linalg.solve(a, b)
    #
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[16.1932, -52.0604, 30.5283],
    #   #  [0, 8.61765, -17.9585],
    #   #  [0, 0, 6.05735]]
    #
    #   pp (b - a.dot(x)).abs.max
    #   # => 4.071100642430302e-16
    #
    # @param a [Numo::NArray] The n-by-n triangular matrix.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param lower [Boolean] The flag indicating whether to use the lower-triangular part of `a`.
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def solve_triangular(a, b, lower: false)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      trtrs = :"#{bchr}trtrs"
      uplo = lower ? 'L' : 'U'
      x, info = Numo::Linalg::Lapack.send(trtrs, a, b.dup, uplo: uplo)
      raise LapackError, "wrong value is given to the #{info}-th argument of #{trtrs} used internally" if info.negative?

      x
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` assuming `A` is
    # a symmetric/Hermitian positive-definite banded matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   # Banded matrix A:
    #   # [4 2 1 0 0]
    #   # [2 5 2 1 0]
    #   # [1 2 6 2 1]
    #   # [0 1 2 7 2]
    #   # [0 0 1 2 8]
    #   #
    #   # The banded representation ab for lower-banded form is:
    #   ab = Numo::DFloat[[4, 5, 6, 7, 8],
    #                     [2, 2, 2, 2, 0],
    #                     [1, 1, 1, 0, 0]]
    #   # The banded representation ab for upper-banded form is:
    #   # ab = Numo::DFloat[[0, 0, 1, 1, 1],
    #   #                   [0, 2, 2, 2, 2],
    #   #                   [4, 5, 6, 7, 8]]
    #   b = Numo::DFloat[1, 2, 3, 4, 5]
    #
    #   x = Numo::Linalg.solveh_banded(ab, b, lower: true)
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[5]
    #   # [0.0903361, 0.210084, 0.218487, 0.331933, 0.514706]
    #
    #   a = ab[0, true].diag + ab[1, 0...-1].diag(1) + ab[2, 0...-2].diag(2) +
    #    ab[1, 0...-1].diag(-1) + ab[2, 0...-2].diag(-2)
    #   pp a.dot(x)
    #   # => Numo::DFloat#shape=[5]
    #   # [1, 2, 3, 4, 5]
    #
    # @param ab [Numo::NArray] The m-by-n array representing the banded matrix `A`.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-k right-hand side matrix.
    # @param lower [Boolean] The flag indicating whether to be in the lower-banded form.
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def solveh_banded(ab, b, lower: false)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if ab.ndim != 2

      bchr = blas_char(ab, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      pbsv = :"#{bchr}pbsv"
      uplo = lower ? 'L' : 'U'
      x, info = Numo::Linalg::Lapack.send(pbsv, ab.dup, b.dup, uplo: uplo)
      raise LapackError, "wrong value is given to the #{info}-th argument of #{pbsv} used internally" if info.negative?
      raise LapackError, 'the leading minor of the matrix is not positive definite' if info.positive?

      x
    end

    # Computes the Singular Value Decomposition (SVD) of a matrix: `A = U * S * V^T`
    #
    # @example
    #   require 'numo/linalg'
    #
    #   x = Numo::DFloat.new(5, 2).rand.dot(Numo::DFloat.new(2, 3).rand)
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.104945, 0.0284236, 0.117406],
    #   #  [0.862634, 0.210945, 0.922135],
    #   #  [0.324507, 0.0752655, 0.339158],
    #   #  [0.67085, 0.102594, 0.600882],
    #   #  [0.404631, 0.116868, 0.46644]]
    #
    #   s, u, vt = Numo::Linalg.svd(x, job: 'S')
    #
    #   z = u.dot(s.diag).dot(vt)
    #   pp z
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.104945, 0.0284236, 0.117406],
    #   #  [0.862634, 0.210945, 0.922135],
    #   #  [0.324507, 0.0752655, 0.339158],
    #   #  [0.67085, 0.102594, 0.600882],
    #   #  [0.404631, 0.116868, 0.46644]]
    #
    #   pp (x - z).abs.max
    #   # => 4.440892098500626e-16
    #
    # @param a [Numo::NArray] Matrix to be decomposed.
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @param job [String] The job option ('A', 'S', or 'N').
    # @return [Array<Numo::NArray>] The singular values and singular vectors ([s, u, vt]).
    def svd(a, driver: 'svd', job: 'A')
      raise ArgumentError, "invalid job: #{job}" unless /^[ASN]/i.match?(job.to_s)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      case driver.to_s
      when 'sdd'
        gesdd = :"#{bchr}gesdd"
        s, u, vt, info = Numo::Linalg::Lapack.send(gesdd, a.dup, jobz: job)
      when 'svd'
        gesvd = :"#{bchr}gesvd"
        s, u, vt, info = Numo::Linalg::Lapack.send(gesvd, a.dup, jobu: job, jobvt: job)
      else
        raise ArgumentError, "invalid driver: #{driver}"
      end

      raise LapackError, "the #{info.abs}-th argument had illegal value" if info.negative?
      raise LapackError, 'the input array has a NAN entry' if info == -4
      raise LapackError, 'the did not converge' if info.positive?

      [s, u, vt]
    end

    # Computes the matrix multiplication of two arrays.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 0], [0, 1]]
    #   b = Numo::DFloat[[4, 1], [2, 2]]
    #   pp Numo::Linalg.matmul(a, b)
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[4, 1],
    #   #  [2, 2]]
    #
    # @param a [Numo::NArray] The first array.
    # @param b [Numo::NArray] The second array.
    # @return [Numo::NArray] The matrix product of `a` and `b`.
    def matmul(a, b)
      Numo::Linalg::Blas.call(:gemm, a, b)
    end

    # Computes the matrix `a` raised to the power of `n`.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 2], [3, 4]]
    #   pp Numo::Linalg.matrix_power(a, 3)
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[37,  54],
    #   #  [81, 118]]
    #
    # @param a [Numo::NArray] The square matrix.
    # @param n [Integer] The exponent.
    # @return [Numo::NArray] The matrix `a` raised to the power of `n`.
    def matrix_power(a, n)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, "exponent n must be an integer: #{n}" unless n.is_a?(Integer)

      if n.zero?
        a.class.eye(a.shape[0])
      elsif n.positive?
        r = a.dup
        (n - 1).times { r = Numo::Linalg.matmul(r, a) }
        r
      else
        inv_a = inv(a)
        r = inv_a.dup
        (-n - 1).times { r = Numo::Linalg.matmul(r, inv_a) }
        r
      end
    end

    # Computes the singular values of a matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 2, 3], [2, 4, 6], [-1, 1, -1]]
    #   pp Numo::Linalg.svdvals(a)
    #   # => Numo::DFloat#shape=[3]
    #   # [8.38434, 1.64402, 5.41675e-17]
    #
    # @param a [Numo::NArray] Matrix to be decomposed.
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @return [Numo::NArray] The singular values of `a`.
    def svdvals(a, driver: 'sdd')
      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      case driver.to_s
      when 'sdd'
        gesdd = :"#{bchr}gesdd"
        s, _u, _vt, info = Numo::Linalg::Lapack.send(gesdd, a.dup, jobz: 'N')
      when 'svd'
        gesvd = :"#{bchr}gesvd"
        s, _u, _vt, info = Numo::Linalg::Lapack.send(gesvd, a.dup, jobu: 'N', jobvt: 'N')
      else
        raise ArgumentError, "invalid driver: #{driver}"
      end

      raise LapackError, "the #{info.abs}-th argument had illegal value" if info.negative?
      raise LapackError, 'the input array has a NAN entry' if info == -4
      raise LapackError, 'the decomposition did not converge' if info.positive?

      s
    end

    # Creates a diagonal matrix from the given singular values.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   s = Numo::DFloat[4, 2, 1]
    #   d = Numo::Linalg.diagsvd(s, 3, 4)
    #   pp d
    #   # =>
    #   # Numo::DFloat#shape=[3,4]
    #   # [[4, 0, 0, 0],
    #   #  [0, 2, 0, 0],
    #   #  [0, 0, 1, 0]]
    #   d = Numo::Linalg.diagsvd(s, 4, 3)
    #   pp d
    #   # =>
    #   # Numo::DFloat#shape=[4,3]
    #   # [[4, 0, 0],
    #   #  [0, 2, 0],
    #   #  [0, 0, 1],
    #   #  [0, 0, 0]]
    #
    # @param s [Numo::NArray] The singular values.
    # @param m [Integer] The number of rows of the constructed matrix.
    # @param n [Integer] The number of columns of the constructed matrix.
    # @return [Numo::NArray] The m-by-n diagonal matrix with the singular values on the diagonal.
    def diagsvd(s, m, n)
      sz_s = s.size
      raise ArgumentError, "size of s must be equal to m or n: s.size=#{sz_s}, m=#{m}, n=#{n}" if sz_s != m && sz_s != n

      mat = s.class.zeros(m, n)
      if sz_s == m
        mat[true, 0...m] = s.diag
      else
        mat[0...n, true] = s.diag
      end
      mat
    end

    # Computes an orthonormal basis for the range of `A` using SVD.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 2, 3], [2, 4, 6], [-1, 1, -1]]
    #   u = Numo::Linalg.orth(a)
    #   pp u
    #   # =>
    #   # Numo::DFloat#shape=[3,2]
    #   # [[-0.446229, -0.0296535],
    #   #  [-0.892459, -0.059307],
    #   #  [0.0663073, -0.997799]]
    #   pp u.transpose.dot(u)
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[1, -1.97749e-16],
    #   #  [-1.97749e-16, 1]]
    #
    # @param a [Numo::NArray] The m-by-n input matrix.
    # @param rcond [Float] The threshold value for small singular values of `a`, default value is `a.shape.max * EPS`.
    # @return [Numo::NArray] The orthonormal basis for the range of `a`.
    def orth(a, rcond: nil)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      s, u, = svd(a, driver: 'sdd', job: 'S')
      tol = if rcond.nil? || rcond.negative?
              a.shape.max * s.class::EPSILON
            else
              rcond
            end
      rank = s.gt(tol * s.max).count
      u[true, 0...rank].dup
    end

    # Computes an orthonormal basis for the null space of `A` using SVD.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 5).rand - 0.5
    #   n = Numo::Linalg.null_space(a)
    #   pp n
    #   # =>
    #   # Numo::DFloat#shape=[5,2]
    #   # [[0.214096, -0.404277],
    #   #  [-0.482225, -0.51557],
    #   #  [-0.584394, -0.246804],
    #   #  [0.596612, -0.351468],
    #   #  [-0.155434, 0.621535]]
    #   pp n.transpose.dot(n)
    #   # =>
    #   # Numo::DFloat#shape=[2,2]
    #   # [[1, 1.31078e-16],
    #   #  [1.31078e-16, 1]]
    #
    # @param a [Numo::NArray] The m-by-n input matrix.
    # @param rcond [Float] The threshold value for small singular values of `a`, default value is `a.shape.max * EPS`.
    # @return [Numo::NArray] The orthonormal basis for the null space of `a`.
    def null_space(a, rcond: nil)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      s, _u, vt = svd(a, driver: 'sdd', job: 'A')
      tol = if rcond.nil? || rcond.negative?
              a.shape.max * s.class::EPSILON
            else
              rcond
            end
      rank = s.gt(tol * s.max).count
      vt[rank...vt.shape[0], true].conj.transpose.dup
    end

    # Computes the LU decomposition of a matrix using partial pivoting.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 4).rand - 0.5
    #   pm, l, u = Numo::Linalg.lu(a)
    #   error = (pm.dot(l).dot(u) - a).abs.max
    #   pp error
    #   # => 5.551115123125783e-17
    #
    #   l, u = Numo::Linalg.lu(a, permute_l: true)
    #   error = (l.dot(u) - a).abs.max
    #   pp error
    #   # => 5.551115123125783e-17
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @param permute_l [Boolean] If true, returns `L` with the permutation applied.
    # @return [Array<Numo::NArray>] if `permute_l` is `false`, the permutation matrix `P`, lower-triangular matrix `L`, and
    #  upper-triangular matrix `U` ([P, L, U]). if `permute_l` is `true`, the permuted lower-triangular matrix `L` and
    #  upper-triangular matrix `U` ([L, U]).
    def lu(a, permute_l: false)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      m, n = a.shape
      k = [m, n].min
      lu, piv = lu_fact(a)
      l = lu.tril.tap { |nary| nary[nary.diag_indices] = 1 }[true, 0...k].dup
      u = lu.triu[0...k, 0...n].dup
      perm = a.class.eye(m).tap do |nary|
        piv.to_a.each_with_index { |i, j| nary[true, [i - 1, j]] = nary[true, [j, i - 1]].dup }
      end

      permute_l ? [perm.dot(l), u] : [perm, l, u]
    end

    # Computes the LU decomposition of a matrix using partial pivoting.
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @return [Array<Numo::NArray>] The LU decomposition and pivot indices ([lu, piv]).
    def lu_fact(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = :"#{bchr}getrf"
      lu, piv, info = Numo::Linalg::Lapack.send(getrf, a.dup)

      raise LapackError, "the #{info.abs}-th argument of getrf had illegal value" if info.negative?

      if info.positive?
        warn("the factorization has been completed, but the factor U[#{info - 1}, #{info - 1}] is " \
             'exactly zero, indicating that the matrix is singular.')
      end

      [lu, piv]
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` using the LU decomposition of `A`.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 3).rand
    #   b = Numo::DFloat.eye(3)
    #   lu, ipiv = Numo::Linalg.lu_fact(a)
    #   x = Numo::Linalg.lu_solve(lu, ipiv, b)
    #
    #   puts (b - a.dot(x)).abs.max
    #   => 2.220446049250313e-16
    #
    # @param lu [Numo::NArray] The LU decomposition of the n-by-n matrix `A`.
    # @param ipiv [Numo::Int32/Int64] The pivot indices from `lu_fact`.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param trans [String] The type of system to be solved.
    #  - 'N': solve `A * x = b` (No transpose),
    #  - 'T': solve `A^T * x = b` (Transpose),
    #  - 'C': solve `A^H * x = b` (Conjugate transpose).
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def lu_solve(lu, ipiv, b, trans: 'N')
      raise Numo::NArray::ShapeError, 'input array lu must be 2-dimensional' if lu.ndim != 2
      raise Numo::NArray::ShapeError, 'input array lu must be square' if lu.shape[0] != lu.shape[1]
      raise Numo::NArray::ShapeError, "incompatible dimensions: lu.shape[0] = #{lu.shape[0]} != b.shape[0] = #{b.shape[0]}" if lu.shape[0] != b.shape[0]
      raise ArgumentError, 'trans must be "N", "T", or "C"' unless %w[N T C].include?(trans)

      bchr = blas_char(lu)
      raise ArgumentError, "invalid array type: #{lu.class}" if bchr == 'n'

      getrs = :"#{bchr}getrs"
      x, info = Numo::Linalg::Lapack.send(getrs, lu, ipiv, b.dup)

      raise LapackError, "the #{info.abs}-th argument of getrs had illegal value" if info.negative?

      x
    end

    # Computes the Cholesky decomposition of a symmetric / Hermitian positive-definite matrix.
    #
    # @param a [Numo::NArray] The n-by-n symmetric / Hermitian positive-definite matrix.
    # @param uplo [String] The part of the matrix to be used ('U' or 'L').
    # @return [Numo::NArray] The upper- / lower-triangular matrix `U` / `L`.
    def cho_fact(a, uplo: 'U')
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, 'uplo must be "U" or "L"' unless %w[U L].include?(uplo)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potrf"
      c, info = Numo::Linalg::Lapack.send(fnc, a.dup, uplo: uplo)

      raise LapackError, "the #{-info}-th argument of #{fnc} had illegal value" if info.negative?

      if info.positive?
        raise LapackError, "the leading principal minor of order #{info} is not positive, " \
                           'and the factorization could not be completed.'
      end

      c
    end

    # Computes the orthogonal Procrustes problem.
    #
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[2, 0, 1], [-2, 0, 0]]
    #   b = a.fliplr
    #   r, scale = Numo::Linalg.orthogonal_procrustes(a, b)
    #
    #   pp b
    #   # =>
    #   # Numo::DFloat(view)#shape=[2,3]
    #   # [[1, 0, 2],
    #   #  [0, 0, -2]]
    #   pp a.dot(r)
    #   # =>
    #   # Numo::DFloat#shape=[2,3]
    #   # [[1, 0, 2],
    #   #  [1.58669e-16, 0, -2]]
    #   pp (b - a.dot(r)).abs.max
    #   # =>
    #   # 2.220446049250313e-16
    #
    # @param a [Numo::NArray] The first input matrix.
    # @param b [Numo::NArray] The second input matrix.
    # @return [Array<Numo::NArray, Float>] The orthogonal matrix `R` and the scale factor `scale`.
    def orthogonal_procrustes(a, b)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array b must be 2-dimensional' if b.ndim != 2
      raise Numo::NArray::ShapeError, "incompatible dimensions: a.shape = #{a.shape} != b.shape = #{b.shape}" if a.shape != b.shape

      m = b.transpose.dot(a.conj).transpose
      s, u, vt = svd(m, driver: 'svd', job: 'S')
      r = u.dot(vt)
      scale = s.sum
      [r, scale]
    end

    # Computes a diagonal similarity transformation that balances a square matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat[[1, 0, 0], [1, 2, 0], [1, 2, 3]]
    #   b, h = Numo::Linalg.matrix_balance(a)
    #   pp b
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[3, 2, 1],
    #   #  [0, 2, 1],
    #   #  [0, 0, 1]]
    #   pp h
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0, 0, 1],
    #   #  [0, 1, 0],
    #   #  [1, 0, 0]]
    #   pp (Numo::Linalg.inv(h).dot(a).dot(h) - b).abs.max
    #   # => 0.0
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param permute [Boolean] The flag indicating whether to permute the matrix.
    # @param scale [Boolean] The flag indicating whether to scale the matrix.
    # @param separate [Boolean] The flag indicating whether to return scaling factors and permutation indices
    #   separately.
    # @return [Array<Numo::NArray, Numo::NArray>] if `separate` is `false`, the balanced matrix and the
    #   similarity transformation matrix `H` ([b, h]). if `separate` is `true`, the balanced matrix, the
    #   scaling factors, and the permutation indices ([b, scaler, perm]).
    def matrix_balance(a, permute: true, scale: true, separate: false) # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/PerceivedComplexity
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      n = a.shape[0]
      raise ArgumentError, 'input array a must be square' if a.shape[1] != n

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      job = if permute && scale
              'B'
            elsif permute && !scale
              'P'
            elsif !permute && scale
              'S'
            else
              'N'
            end
      fnc = :"#{bchr}gebal"
      b, lo, hi, prm_scl, info = Numo::Linalg::Lapack.send(fnc, a.dup, job: job)

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?

      # convert from Fortran style index to Ruby style index.
      lo -= 1
      hi -= 1
      iprm_scl = Numo::Int32.cast(prm_scl) - 1

      # extract scaling factors
      scaler = prm_scl.class.ones(n)
      scaler[lo...(hi + 1)] = prm_scl[lo...(hi + 1)]

      # extract permutation indices
      perm = Numo::Int32.new(n).seq
      if hi < n - 1
        iprm_scl[(hi + 1)...n].to_a.reverse.each.with_index(1) do |s, i|
          j = n - i
          next if s == j

          tmp_ls, tmp_lj = perm[[s, j]].to_a
          tmp_rj, tmp_rs = perm[[j, s]].to_a
          perm[[s, j]] = [tmp_rj, tmp_rs]
          perm[[j, s]] = [tmp_ls, tmp_lj]
        end
      end
      if lo > 0 # rubocop:disable Style/NumericPredicate
        iprm_scl[0...lo].to_a.each_with_index do |s, j|
          next if s == j

          tmp_ls, tmp_lj = perm[[s, j]].to_a
          tmp_rj, tmp_rs = perm[[j, s]].to_a
          perm[[s, j]] = [tmp_rj, tmp_rs]
          perm[[j, s]] = [tmp_ls, tmp_lj]
        end
      end

      return [b, scaler, perm] if separate

      # construct inverse permutation matrix
      inv_perm = Numo::Int32.zeros(n)
      inv_perm[perm] = Numo::Int32.new(n).seq
      h = scaler.diag[inv_perm, true].dup

      [b, h]
    end

    # Computes the eigenvalues and right and/or left eigenvectors of a general square matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(5, 5).rand - 0.5
    #   w, _vl, vr = Numo::Linalg.eig(a)
    #   error = (a.dot(vr) - vr.dot(w.diag)).abs.max
    #   pp error
    #   # => 4.718447854656915e-16
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param left [Boolean] The flag indicating whether to compute the left eigenvectors.
    # @param right [Boolean] The flag indicating whether to compute the right eigenvectors.
    # @return [Array<Numo::NArray>] The eigenvalues, left eigenvectors, and right eigenvectors.
    def eig(a, left: false, right: true) # rubocop:disable  Metrics/AbcSize, Metrics/PerceivedComplexity
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      jobvl = left ? 'V' : 'N'
      jobvr = right ? 'V' : 'N'

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}geev"
      if %w[z c].include?(bchr)
        w, vl, vr, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvl: jobvl, jobvr: jobvr)
      else
        wr, wi, vl, vr, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvl: jobvl, jobvr: jobvr)
      end

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'the QR algorithm failed to compute all the eigenvalues.' if info.positive?

      if %w[d s].include?(bchr)
        w = wr + (wi * 1.0i)
        ids = wi.gt(0).where
        unless ids.empty?
          cast_class = bchr == 'd' ? Numo::DComplex : Numo::SComplex
          if left
            tmp = cast_class.cast(vl)
            tmp[true, ids].imag = vl[true, ids + 1]
            tmp[true, ids + 1] = tmp[true, ids].conj
            vl = tmp
          end
          if right
            tmp = cast_class.cast(vr)
            tmp[true, ids].imag = vr[true, ids + 1]
            tmp[true, ids + 1] = tmp[true, ids].conj
            vr = tmp
          end
        end
      end

      [w, left ? vl : nil, right ? vr : nil]
    end

    # Computes the eigenvalues of a general square matrix.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The eigenvalues.
    def eigvals(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}geev"
      if %w[z c].include?(bchr)
        w, _vl, _vr, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvl: 'N', jobvr: 'N')
      else
        wr, wi, _vl, _vr, info = Numo::Linalg::Lapack.send(fnc, a.dup, jobvl: 'N', jobvr: 'N')
        w = wr + (wi * 1.0i)
      end

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'the QR algorithm failed to compute all the eigenvalues.' if info.positive?

      w
    end

    # Computes the eigenvalues of a symmetric / Hermitian matrix by solving an ordinary / generalized eigenvalue problem.
    #
    # @param a [Numo::NArray] The n-by-n symmetric / Hermitian matrix.
    # @param b [Numo::NArray] The n-by-n symmetric / Hermitian matrix. If nil, identity matrix is assumed.
    # @param vals_range [Range/Array]
    #   The range of indices of the eigenvalues (in ascending order) and corresponding eigenvectors to be returned.
    #   If nil, all eigenvalues and eigenvectors are computed.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param turbo [Bool] The flag indicating whether to use a divide and conquer algorithm. If vals_range is given, this flag is ignored.
    # @return [Numo::NArray] The eigenvalues.
    def eigvalsh(a, b = nil, vals_range: nil, uplo: 'U', turbo: false)
      eigh(a, b, vals_only: true, vals_range: vals_range, uplo: uplo, turbo: turbo)[0]
    end

    # Computes the Bunch-Kaufman decomposition of a symmetric / Hermitian matrix.
    # The factorization has the form `A = U * D * U^T` or `A = L * D * L^T`,
    # where `U` (or `L`) is a product of permutation and unit upper
    # (lower) triangular matrices, and `D` is a block diagonal matrix.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(5, 5).rand
    #   a = 0.5 * (a + a.transpose)
    #   u, d, _perm = Numo::Linalg.ldl(a)
    #   error = (a - u.dot(d).dot(u.transpose)).abs.max
    #   pp error
    #   # => 5.551115123125783e-17
    #
    # @param a [Numo::NArray] The n-by-n symmetric / Hermitian matrix.
    # @param uplo [String] The part of the matrix to be used ('U' or 'L').
    # @param hermitian [Boolean] The flag indicating whether `a` is Hermitian.
    # @return [Array<Numo::NArray>] The permutated upper (lower) triangular matrix, the block diagonal matrix, and the permutation indices.
    def ldl(a, uplo: 'U', hermitian: true)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      complex = bchr =~ /c|z/
      fnc = complex && hermitian ? :"#{bchr}hetrf" : :"#{bchr}sytrf"
      lud = a.dup
      ipiv, info = Numo::Linalg::Lapack.send(fnc, lud, uplo: uplo)

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?

      if info.positive?
        warn("the factorization has been completed, but the D[#{info - 1}, #{info - 1}] is " \
             'exactly zero, indicating that the block diagonal matrix is singular.')
      end

      _lud_permutation(lud, ipiv, uplo: uplo, hermitian: hermitian)
    end

    # Compute the condition number of a matrix.
    #
    # @param a [Numo::NArray] The input matrix.
    # @param ord [String/Symbol/Integer] The order of the norm.
    # nil or 2: 2-norm using singular values, 'fro': Frobenius norm, 'info': infinity norm, and 1: 1-norm.
    # @return [Numo::NArray] The condition number of the matrix.
    def cond(a, ord = nil)
      if ord.nil? || ord == 2 || ord == -2
        svals = svdvals(a)
        if ord == -2
          svals[false, -1] / svals[false, 0]
        else
          svals[false, 0] / svals[false, -1]
        end
      else
        inv_a = inv(a)
        norm(a, ord, axis: [-2, -1]) * norm(inv_a, ord, axis: [-2, -1])
      end
    end

    # Computes the sign and natural logarithm of the determinant of a matrix.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Array<Float/Complex>] The sign and natural logarithm of the determinant of `a` ([sign, logdet]).
    def slogdet(a)
      lu, ipiv = lu_fact(a)
      dg = lu.diagonal
      return 0, (-Float::INFINITY) if dg.eq(0).any?

      idx = ipiv.class.new(ipiv.shape[-1]).seq(1)
      n_nonzero = ipiv.ne(idx).count(axis: -1)
      sign = ((-1.0)**(n_nonzero % 2)) * (dg / dg.abs).prod

      logdet = Numo::NMath.log(dg.abs).sum(axis: -1)

      [sign, logdet]
    end

    # Computes the rank of a matrix using SVD.
    #
    # @param a [Numo::NArray] The input matrix.
    # @param tol [Float] The threshold value for small singular values of `a`.
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @return [Integer] The rank of the matrix.
    def matrix_rank(a, tol: nil, driver: 'svd')
      return a.ne(0).any? ? 1 : 0 if a.ndim < 2

      s = svdvals(a, driver: driver)
      tol ||= s.max(axis: -1, keepdims: true) * (a.shape[-2..].max * s.class::EPSILON)
      s.gt(tol).count(axis: -1)
    end

    # Computes the least-squares solution to a linear matrix equation.
    #
    # @param a [Numo::NArray] The m-by-n input matrix.
    # @param b [Numo::NArray] The m-dimensional right-hand side vector or the m-by-nrhs right-hand side matrix.
    # @param driver [String] The LAPACK driver to be used (This argument is ignored, 'lsd' is always used).
    # @param rcond [Float] The threshold value for small singular values of `a`.
    # @return [Array<Numo::NArray, Float/Complex, Integer, Numo::NArray>] The least-squares solution matrix / vector `x`,
    #   the sum of squared residuals, the effective rank of `a`, and the singular values of `a`.
    def lstsq(a, b, driver: 'lsd', rcond: nil) # rubocop:disable Lint/UnusedMethodArgument, Metrics/AbcSize
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, "incompatible dimensions: a.shape[0] = #{a.shape[0]} != b.shape[0] = #{b.shape[0]}" if a.shape[0] != b.shape[0]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      m, n = a.shape
      if m < n
        if b.ndim == 1
          x = Numo::DFloat.zeros(n)
          x[0...b.size] = b
        else
          x = Numo::DFloat.zeros(n, b.shape[1])
          x[0...b.shape[0], 0...b.shape[1]] = b
        end
      else
        x = b.dup
      end

      fnc = :"#{bchr}gelsd"
      s, rank, info = Numo::Linalg::Lapack.send(fnc, a.dup, x, rcond: rcond)

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'the algorithm for computing the SVD failed to converge' if info.positive?

      resids = x.class[]
      if m > n
        if rank == n
          resids = if b.ndim == 1
                     (x[n..].abs**2).sum(axis: 0)
                   else
                     (x[n..-1, true].abs**2).sum(axis: 0)
                   end
        end
        x = if b.ndim == 1
              x[false, 0...n]
            else
              x[false, 0...n, true]
            end
      end

      [x, resids, rank, s]
    end

    # Computes the matrix exponential using a scaling and squaring algorithm with a Pade approximation.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param ord [Integer] The order of the Padé approximation.
    # @return [Numo::NArray] The matrix exponential of `a`.
    #
    # Reference:
    # - C. Moler and C. Van Loan, "Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later," SIAM Review, vol. 45, no. 1, pp. 3-49, 2003.
    def expm(a, ord = 8) # rubocop:disable Metrics/AbcSize
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      norm = a.abs.max
      n_sqr = norm.positive? ? [0, Math.log2(norm).to_i + 1].max : 0
      a /= 2**n_sqr

      x = a.dup
      c = 0.5
      sgn = 1
      nume = a.class.eye(a.shape[0]) + (c * a)
      deno = a.class.eye(a.shape[0]) - (c * a)
      (2..ord).each do |k|
        c *= (ord - k + 1).fdiv(k * ((2 * ord) - k + 1))
        x = a.dot(x)
        c_x = c * x
        nume += c_x
        deno += sgn * c_x
        sgn = -sgn
      end

      a_expm = Numo::Linalg.solve(deno, nume)
      n_sqr.times { a_expm = a_expm.dot(a_expm) }
      a_expm
    end

    # Computes the matrix logarithm using its eigenvalue decomposition.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix logarithm of `a`.
    def logm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      ev, vl, = eig(a, left: true, right: false)
      v = vl.transpose.conj
      inv_v = Numo::Linalg.inv(v)
      log_ev = Numo::NMath.log(ev)

      inv_v.dot(log_ev.diag).dot(v)
    end

    # Computes the matrix sine using the matrix exponential.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix sine of `a`.
    def sinm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      b = a * 1.0i
      if %w[z c].include?(bchr)
        -0.5i * (expm(b) - expm(-b))
      else
        expm(b).imag
      end
    end

    # Computes the matrix cosine using the matrix exponential.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix cosine of `a`.
    def cosm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      b = a * 1.0i
      if %w[z c].include?(bchr)
        0.5 * (expm(b) + expm(-b))
      else
        expm(b).real
      end
    end

    # Computes the matrix tangent.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix tangent of `a`.
    def tanm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      a_sin = sinm(a)
      a_cos = cosm(a)
      a_sin.dot(Numo::Linalg.inv(a_cos))
    end

    # Computes the matrix hyperbolic sine using the matrix exponential.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix hyperbolic sine of `a`.
    def sinhm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      0.5 * (expm(a) - expm(-a))
    end

    # Computes the matrix hyperbolic cosine using the matrix exponential.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix hyperbolic cosine of `a`.
    def coshm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      0.5 * (expm(a) + expm(-a))
    end

    # Computes the matrix hyperbolic tangent.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix hyperbolic tangent of `a`.
    def tanhm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      a_sinh = sinhm(a)
      a_cosh = coshm(a)
      a_sinh.dot(Numo::Linalg.inv(a_cosh))
    end

    # Computes the square root of a matrix using its eigenvalue decomposition.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix square root of `a`.
    def sqrtm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      ev, vl, = eig(a, left: true, right: false)
      v = vl.transpose.conj
      inv_v = Numo::Linalg.inv(v)
      sqrt_ev = Numo::NMath.sqrt(ev)

      inv_v.dot(sqrt_ev.diag).dot(v)
    end

    # Computes the matrix sign function using its inverse and square root matrices.
    #
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Numo::NArray] The matrix sign function of `a`.
    def signm(a)
      raise Numo::NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise Numo::NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      a_sqrt = sqrtm(a.dot(a))
      a_inv = Numo::Linalg.inv(a)
      a_inv.dot(a_sqrt)
    end

    # Computes the inverse of a matrix using its LU decomposition.
    #
    # @param lu [Numo::NArray] The LU decomposition of the n-by-n matrix `A`.
    # @param ipiv [Numo::Int32] The pivot indices from `lu_fact`.
    # @return [Numo::NArray] The inverse of the matrix `A`.
    def lu_inv(lu, ipiv)
      bchr = blas_char(lu)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}getri"
      inv, info = Numo::Linalg::Lapack.send(fnc, lu.dup, ipiv)

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?
      raise LapackError, 'the matrix is singular and its inverse could not be computed' if info.positive?

      inv
    end

    # Computes the inverse of a symmetric / Hermitian positive-definite matrix using its Cholesky decomposition.
    #
    # @example
    #   require 'numo/linalg'
    #
    #   a = Numo::DFloat.new(3, 5).rand - 0.5
    #   a = a.dot(a.transpose)
    #   c = Numo::Linalg.cho_fact(a)
    #   tri_inv_a = Numo::Linalg.cho_inv(c)
    #   tri_inv_a = tri_inv_a.triu
    #   inv_a = tri_inv_a + tri_inv_a.transpose - tri_inv_a.diagonal.diag
    #   error = (inv_a.dot(a) - Numo::DFloat.eye(3)).abs.max
    #   pp error
    #   # => 1.923726113137665e-15
    #
    # @param a [Numo::NArray] The Cholesky decomposition of the n-by-n symmetric / Hermitian positive-definite matrix.
    # @param uplo [String] The part of the matrix to be used ('U' or 'L').
    # @return [Numo::NArray] The upper- / lower-triangular matrix of the inverse of `a`.
    def cho_inv(a, uplo: 'U')
      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potri"
      inv, info = Numo::Linalg::Lapack.send(fnc, a.dup, uplo: uplo)

      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?

      if info.positive?
        raise LapackError, "the (#{info - 1}, #{info - 1})-th element of the factor U or L is zero, " \
                           'and the inverse could not be computed.'
      end

      inv
    end

    # @!visibility private
    def _lud_permutation(lud, ipiv, uplo: 'U', hermitian: true) # rubocop:disable Metrics/AbcSize,  Metrics/MethodLength, Metrics/PerceivedComplexity
      n = lud.shape[0]
      d = lud.class.zeros(n, n)
      perm = Numo::Int32.new(n).seq
      if uplo == 'U'
        u = lud.triu.tap { |m| m[m.diag_indices] = 1 }
        # If IPIV(k) > 0, then rows and columns k and IPIV(k) were interchanged
        # and D(k,k) is a 1-by-1 diagonal block.
        # IF UPLO = 'U' and If IPIV(k) = IPIV(k-1) < 0, then
        # rows and columns k-1 and -IPIV(k) were interchanged
        # and D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
        changed_2x2 = false
        n.times do |k|
          d[k, k] = lud[k, k]
          if ipiv[k].positive?
            i = ipiv[k] - 1
            u[[i, k], 0..k] = u[[k, i], 0..k].dup
            perm[[i, k]] = perm[[k, i]].dup
          elsif k.positive? && ipiv[k].negative? && ipiv[k] == ipiv[k - 1] && !changed_2x2
            i = -ipiv[k] - 1
            d[k - 1, k] = lud[k - 1, k]
            d[k, k - 1] = hermitian ? d[k - 1, k].conj : d[k - 1, k]
            u[k - 1, k] = 0
            u[[i, k - 1], 0..k] = u[[k - 1, i], 0..k].dup
            perm[[i, k - 1]] = perm[[k - 1, i]].dup
            changed_2x2 = true
            next
          end
          changed_2x2 = false if changed_2x2
        end
        [u, d, perm.sort_index]
      else
        l = lud.tril.tap { |m| m[m.diag_indices] = 1 }
        # If UPLO = 'L' and IPIV(k) = IPIV(k+1) < 0, then
        # rows and columns k+1 and -IPIV(k) were interchanged
        # and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
        changed_2x2 = false
        (n - 1).downto(0) do |k|
          d[k, k] = lud[k, k]
          if ipiv[k].positive?
            i = ipiv[k] - 1
            l[[i, k], k...n] = l[[k, i], k...n].dup
            perm[[i, k]] = perm[[k, i]].dup
          elsif k < n - 1 && ipiv[k].negative? && ipiv[k] == ipiv[k + 1] && !changed_2x2
            i = -ipiv[k] - 1
            d[k + 1, k] = lud[k + 1, k]
            d[k, k + 1] = hermitian ? d[k + 1, k].conj : d[k + 1, k]
            l[k + 1, k] = 0
            l[[i, k + 1], k...n] = l[[k + 1, i], k...n].dup
            perm[[i, k + 1]] = perm[[k + 1, i]].dup
            changed_2x2 = true
            next
          end
          changed_2x2 = false if changed_2x2
        end
        [l, d, perm.sort_index]
      end
    end

    private_class_method :_lud_permutation
  end
end
