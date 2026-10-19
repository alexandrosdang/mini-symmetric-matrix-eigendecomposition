from __future__ import annotations


def _is_real(x):
    """Check that x is a real int or float (not bool, not complex)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, complex):
        return False
    return isinstance(x, (int, float))


def _copy_matrix(a):
    return [row[:] for row in a]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def jacobi_eigen_decomposition(matrix, *, tolerance=1e-12, max_sweeps=100):
    """Diagonalize a real symmetric matrix using the cyclic Jacobi algorithm.

    Args:
        matrix: A list of lists forming a real symmetric matrix.
        tolerance: Off-diagonal Frobenius norm target.
        max_sweeps: Upper bound on full sweep iterations.

    Returns:
        eigenvalues: list of floats (ascending order).
        eigenvectors: list of lists; column eigenvectors[:, k]
            corresponds to eigenvalues[k].

    Raises:
        ValueError: on malformed or non-symmetric input.

    Notes:
        - Bools are rejected because they are not numeric scalars.
        - Complex values are rejected because this routine is real-only.
        - The cyclic sweep strategy visits (i, j) pairs in index order,
          which is deterministic and simple to reason about.
        - Non-convergence raises rather than returning silently.
        - Eigenvectors are normalized to unit Euclidean length.
        - For repeated eigenvalues, the columns of V are still orthonormal,
          but any vector in the degenerate eigenspace is acceptable.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty list of rows")
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("matrix must be square (list of equal-length rows)")
    for i in range(n):
        for j in range(n):
            if not _is_real(matrix[i][j]):
                raise ValueError(
                    f"matrix[{i}][{j}] must be a real number (int or float)"
                )
    for i in range(n):
        for j in range(i + 1, n):
            a = float(matrix[i][j])
            b = float(matrix[j][i])
            if abs(a - b) > tolerance:
                raise ValueError("matrix must be symmetric within tolerance")

    a = _copy_matrix(matrix)
    v = _identity(n)

    for sweep in range(max_sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        off_norm = (2.0 * off) ** 0.5

        if off_norm <= tolerance:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                app = a[p][p]
                aqq = a[q][q]

                if abs(apq) <= 1e-300:
                    continue

                phi = 0.5 * (aqq - app) / apq
                if phi >= 0.0:
                    t = 1.0 / (phi + (1.0 + phi * phi) ** 0.5)
                else:
                    t = -1.0 / (-phi + (1.0 + phi * phi) ** 0.5)

                c = 1.0 / (1.0 + t * t) ** 0.5
                s = t * c

                # Apply Givens rotation to A (columns p, q)
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq

                # Apply to rows p, q (must be second, distinct step)
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk

                # Accumulate rotation in V
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
        else:
            continue
        break
    else:
        raise RuntimeError("Jacobi algorithm failed to converge within max_sweeps")

    eigenvalues = [a[i][i] for i in range(n)]

    # Sort ascending; each eigenvalue keeps its eigenvector column.
    order = sorted(range(n), key=lambda k: eigenvalues[k])
    sorted_eigenvalues = [eigenvalues[k] for k in order]
    sorted_eigenvectors = [[v[i][k] for k in order] for i in range(n)]

    # Normalize columns to unit length (guard against zero vectors).
    for k in range(n):
        col_norm_sq = 0.0
        for i in range(n):
            val = sorted_eigenvectors[i][k]
            col_norm_sq += val * val
        if col_norm_sq > 0.0:
            inv = 1.0 / (col_norm_sq ** 0.5)
            for i in range(n):
                sorted_eigenvectors[i][k] *= inv

    return sorted_eigenvalues, sorted_eigenvectors
