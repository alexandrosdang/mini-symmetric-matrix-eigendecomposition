import math
import unittest

from mini_symmetric_matrix_eigendecomposition import jacobi_eigen_decomposition


class TestJacobiEigendecomposition(unittest.TestCase):
    def test_diagonal_matrix_returns_diagonal_and_identity(self):
        m = [[2.0, 0.0], [0.0, 3.0]]
        evals, evecs = jacobi_eigen_decomposition(m)
        self.assertAlmostEqual(evals[0], 2.0, places=10)
        self.assertAlmostEqual(evals[1], 3.0, places=10)
        self.assertAlmostEqual(evecs[0][0], 1.0, places=10)
        self.assertAlmostEqual(evecs[1][1], 1.0, places=10)
        self.assertAlmostEqual(evecs[0][1], 0.0, places=10)
        self.assertAlmostEqual(evecs[1][0], 0.0, places=10)

    def test_2x2_off_diagonal(self):
        m = [[0.0, 1.0], [1.0, 0.0]]
        evals, evecs = jacobi_eigen_decomposition(m)
        self.assertAlmostEqual(evals[0], -1.0, places=10)
        self.assertAlmostEqual(evals[1], 1.0, places=10)

        # Verify A * v_k = lambda_k * v_k for each column.
        for k in range(2):
            vk = [evecs[i][k] for i in range(2)]
            avk = [sum(m[r][c] * vk[c] for c in range(2)) for r in range(2)]
            for r in range(2):
                self.assertAlmostEqual(avk[r], evals[k] * vk[r], places=9)

    def test_eigenvectors_are_orthonormal(self):
        m = [
            [4.0, 1.0, 2.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 5.0],
        ]
        n = 3
        evals, evecs = jacobi_eigen_decomposition(m)

        # Each column unit length.
        for k in range(n):
            norm = math.sqrt(sum(evecs[i][k] ** 2 for i in range(n)))
            self.assertAlmostEqual(norm, 1.0, places=9)

        # Columns mutually orthogonal.
        for k in range(n):
            for l in range(k + 1, n):
                dot = sum(evecs[i][k] * evecs[i][l] for i in range(n))
                self.assertAlmostEqual(dot, 0.0, places=9)

    def test_reconstruction(self):
        m = [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
        n = 3
        evals, evecs = jacobi_eigen_decomposition(m)

        # Reconstruct: V * diag(evals) * V^T == m.
        recon = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += evecs[i][k] * evals[k] * evecs[j][k]
                recon[i][j] = s
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(recon[i][j], m[i][j], places=8)

    def test_eigenvalues_sorted_ascending(self):
        m = [[5.0, 2.0], [2.0, 1.0]]
        evals, _ = jacobi_eigen_decomposition(m)
        self.assertLessEqual(evals[0], evals[1])

    def test_integer_input_accepted(self):
        m = [[2, 1], [1, 2]]
        evals, evecs = jacobi_eigen_decomposition(m)
        self.assertAlmostEqual(evals[0], 1.0, places=10)
        self.assertAlmostEqual(evals[1], 3.0, places=10)

    def test_non_symmetric_raises(self):
        m = [[1.0, 2.0], [3.0, 4.0]]
        with self.assertRaises(ValueError):
            jacobi_eigen_decomposition(m)

    def test_non_square_raises(self):
        m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError):
            jacobi_eigen_decomposition(m)

    def test_empty_matrix_raises(self):
        with self.assertRaises(ValueError):
            jacobi_eigen_decomposition([])

    def test_bool_entry_rejected(self):
        m = [[1.0, True], [True, 2.0]]
        with self.assertRaises(ValueError):
            jacobi_eigen_decomposition(m)

    def test_complex_entry_rejected(self):
        m = [[1.0, 2.0 + 0j], [2.0, 3.0]]
        with self.assertRaises(ValueError):
            jacobi_eigen_decomposition(m)

    def test_non_convergence_raises(self):
        m = [[1.0, 2.0], [2.0, 1.0]]
        with self.assertRaises(RuntimeError):
            jacobi_eigen_decomposition(m, max_sweeps=0)

    def test_repeated_eigenvalues_still_orthonormal(self):
        # Identity-like matrix with a repeated eigenvalue of 1.
        m = [[1.0, 0.0], [0.0, 1.0]]
        n = 2
        evals, evecs = jacobi_eigen_decomposition(m)
        self.assertAlmostEqual(evals[0], 1.0, places=10)
        self.assertAlmostEqual(evals[1], 1.0, places=10)
        for k in range(n):
            norm = math.sqrt(sum(evecs[i][k] ** 2 for i in range(n)))
            self.assertAlmostEqual(norm, 1.0, places=9)
        dot = sum(evecs[i][0] * evecs[i][1] for i in range(n))
        self.assertAlmostEqual(dot, 0.0, places=9)

    def test_4x4_diagonal_dominant(self):
        m = [
            [10.0, 1.0, 2.0, 3.0],
            [1.0, 20.0, 4.0, 5.0],
            [2.0, 4.0, 30.0, 6.0],
            [3.0, 5.0, 6.0, 40.0],
        ]
        n = 4
        evals, evecs = jacobi_eigen_decomposition(m)

        for i in range(n):
            for j in range(i + 1, n):
                self.assertLessEqual(evals[i], evals[j])

        for k in range(n):
            norm = math.sqrt(sum(evecs[i][k] ** 2 for i in range(n)))
            self.assertAlmostEqual(norm, 1.0, places=8)

        recon = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += evecs[i][k] * evals[k] * evecs[j][k]
                recon[i][j] = s
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(recon[i][j], m[i][j], places=7)


if __name__ == "__main__":
    unittest.main()
