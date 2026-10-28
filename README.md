# mini-symmetric-matrix-eigendecomposition

Diagonalizes real symmetric matrices using the cyclic Jacobi eigenvalue algorithm with zero third-party dependencies (standard library only).

## Usage

```python
from mini_symmetric_matrix_eigendecomposition import jacobi_eigen_decomposition

matrix = [
    [4.0, 1.0, 2.0],
    [1.0, 3.0, 0.0],
    [2.0, 0.0, 5.0],
]

eigenvalues, eigenvectors = jacobi_eigen_decomposition(matrix)
print(eigenvalues)      # ascending order, e.g. [2.0, 4.4..., 5.6...]
print(eigenvectors[0])  # first row of the eigenvector matrix
```

`jacobi_eigen_decomposition(matrix, *, tolerance=1e-12, max_sweeps=100)` returns a tuple `(eigenvalues, eigenvectors)`:

- `eigenvalues`: list of floats in ascending order.
- `eigenvectors`: list of lists forming an orthonormal matrix where column `k` is the eigenvector for `eigenvalues[k]`.

## Why this exists

This library solves the narrow problem of diagonalizing a real symmetric matrix when you cannot or do not want to depend on NumPy, SciPy, or any compiled extension. The trade-off is speed: the Jacobi method is `O(n^3)` per sweep and typically needs several sweeps, so it is best suited for small matrices (roughly `n <= 100`). The payoff is a pure-Python, dependency-free routine with deterministic behavior.

## Edge cases

- Booleans are rejected as entries, even though `isinstance(True, int)` is true in Python. They are not meaningful numeric scalars for this computation.
- Complex entries are rejected; this is a real-only routine.
- Non-convergence (controlled by `max_sweeps`) raises `RuntimeError` rather than returning silently.
- For repeated eigenvalues, the returned eigenvectors span the correct eigenspace and are orthonormal, but any valid orthonormal basis of that space is an acceptable result.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

