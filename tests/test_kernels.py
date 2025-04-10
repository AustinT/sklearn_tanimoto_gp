# tests/test_kernels.py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.gaussian_process.kernels import Kernel  # For comparison/sanity check

from sklearn_tanimoto_gp import (
    DotProductTanimoto,
    MinMaxTanimoto,
    Tanimoto,
    TanimotoBinary,
)

# Sample data (non-negative often assumed for Tanimoto)
X1 = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
Y1 = np.array([[1.0, 1.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

X_zeros = np.zeros((2, 3))
X_single_zero = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

kernel_classes = [MinMaxTanimoto, Tanimoto, DotProductTanimoto, TanimotoBinary]


@pytest.mark.parametrize("KernelClass", kernel_classes)
def test_kernel_inheritance(KernelClass):
    """Test if kernels inherit correctly."""
    kernel = KernelClass()
    assert isinstance(kernel, Kernel)


@pytest.mark.parametrize("KernelClass", kernel_classes)
def test_kernel_diag_is_one(KernelClass):
    """Test if the diagonal of k(X, X) is always 1."""
    kernel = KernelClass()
    diag_X1 = kernel.diag(X1)
    diag_zeros = kernel.diag(X_zeros)
    diag_single_zero = kernel.diag(X_single_zero)

    assert_allclose(diag_X1, np.ones(X1.shape[0]))
    assert_allclose(diag_zeros, np.ones(X_zeros.shape[0]))
    assert_allclose(diag_single_zero, np.ones(X_single_zero.shape[0]))


@pytest.mark.parametrize("KernelClass", kernel_classes)
def test_kernel_is_not_stationary(KernelClass):
    """Test that the kernels report as non-stationary."""
    kernel = KernelClass()
    assert not kernel.is_stationary()


@pytest.mark.parametrize("KernelClass", kernel_classes)
def test_kernel_repr(KernelClass):
    """Test the __repr__ method."""
    kernel = KernelClass()
    assert repr(kernel) == f"{KernelClass.__name__}()"


# --- MinMaxTanimoto Specific Tests ---


def test_minmax_tanimoto_values_kXX():
    """Test MinMaxTanimoto k(X, X) values."""
    kernel = MinMaxTanimoto()
    K = kernel(X1)

    # Expected values calculated manually or using definition
    # k(x,y) = sum(min(xi, yi)) / sum(max(xi, yi))
    # k(X1[0], X1[0]) = (1+2+0)/(1+2+0) = 1
    # k(X1[1], X1[1]) = (3+0+1)/(3+0+1) = 1
    # k(X1[2], X1[2]) = (0+0+0)/(0+0+0) = 1 (by definition)
    # k(X1[3], X1[3]) = (4+5+6)/(4+5+6) = 1

    # k(X1[0], X1[1]) = min(1,3)+min(2,0)+min(0,1) / max(1,3)+max(2,0)+max(0,1)
    #                = (1+0+0) / (3+2+1) = 1 / 6
    # k(X1[0], X1[2]) = (0+0+0) / (1+2+0) = 0 / 3 = 0
    # k(X1[0], X1[3]) = min(1,4)+min(2,5)+min(0,6) / max(1,4)+max(2,5)+max(0,6)
    #                = (1+2+0) / (4+5+6) = 3 / 15 = 0.2

    # k(X1[1], X1[2]) = (0+0+0) / (3+0+1) = 0 / 4 = 0
    # k(X1[1], X1[3]) = min(3,4)+min(0,5)+min(1,6) / max(3,4)+max(0,5)+max(1,6)
    #                = (3+0+1) / (4+5+6) = 4 / 15 = 0.2666...

    # k(X1[2], X1[3]) = (0+0+0) / (4+5+6) = 0 / 15 = 0

    expected_K = np.array(
        [[1.0, 1 / 6, 0.0, 3 / 15], [1 / 6, 1.0, 0.0, 4 / 15], [0.0, 0.0, 1.0, 0.0], [3 / 15, 4 / 15, 0.0, 1.0]]
    )

    assert_allclose(K, expected_K)


def test_minmax_tanimoto_values_kXY():
    """Test MinMaxTanimoto k(X, Y) values."""
    kernel = MinMaxTanimoto()
    K = kernel(X1, Y1)

    # k(X1[0], Y1[0]=1,1,1) = min(1,1)+min(2,1)+min(0,1) / max(1,1)+max(2,1)+max(0,1)
    #                      = (1+1+0) / (1+2+1) = 2 / 4 = 0.5
    # k(X1[0], Y1[1]=3,0,1) = min(1,3)+min(2,0)+min(0,1) / max(1,3)+max(2,0)+max(0,1)
    #                      = (1+0+0) / (3+2+1) = 1 / 6
    # k(X1[0], Y1[2]=0,0,0) = 0 / (1+2+0) = 0

    # k(X1[1], Y1[0]=1,1,1) = min(3,1)+min(0,1)+min(1,1) / max(3,1)+max(0,1)+max(1,1)
    #                      = (1+0+1) / (3+1+1) = 2 / 5 = 0.4
    # k(X1[1], Y1[1]=3,0,1) = (3+0+1) / (3+0+1) = 1.0
    # k(X1[1], Y1[2]=0,0,0) = 0 / (3+0+1) = 0

    # k(X1[2], Y1[0]=1,1,1) = 0 / (1+1+1) = 0
    # k(X1[2], Y1[1]=3,0,1) = 0 / (3+0+1) = 0
    # k(X1[2], Y1[2]=0,0,0) = 0 / 0 = 1.0 (by definition)

    # k(X1[3], Y1[0]=1,1,1) = min(4,1)+min(5,1)+min(6,1) / max(4,1)+max(5,1)+max(6,1)
    #                      = (1+1+1) / (4+5+6) = 3 / 15 = 0.2
    # k(X1[3], Y1[1]=3,0,1) = min(4,3)+min(5,0)+min(6,1) / max(4,3)+max(5,0)+max(6,1)
    #                      = (3+0+1) / (4+5+6) = 4 / 15
    # k(X1[3], Y1[2]=0,0,0) = 0 / (4+5+6) = 0

    expected_K = np.array([[0.5, 1 / 6, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 4 / 15, 0.0]])

    assert_allclose(K, expected_K)


# --- DotProductTanimoto Specific Tests ---


def test_dotprod_tanimoto_values_kXX():
    """Test DotProductTanimoto k(X, X) values."""
    kernel = DotProductTanimoto()
    K = kernel(X1)

    # k(x,y) = <x,y> / (|x|^2 + |y|^2 - <x,y>)
    # Norms sq: |X1[0]|^2 = 1+4 = 5
    #           |X1[1]|^2 = 9+1 = 10
    #           |X1[2]|^2 = 0
    #           |X1[3]|^2 = 16+25+36 = 77

    # Diagonals k(x,x) = |x|^2 / (|x|^2 + |x|^2 - |x|^2) = |x|^2 / |x|^2 = 1 (for non-zero x)
    # k(0, 0) = 1 by definition. All diags should be 1.

    # k(X1[0], X1[1]) = <(1,2,0), (3,0,1)> / (|X1[0]|^2 + |X1[1]|^2 - <...>)
    #                = (1*3 + 2*0 + 0*1) / (5 + 10 - 3)
    #                = 3 / (15 - 3) = 3 / 12 = 0.25
    # k(X1[0], X1[2]=0) = 0 / (5 + 0 - 0) = 0
    # k(X1[0], X1[3]) = <(1,2,0), (4,5,6)> / (5 + 77 - <...>)
    #                = (1*4 + 2*5 + 0*6) / (82 - 14)
    #                = 14 / 68 = 7 / 34 approx 0.20588

    # k(X1[1], X1[2]=0) = 0 / (10 + 0 - 0) = 0
    # k(X1[1], X1[3]) = <(3,0,1), (4,5,6)> / (10 + 77 - <...>)
    #                = (3*4 + 0*5 + 1*6) / (87 - 18)
    #                = 18 / 69 = 6 / 23 approx 0.26087

    # k(X1[2]=0, X1[3]) = 0 / (0 + 77 - 0) = 0

    expected_K = np.array(
        [[1.0, 3 / 12, 0.0, 14 / 68], [3 / 12, 1.0, 0.0, 18 / 69], [0.0, 0.0, 1.0, 0.0], [14 / 68, 18 / 69, 0.0, 1.0]]
    )

    assert_allclose(K, expected_K)


def test_dotprod_tanimoto_values_kXY():
    """Test DotProductTanimoto k(X, Y) values."""
    kernel = DotProductTanimoto()
    K = kernel(X1, Y1)

    # Norms sq X1: 5, 10, 0, 77
    # Norms sq Y1: |Y1[0]|^2 = 1+1+1=3
    #              |Y1[1]|^2 = 9+0+1=10
    #              |Y1[2]|^2 = 0

    # k(X1[0], Y1[0]) = <(1,2,0), (1,1,1)> / (5 + 3 - <...>)
    #                = (1*1 + 2*1 + 0*1) / (8 - 3) = 3 / 5 = 0.6
    # k(X1[0], Y1[1]) = <(1,2,0), (3,0,1)> / (5 + 10 - <...>)
    #                = (1*3 + 2*0 + 0*1) / (15 - 3) = 3 / 12 = 0.25
    # k(X1[0], Y1[2]=0) = 0 / (5 + 0 - 0) = 0

    # k(X1[1], Y1[0]) = <(3,0,1), (1,1,1)> / (10 + 3 - <...>)
    #                = (3*1 + 0*1 + 1*1) / (13 - 4) = 4 / 9 approx 0.444
    # k(X1[1], Y1[1]) = <(3,0,1), (3,0,1)> / (10 + 10 - <...>)
    #                = 10 / (20 - 10) = 10 / 10 = 1.0
    # k(X1[1], Y1[2]=0) = 0 / (10 + 0 - 0) = 0

    # k(X1[2]=0, Y1[0]) = 0 / (0 + 3 - 0) = 0
    # k(X1[2]=0, Y1[1]) = 0 / (0 + 10 - 0) = 0
    # k(X1[2]=0, Y1[2]=0) = 0 / (0 + 0 - 0) = 1.0 (by definition)

    # k(X1[3], Y1[0]) = <(4,5,6), (1,1,1)> / (77 + 3 - <...>)
    #                = (4*1+5*1+6*1) / (80 - 15) = 15 / 65 = 3 / 13 approx 0.2307
    # k(X1[3], Y1[1]) = <(4,5,6), (3,0,1)> / (77 + 10 - <...>)
    #                = (4*3+5*0+6*1) / (87 - 18) = 18 / 69 = 6 / 23 approx 0.26087
    # k(X1[3], Y1[2]=0) = 0 / (77 + 0 - 0) = 0

    expected_K = np.array([[0.6, 0.25, 0.0], [4 / 9, 1.0, 0.0], [0.0, 0.0, 1.0], [15 / 65, 18 / 69, 0.0]])

    assert_allclose(K, expected_K)


def test_tanimoto_aliases():
    """Check that Tanimoto is an alias for MinMaxTanimoto and TanimotoBinary is an alias for DotProductTanimoto."""
    assert Tanimoto is MinMaxTanimoto
    assert TanimotoBinary is DotProductTanimoto
