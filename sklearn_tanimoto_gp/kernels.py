# sklearn_tanimoto_gp/kernels.py
import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.metrics.pairwise import check_pairwise_arrays


class MinMaxTanimoto(Kernel):
    """MinMax Tanimoto kernel for Gaussian Processes.

    Also known as the Jaccard Index or Tanimoto Coefficient for continuous
    or count data when defined using min and max operations.

    The kernel is defined as::

        k(x, y) = sum(min(x_i, y_i)) / sum(max(x_i, y_i))

    where x and y are feature vectors.

    This kernel is non-stationary. It's often used for non-negative data,
    such as count data or molecular fingerprints represented as dense vectors.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_tanimoto_gp.kernels import MinMaxTanimoto
    >>> X = np.array([[1, 2, 0], [3, 0, 1], [0, 0, 0]])
    >>> kernel = MinMaxTanimoto()
    >>> kernel(X)
    array([[1.        , 0.2       , 0.        ],
           [0.2       , 1.        , 0.        ],
           [0.        , 0.        , 1.        ]]) # k(0, 0) defined as 1
    >>> kernel(X, np.array([[1, 1, 1]]))
    array([[0.5       ],
           [0.25      ],
           [0.        ]])
    """

    def __init__(self):
        # No hyperparameters for this kernel
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed. Ignored for this kernel
            as it has no hyperparameters.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """
        if eval_gradient:
            raise ValueError(f"{self.__class__.__name__} kernel does not support gradient evaluation.")

        X, Y = check_pairwise_arrays(X, Y)

        # Calculate pairwise sum(min(x_i, y_i)) using broadcasting
        # Shape: (n_samples_X, n_samples_Y, n_features)
        min_matrix = np.minimum(X[:, np.newaxis, :], Y[np.newaxis, :, :])
        # Shape: (n_samples_X, n_samples_Y)
        numerator = np.sum(min_matrix, axis=2)

        # Calculate pairwise sum(max(x_i, y_i)) using broadcasting
        # Shape: (n_samples_X, n_samples_Y, n_features)
        max_matrix = np.maximum(X[:, np.newaxis, :], Y[np.newaxis, :, :])
        # Shape: (n_samples_X, n_samples_Y)
        denominator = np.sum(max_matrix, axis=2)

        # Handle case where denominator is zero (i.e., X[i] == 0 and Y[j] == 0)
        # In this case, numerator is also zero. Define k(0, 0) = 1.
        K = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator, dtype=np.float64),  # Output array, init with 1s for k(0,0)
            where=(denominator != 0),  # Condition to perform division
        )

        return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)).
        k(x, x) = sum(min(x_i, x_i)) / sum(max(x_i, x_i))
                = sum(x_i) / sum(x_i)
        This equals 1 if sum(x_i) != 0, and is 0/0 if sum(x_i) == 0.
        We define k(0, 0) = 1.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, X)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # k(x, x) is always 1, including the k(0, 0) = 1 definition
        return np.ones(X.shape[0], dtype=np.float64)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    # get_params, set_params, clone are inherited from Kernel and work
    # correctly as there are no hyperparameters.


class DotProductTanimoto(Kernel):
    """Dot Product Tanimoto kernel for Gaussian Processes.

    Also known as the Bhattacharyya coefficient or cosine similarity
    variant depending on context and data normalization.

    The kernel is defined as::

        k(x, y) = <x, y> / (|x|^2 + |y|^2 - <x, y>)

    where <x, y> is the dot product (inner product) and |x|^2 = <x, x>.

    This kernel is non-stationary.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_tanimoto_gp.kernels import DotProductTanimoto
    >>> X = np.array([[1., 2.], [3., 0.], [0., 0.]])
    >>> kernel = DotProductTanimoto()
    >>> kernel(X)
    array([[ 1.        ,  0.15      ,  0.        ],
           [ 0.15      ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]]) # k(0,0) defined as 1
    >>> Y = np.array([[1., 1.]])
    >>> kernel(X, Y)
    array([[ 0.75      ],
           [ 0.3       ],
           [ 0.        ]])
    """

    def __init__(self):
        # No hyperparameters for this kernel
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed. Ignored for this kernel
            as it has no hyperparameters.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """
        if eval_gradient:
            raise ValueError(f"{self.__class__.__name__} kernel does not support gradient evaluation.")

        X, Y = check_pairwise_arrays(X, Y)

        # Calculate pairwise dot products <x, y>
        # Shape: (n_samples_X, n_samples_Y)
        numerator = X @ Y.T

        # Calculate squared norms |x|^2 and |y|^2
        # Shape: (n_samples_X,)
        norm_X_sq = np.sum(X**2, axis=1)
        # Shape: (n_samples_Y,) if Y is not X, else (n_samples_X,)
        if Y is X:
            norm_Y_sq = norm_X_sq
        else:
            norm_Y_sq = np.sum(Y**2, axis=1)

        # Calculate denominator: |x|^2 + |y|^2 - <x, y> using broadcasting
        # Shapes: (n_samples_X, 1) + (1, n_samples_Y) - (n_samples_X, n_samples_Y)
        # Result shape: (n_samples_X, n_samples_Y)
        denominator = norm_X_sq[:, np.newaxis] + norm_Y_sq[np.newaxis, :] - numerator

        # Handle case where denominator is zero (i.e., X[i] == 0 and Y[j] == 0)
        # In this case, numerator is also zero. Define k(0, 0) = 1.
        K = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator, dtype=np.float64),  # Output array, init with 1s for k(0,0)
            where=(denominator != 0),  # Condition to perform division
        )

        return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)).
        k(x, x) = <x, x> / (|x|^2 + |x|^2 - <x, x>)
                = |x|^2 / (2*|x|^2 - |x|^2)
                = |x|^2 / |x|^2
        This equals 1 if |x|^2 != 0, and is 0/0 if x = 0.
        We define k(0, 0) = 1.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        # k(x, x) is always 1, including the k(0, 0) = 1 definition
        return np.ones(X.shape[0], dtype=np.float64)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    # get_params, set_params, clone are inherited from Kernel and work
    # correctly as there are no hyperparameters.
