"""MyGraph class and utilities

For modifying python-graphslam base graph class.

"""

import numpy as np
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from graphslam.graph import Graph
from collections import defaultdict
from scipy.sparse import lil_matrix
from functools import reduce

import time
from numba import njit
from numba.experimental import jitclass


# pylint: disable=too-few-public-methods
class _Chi2GradientHessian:
    r"""A class that is used to aggregate the :math:`\chi^2` error, gradient, and Hessian.

    Parameters
    ----------
    dim : int
        The compact dimensionality of the poses

    Attributes
    ----------
    chi2 : float
        The :math:`\chi^2` error
    dim : int
        The compact dimensionality of the poses
    gradient : defaultdict
        The contributions to the gradient vector
    hessian : defaultdict
        The contributions to the Hessian matrix

    """
    def __init__(self, dim):
        self.chi2 = 0.
        self.dim = dim
        self.gradient = defaultdict(lambda: np.zeros(dim))
        self.hessian = defaultdict(lambda: np.zeros((dim, dim)))


    @staticmethod
    def update(chi2_grad_hess, incoming):
        r"""Update the :math:`\chi^2` error and the gradient and Hessian dictionaries.
        Parameters
        ----------
        chi2_grad_hess : _Chi2GradientHessian
            The ``_Chi2GradientHessian`` that will be updated
        incoming : tuple
            TODO
        """
        chi2_grad_hess.chi2 += incoming[0]

        for idx, contrib in incoming[1].items():
            chi2_grad_hess.gradient[idx] += contrib

        for (idx1, idx2), contrib in incoming[2].items():
            if idx1 <= idx2:
                chi2_grad_hess.hessian[idx1, idx2] += contrib
            else:
                chi2_grad_hess.hessian[idx2, idx1] += np.transpose(contrib)

        return chi2_grad_hess


class MyGraph(Graph):
    """MyGraph class.
    
    This class is a wrapper around the graphslam Graph class.
    
    """
    
    def _calc_chi2_gradient_hessian(self):
        r"""Calculate the :math:`\chi^2` error, the gradient :math:`\mathbf{b}`, and the Hessian :math:`H`.

        """
        n = len(self._vertices)
        dim = len(self._vertices[0].pose.to_compact())
        chi2_gradient_hessian = reduce(_Chi2GradientHessian.update, (e.calc_chi2_gradient_hessian() for e in self._edges), _Chi2GradientHessian(dim))

        self._chi2 = chi2_gradient_hessian.chi2

        # Fill in the gradient vector
        self._gradient = np.zeros(n * dim, dtype=np.float64)
        for idx, contrib in chi2_gradient_hessian.gradient.items():
            # If a vertex is fixed, its block in the gradient vector is zero and so there is nothing to do
            if idx not in self._fixed_vertices:
                self._gradient[idx * dim: (idx + 1) * dim] += contrib

        # Fill in the Hessian matrix
        self._hessian = lil_matrix((n * dim, n * dim), dtype=np.float64)
        for (row_idx, col_idx), contrib in chi2_gradient_hessian.hessian.items():
            if row_idx in self._fixed_vertices or col_idx in self._fixed_vertices:
                # For fixed vertices, the diagonal block is the identity matrix and the off-diagonal blocks are zero
                if row_idx == col_idx:
                    self._hessian[row_idx * dim: (row_idx + 1) * dim, col_idx * dim: (col_idx + 1) * dim] = np.eye(dim)
                continue

            self._hessian[row_idx * dim: (row_idx + 1) * dim, col_idx * dim: (col_idx + 1) * dim] = contrib

            if row_idx != col_idx:
                self._hessian[col_idx * dim: (col_idx + 1) * dim, row_idx * dim: (row_idx + 1) * dim] = np.transpose(contrib)



    def optimize(self, tol=1e-4, max_iter=20, fix_first_pose=True, verbose=False):
        r"""Optimize the :math:`\chi^2` error for the ``Graph``.

        Parameters
        ----------
        tol : float
            If the relative decrease in the :math:`\chi^2` error between iterations is less than ``tol``, we will stop
        max_iter : int
            The maximum number of iterations
        fix_first_pose : bool
            If ``True``, we will fix the first pose

        """
        n = len(self._vertices)

        if fix_first_pose:
            self._vertices[0].fixed = True

        # Populate the set of fixed vertices
        self._fixed_vertices = {i for i, v in enumerate(self._vertices) if v.fixed}

        # Previous iteration's chi^2 error
        chi2_prev = -1.

        # For displaying the optimization progress
        if verbose:
            print("\nIteration                chi^2        rel. change")
            print("---------                -----        -----------")

        # DEBUG: calc hessian and spsolve timing
        calc_time = 0
        solve_time = 0

        for i in range(max_iter):
            start_time = time.time()
            self._calc_chi2_gradient_hessian()
            calc_time += time.time() - start_time

            # Check for convergence (from the previous iteration); this avoids having to calculate chi^2 twice
            if i > 0:
                rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
                if verbose:
                    print("{:9d} {:20.4f} {:18.6f}".format(i, self._chi2, -rel_diff))
                if self._chi2 < chi2_prev and rel_diff < tol:
                    print("Calc time: ", calc_time)
                    print("Solve time: ", solve_time)
                    return
            else:
                if verbose:
                    print("{:9d} {:20.4f}".format(i, self._chi2))
                pass

            # Update the previous iteration's chi^2 error
            chi2_prev = self._chi2

            # Solve for the updates
            #hessian_evals = np.linalg.eig(self._hessian.toarray())[0]
            #print("Min hessian eigenvalue: ", np.min(hessian_evals))
            #print("Hessian eigenvalues: ", hessian_evals)
            #print("Hessian ", self._hessian.toarray())
            #print("Gradient ", self._gradient)
            # if np.any(hessian_evals == 0):
            #     print("Matrix singular")
            # else:
            #     dx = spsolve(self._hessian, -self._gradient)  
            start_time = time.time()
            dx = spsolve(self._hessian, -self._gradient)  # pylint: disable=invalid-unary-operand-type
            solve_time += time.time() - start_time

            # Apply the updates
            for v, dx_i in zip(self._vertices, np.split(dx, n)):
                v.pose += dx_i

        # If we reached the maximum number of iterations, print out the final iteration's results
        self.calc_chi2()
        rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
        if verbose:
            print("{:9d} {:20.4f} {:18.6f}".format(max_iter, self._chi2, -rel_diff))
        
        print("Calc time: ", calc_time)
        print("Solve time: ", solve_time)