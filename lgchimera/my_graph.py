"""MyGraph class and utilities

For modifying python-graphslam base graph class.

"""

import numpy as np
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from graphslam.graph import Graph


class MyGraph(Graph):
    
    def optimize(self, tol=1e-4, max_iter=20, fix_first_pose=True):
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
        print("\nIteration                chi^2        rel. change")
        print("---------                -----        -----------")

        for i in range(max_iter):
            self._calc_chi2_gradient_hessian()

            # Check for convergence (from the previous iteration); this avoids having to calculate chi^2 twice
            if i > 0:
                rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
                print("{:9d} {:20.4f} {:18.6f}".format(i, self._chi2, -rel_diff))
                if self._chi2 < chi2_prev and rel_diff < tol:
                    return
            else:
                print("{:9d} {:20.4f}".format(i, self._chi2))
                pass

            # Update the previous iteration's chi^2 error
            chi2_prev = self._chi2

            # Solve for the updates
            #print("Hessian eigenvalues: ", np.linalg.eig(self._hessian.toarray())[0])
            try:
                dx = spsolve(self._hessian, -self._gradient)  # pylint: disable=invalid-unary-operand-type
            except MatrixRankWarning:
                print("Warning: Hessian is rank deficient. Using pseudo-inverse.")
                print("This may be due to a fixed pose or a pose with no edges.")
                print("If this warning persists, try fixing more poses.")
                dx = np.linalg.pinv(self._hessian) @ -self._gradient

            # Apply the updates
            for v, dx_i in zip(self._vertices, np.split(dx, n)):
                v.pose += dx_i

        # If we reached the maximum number of iterations, print out the final iteration's results
        self.calc_chi2()
        rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
        print("{:9d} {:20.4f} {:18.6f}".format(max_iter, self._chi2, -rel_diff))