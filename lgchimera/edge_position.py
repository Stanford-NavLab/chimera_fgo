# Copyright (c) 2020 Jeff Irion and contributors
#TODO: Add licence document from python-graphslam

"""A class for position-only (no orientation) edges.
"""


import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from graphslam.pose.r2 import PoseR2
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se3 import PoseSE3

from graphslam.edge.base_edge import BaseEdge


class EdgePosition(BaseEdge):
    """A class for representing position edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : PoseR3
        The expected measurement :math:`\mathbf{z}_j`

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : PoseR3
        The expected measurement :math:`\mathbf{z}_j`
    
    """

    def calc_error(self):
        """Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.
        .. math::
           \mathbf{e}_j = \mathbf{z}_j - (p_2 \ominus p_1)

        Returns
        -------
        np.ndarray
            The error for the edge

        """
        print("calc error")
        print(self.estimate)
        print(self.vertices[0].pose)
        #return (self.estimate - (self.vertices[1].pose - self.vertices[0].pose)).to_compact()
        #return np.hstack((self.estimate - (self.vertices[1].pose[:3].to_array() - self.vertices[0].pose[:3].to_array()), np.zeros(3)))
        return np.hstack((self.estimate - self.vertices[0].pose[:3].to_array(), np.zeros(3)))


    def calc_jacobians(self):
        """Calculate the Jacobian of the edge's error with respect to each constrained pose.
        .. math::
           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]

        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        print("HELLO")
        # print([np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_other(self.vertices[0].pose)), self.vertices[0].pose.jacobian_boxplus()),
        #         np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_self(self.vertices[0].pose)), self.vertices[1].pose.jacobian_boxplus())])
        Ji = np.zeros((6,6))
        Ji[:3,:3] = -np.eye(3)
        Jj = np.zeros((6,6))
        #Jj[:3,:3] = np.eye(3)
        return [Ji]
        #return [A, np.eye(6)]

    
    # def calc_chi2_gradient_hessian(self):
    #     pass