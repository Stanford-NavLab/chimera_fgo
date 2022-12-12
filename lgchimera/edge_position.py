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
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
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
        return self.estimate - (self.vertices[1].pose[:3] - self.vertices[0].pose[:3])


    def calc_jacobians(self):
        """Calculate the Jacobian of the edge's error with respect to each constrained pose.
        .. math::
           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]

        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        return [np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_other(self.vertices[0].pose)), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_self(self.vertices[0].pose)), self.vertices[1].pose.jacobian_boxplus())]


    def calc_chi2_loss(self, pose_0=None, pose_1=None):
        _, pose_0 = self.assign_pose(pose_0, 0)
        _, pose_1 = self.assign_pose(pose_1, 1)
        err = (self.estimate - (pose_1 - pose_0)).to_compact()
        chi2_loss = np.dot(np.dot(err, self.information), err)
        return chi2_loss


    def calc_chi2_gradient(self, pose_0=None, pose_1=None):
        _, pose_0 = self.assign_pose(pose_0, 0)
        _, pose_1 = self.assign_pose(pose_1, 1)
        err = (self.estimate - (pose_0 - pose_1)).to_compact()
        j_wrt_p0 = np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(pose_1 - pose_0), pose_1.jacobian_self_ominus_other_wrt_other(pose_0)), pose_0.jacobian_boxplus())
        j_wrt_p1 = np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(pose_1 - pose_0), pose_1.jacobian_self_ominus_other_wrt_self(pose_0)), pose_1.jacobian_boxplus())
        #TODO: Make this more compact
        grad_p0 = 2*np.dot(np.dot(err, self.information), j_wrt_p0)
        grad_p1 = 2*np.dot(np.dot(err, self.information), j_wrt_p1)
        grad = [grad_p0, grad_p1]
        return grad


    def calc_chi2_hessian(self, pose_0=None, pose_1=None):
        pose_0 = self.assign_pose(pose_0, 0)
        pose_1 = self.assign_pose(pose_1, 1)
        jacobians = self.calc_chi2_gradient(pose_0, pose_1)
        h_wrt_00 = np.dot(np.dot(np.transpose(jacobians[0]), self.information), jacobians[0])
        h_wrt_01 = np.dot(np.dot(np.transpose(jacobians[0]), self.information), jacobians[1])
        h_wrt_10 = np.dot(np.dot(np.transpose(jacobians[1]), self.information), jacobians[0])
        h_wrt_11 = np.dot(np.dot(np.transpose(jacobians[1]), self.information), jacobians[1])
        hessians = [h_wrt_00, h_wrt_01, h_wrt_10, h_wrt_11]
        return hessians
