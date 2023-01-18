"""PoseGraph class and utilities

This module defines the PoseGraph class and relevant utilities.

"""

import numpy as np
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3
from graphslam.pose.r3 import PoseR3
import plotly.graph_objects as go

from chimera_fgo.geom_util import R_to_quat, quat_to_R
from chimera_fgo.general import SuppressPrint
from chimera_fgo.edge_position import EdgePosition
from chimera_fgo.my_graph import MyGraph


class PositionGraph:
    """PositionGraph class.

    This class is a wrapper around the graphslam Graph class. 

    Vertices: nodes (positions) and factors (GPS positions)

    Node indexing starts at 1
    Vertex index for node i is -i

    Attributes
    ----------
    graph : Graph
        graphslam Graph
    
    
    Methods
    -------
    plot()

    """

    def __init__(self):
        """Constructor

        Initialize a PoseGraph (default empty)

        """
        self.graph = MyGraph([], [])
        # self.graph._edges = edges
        # self.graph._vertices = vertices
        

    def add_node(self, id, position):
        """Add node with ID and pose

        Parameters
        ----------
        id : int
            Vertex ID
        pose : tuple (R,t)
            Tuple of rotation and translation
        
        """
        p = PoseR3(position)
        v = Vertex(id, p)
        self.graph._vertices.append(v)

    
    def add_edge(self, ids, translation, information=np.eye(3)):
        """Add vertex with ID and relative translation

        Parameters
        ----------
        ids : list
            Pair of vertex IDs for this edge
        translation : np.array (3 x 1)
            Translation
        information : np.array (3 x 3)
            Information matrix
        
        """
        estimate = PoseR3(translation)
        e = EdgeOdometry(ids, information, estimate)
        self.graph._edges.append(e)


    def add_factor(self, id, position, information=np.eye(3)):
        """Add factor (vertex with identity edge) 

        GPS position factor represented as vertex containing position measurement
        connected to associated pose vertex with identity edge.

        Parameters
        ----------
        id : int
            Vertex ID
        position : np.array (3 x 1)
            GPS position measurement
        information : np.array (3 x 3)
            Information matrix
        
        """
        # "Fake" GPS node with R3 edge
        # Factor vertex
        p = PoseR3(position)
        v = Vertex(-id, p, fixed=True)
        self.graph._vertices.append(v)
        # Factor edge
        estimate = PoseR3(np.zeros(3))
        #e = EdgePosition([-id, id], information, estimate)
        e = EdgeOdometry([-id, id], information, estimate)
        self.graph._edges.append(e)

        # # Dangling factor
        # estimate = PoseR3(pose[1])
        # e = EdgePosition([id], information, estimate)
        # self.graph._edges.append(e)


    def get_positions(self):
        """Retrieve positions from graph vertices

        Returns
        -------
        positions : np.array (N x 3) 
            Array of positions

        """
        positions = []
        for v in self.graph._vertices:
            if v.id >= 0:
                positions.append(v.pose.position)
        return np.array(positions)


    def trim_window(self, n=1):
        """Maintain fixed window size

        Assumes graph has window_len + 1 nodes

        Parameters
        ----------
        window_len : int
            Window length

        """
        for i in range(n+1):
            # Remove first node (and its GPS node)
            del self.graph._vertices[0]

            # Remove associated edges
            del self.graph._edges[0]


    def optimize(self, max_iter=20, window=None, suppress_output=True):
        """Optimize the pose graph

        Parameters
        ----------
        window : tuple (start, end)
            Window of poses to optimize
        suppress_output : bool
            Suppress output from optimizer

        """
        # Link edges before calling optimize
        self.graph._link_edges()

        # Fix nodes outside window
        # if window is not None:
        #     for v in self.graph._vertices:
        #         if v.id < 0:
        #             v.fixed = True
        #         elif v.id < window[0] or v.id > window[1]:
        #             v.fixed = True
        #         else:
        #             v.fixed = False

        # Suppress output
        if suppress_output:
            with SuppressPrint():
                self.graph.optimize(max_iter=max_iter)
        else:
            self.graph.optimize(max_iter=max_iter)

        # Unfix nodes
        # if window is not None:
        #     for v in self.graph._vertices:
        #         if v.id < 0:
        #             v.fixed = True
        #         else:
        #             v.fixed = False


    def plot_trace(self, marker_size=5, edge_width=5):
        """Generate plot trace
        
        """
        # Link edges before plotting
        self.graph._link_edges()

        fixed = []
        free = []
        for v in self.graph._vertices:
            if v.id < 0:
                fixed.append(v.pose.position)
            else:
                free.append(v.pose.position)
        fixed = np.array(fixed)
        free = np.array(free)

        # # Draw fixed vertices as boxes
        # factors = go.Scatter3d(x=fixed[:,0], y=fixed[:,1], z=fixed[:,2], showlegend=False, 
        #     mode='markers', marker=dict(size=marker_size, color='orange', symbol='square'))

        # # Non-fixed vertices
        # nodes = go.Scatter3d(x=free[:,0], y=free[:,1], z=free[:,2], showlegend=False, 
        #     mode='markers', marker=dict(size=marker_size, color='blue', symbol='circle'))

        factors = go.Scatter(x=fixed[:,0], y=fixed[:,1], showlegend=False, 
            mode='markers', marker=dict(size=marker_size, color='orange', symbol='square'))

        # Non-fixed vertices
        nodes = go.Scatter(x=free[:,0], y=free[:,1], showlegend=False, 
            mode='markers', marker=dict(size=marker_size, color='blue', symbol='circle'))


        # Edges
        # edge_x = []
        # edge_y = []
        # edge_z = []
        # for edge in self.graph._edges:
        #     x0, y0, z0 = edge.vertices[0].pose[:3]
        #     x1, y1, z1 = edge.vertices[1].pose[:3]
        #     edge_x.append(x0)
        #     edge_x.append(x1)
        #     edge_x.append(None)
        #     edge_y.append(y0)
        #     edge_y.append(y1)
        #     edge_y.append(None)
        #     edge_z.append(z0)
        #     edge_z.append(z1)
        #     edge_z.append(None)
        # edges = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', showlegend=False, 
        #     line=dict(width=edge_width, color='orange'))

        edge_x = []
        edge_y = []
        for edge in self.graph._edges:
            x0, y0 = edge.vertices[0].pose[:2]
            x1, y1 = edge.vertices[1].pose[:2]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edges = go.Scatter(x=edge_x, y=edge_y, mode='lines', showlegend=False, 
            line=dict(width=edge_width, color='orange'))

        return [factors, nodes, edges]


    def test_statistic(self):
        """Compute test statistic for spoofing detection

        Test statistic is the sum of the residuals over GPS factors

        Returns
        -------
        q : float
            Test statistic 

        """
        # Link edges before computation
        self.graph._link_edges()

        # Compute test statistic
        q = 0
        for e in self.graph._edges:
            if e.vertex_ids[0] < 0:
                q += e.calc_chi2()
        return q
        

    def detect_loop_closures(self):
        """Search for loop closures

        """
        


    def trim_loop_closures(self):
        """
        """
    

    