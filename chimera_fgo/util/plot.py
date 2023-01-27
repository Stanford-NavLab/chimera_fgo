"""Plotting

This module contains functions for plotting the results of the FGO experiments.

"""

import numpy as np
import os
import plotly.graph_objects as go

from chimera_fgo.util.kitti import process_kitti_gt


def plot_trajectories(gt_enu, graph_positions, lidar_positions=None, spoofed_positions=None, detect_idx=None):
    """Plot FGO against ground-truth and spoofed trajectories.
    """
    ATTACK_START = 1000

    fgo_traj = go.Scatter(x=graph_positions[:,0], y=graph_positions[:,1], name='FGO')
    gt_traj = go.Scatter(x=gt_enu[:,0], y=gt_enu[:,1], name='Ground-truth', line=dict(color='black'))
    start = go.Scatter(x=[0], y=[0], name='Start', mode='markers', marker=dict(size=10, color='blue'), showlegend=True)
    plot_data = [fgo_traj, gt_traj, start]

    if lidar_positions is not None:
        lidar_traj = go.Scatter(x=lidar_positions[:,0], y=lidar_positions[:,1], name='Lidar', line=dict(color='green'))
        plot_data += [lidar_traj]

    if spoofed_positions is not None:
        spoof_traj = go.Scatter(x=spoofed_positions[:,0], y=spoofed_positions[:,1], 
            name='Spoofed', line=dict(color='red', dash='dash')) 
        spoof_start = go.Scatter(x=[spoofed_positions[ATTACK_START,0]], y=[spoofed_positions[ATTACK_START,1]], 
            name='Spoofing start', mode='markers', marker=dict(size=10, color='red'), showlegend=True)
        plot_data += [spoof_traj, spoof_start]

    if detect_idx is not None:
        detect = go.Scatter(x=[graph_positions[detect_idx,0]], y=[graph_positions[detect_idx,1]], 
            name='Detection', mode='markers', hovertext=str(detect_idx), marker=dict(size=10, color='green'), showlegend=True)
        plot_data += [detect]
    
    fig = go.Figure(data=plot_data)
    fig.update_layout(width=700, height=700, xaxis_title='East [m]', yaxis_title='North [m]')
    #fig.update_layout(legend=dict(x=0.05, y=0.98), font=dict(size=18))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    fig.update_xaxes(autorange=True)
    fig.show()


def plot_MC(kitti_seq, run_name):
    """Plot Monte Carlo runs
    """
    results_path = os.path.join(os.getcwd(), '..', 'results', kitti_seq, run_name)
    results_files = os.listdir(results_path)
    start_idx = 1550 if kitti_seq == '0028' else 0
    gtpath = os.path.join(os.getcwd(), '..', 'data', 'kitti', kitti_seq, 'oxts', 'data')
    gt_enu, gt_Rs, gt_attitudes = process_kitti_gt(gtpath, start_idx=start_idx)

    fig = go.Figure()
    for i, fname in enumerate(results_files):
        results = np.load(os.path.join(results_path, fname))
        graph_positions = results['positions']
        showlegend = True if i == 0 else False
        fig.add_trace(go.Scatter(x=graph_positions[:,0], y=graph_positions[:,1], name='FGO', showlegend=showlegend, line=dict(color='blue')))

    N = len(graph_positions)
    fig.add_trace(go.Scatter(x=gt_enu[:N,0], y=gt_enu[:N,1], name='Ground-truth', line=dict(color='black')))
    fig.update_layout(width=700, height=700, xaxis_title='East [m]', yaxis_title='North [m]')
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    return fig
