"""Registration

Functions and utilities for point-cloud registration.

"""

import open3d as o3d


def initalize_source_and_target(source_data, target_data):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    source.estimate_normals()
    source.orient_normals_towards_camera_location()

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_data)
    target.estimate_normals()
    target.orient_normals_towards_camera_location()

    return source, target

def p2p_ICP(source, target, threshold, trans_init):
    start_time = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    eval_time = time.time() - start_time
    return reg_p2p, eval_time