import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import numpy as np
import open3d as o3d

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def SparseCLIP_visualize(sample):
    lidar = sample['raw_lidar'][-1] if sample['raw_lidar'].ndim == 3 else sample['raw_lidar']
    if isinstance(lidar, torch.Tensor):
        lidar = lidar.cpu().numpy()
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lidar[:,0], lidar[:,1], lidar[:,2], s=0.2, c='blue', alpha=0.5)

    for b in sample['all_bboxes']:
        draw_bbox(ax, b['bbox'], color='red')
    
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-10, 90)
    plt.show()

def SparseCLIP_visualize_open3d(sample, point_size=1):
    # lidar
    lidar = sample['raw_lidar'][-1] if sample['raw_lidar'].ndim == 3 else sample['raw_lidar']
    if isinstance(lidar, torch.Tensor):
        lidar = lidar.cpu().numpy()
    lidar = lidar[:, :3].astype(np.float64)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray points

    # bbox
    bbox_lines = []
    for b in sample['all_bboxes']:
        bbox_lines.append(draw_bbox_o3d(b['bbox']))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for box in bbox_lines:
        vis.add_geometry(box)

    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([pcd, *bbox_lines])

def draw_bbox(ax, bbox, color='red'):
    """
    Draw 3D bbox lines from [x, y, z, dx, dy, dz, yaw]
    :param ax: matplotlib 3D axis
    :param bbox: list or array of 7 elements [x, y, z, dx, dy, dz, yaw]
    """
    x, y, z, w, l, h, yaw = bbox
    # rotation matrix around z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    
    # 8 corners relative to center
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = R @ corners
    corners += np.array([[x],[y],[z]])
    corners = corners.T  # (8,3)
    
    # draw edges
    # top face
    top = [0,1,2,3,0]
    for i in range(4):
        ax.plot(corners[top[i:i+2],0], corners[top[i:i+2],1], corners[top[i:i+2],2], c=color)
    # bottom face
    bottom = [4,5,6,7,4]
    for i in range(4):
        ax.plot(corners[bottom[i:i+2],0], corners[bottom[i:i+2],1], corners[bottom[i:i+2],2], c=color)
    # vertical edges
    for i in range(4):
        ax.plot([corners[i,0], corners[i+4,0]],
                [corners[i,1], corners[i+4,1]],
                [corners[i,2], corners[i+4,2]], c=color)
    ax.scatter([x], [y], [z], color='green', s=20)

def draw_bbox_o3d(bbox):
    """
    Create an Open3D LineSet from [x, y, z, dx, dy, dz, yaw]
    """
    x, y, z, w, l, h, yaw = bbox
    # rotation matrix around z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    
    # 8 corners relative to center
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = R @ corners
    corners += np.array([[x],[y],[z]])
    corners = corners.T  # (8,3)

    # define lines
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # top
        [4,5],[5,6],[6,7],[7,4],  # bottom
        [0,4],[1,5],[2,6],[3,7]   # vertical
    ]
    colors = [[1,0,0] for _ in lines]  # red lines
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
