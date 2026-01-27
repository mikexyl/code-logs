#!/usr/bin/env python3
"""
Script to plot a g2o pose graph file.
Visualizes vertices (poses) and edges (constraints) in 2D and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parse_g2o_file(filename):
    """Parse a g2o file and extract vertices and edges."""
    vertices = {}
    edges = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'VERTEX_SE3:QUAT':
                # VERTEX_SE3:QUAT id x y z qx qy qz qw
                vertex_id = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                vertices[vertex_id] = (x, y, z)
                
            elif parts[0] == 'EDGE_SE3:QUAT':
                # EDGE_SE3:QUAT id1 id2 dx dy dz dqx dqy dqz dqw info_matrix...
                id1 = int(parts[1])
                id2 = int(parts[2])
                edges.append((id1, id2))
    
    return vertices, edges


def plot_pose_graph_2d(vertices, edges, ax=None):
    """Plot the pose graph in 2D (XY plane)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Group vertices by their ID prefix (assuming similar IDs belong to same robot/trajectory)
    vertex_groups = {}
    for vid, (x, y, z) in vertices.items():
        # Group by first few digits
        group_id = vid // 10000000000000000
        if group_id not in vertex_groups:
            vertex_groups[group_id] = []
        vertex_groups[group_id].append((vid, x, y, z))
    
    # Plot each trajectory with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(vertex_groups)))
    
    for idx, (group_id, group_vertices) in enumerate(sorted(vertex_groups.items())):
        # Sort vertices by ID to connect them in order
        group_vertices.sort(key=lambda v: v[0])
        xs = [v[1] for v in group_vertices]
        ys = [v[2] for v in group_vertices]
        
        # Plot trajectory
        ax.plot(xs, ys, 'o-', color=colors[idx], label=f'Robot {group_id}', 
                markersize=4, linewidth=1.5, alpha=0.7)
        
        # Mark start and end
        ax.plot(xs[0], ys[0], 'o', color=colors[idx], markersize=10, 
                markeredgecolor='black', markeredgewidth=2)
        ax.plot(xs[-1], ys[-1], 's', color=colors[idx], markersize=10, 
                markeredgecolor='black', markeredgewidth=2)
    
    # Plot edges (loop closures and inter-robot connections)
    for id1, id2 in edges:
        if id1 in vertices and id2 in vertices:
            group1 = id1 // 10000000000000000
            group2 = id2 // 10000000000000000
            
            # Only plot edges that are not sequential within the same robot
            if group1 != group2 or abs(id1 - id2) > 1:
                x1, y1, _ = vertices[id1]
                x2, y2, _ = vertices[id2]
                ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Pose Graph - Top View (XY)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return ax


def plot_pose_graph_3d(vertices, edges):
    """Plot the pose graph in 3D."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Group vertices by their ID prefix
    vertex_groups = {}
    for vid, (x, y, z) in vertices.items():
        group_id = vid // 10000000000000000
        if group_id not in vertex_groups:
            vertex_groups[group_id] = []
        vertex_groups[group_id].append((vid, x, y, z))
    
    # Plot each trajectory with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(vertex_groups)))
    
    for idx, (group_id, group_vertices) in enumerate(sorted(vertex_groups.items())):
        group_vertices.sort(key=lambda v: v[0])
        xs = [v[1] for v in group_vertices]
        ys = [v[2] for v in group_vertices]
        zs = [v[3] for v in group_vertices]
        
        # Plot trajectory
        ax.plot(xs, ys, zs, 'o-', color=colors[idx], label=f'Robot {group_id}', 
                markersize=3, linewidth=1.5, alpha=0.7)
        
        # Mark start and end
        ax.scatter(xs[0], ys[0], zs[0], color=colors[idx], s=100, 
                   edgecolors='black', linewidths=2, marker='o')
        ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[idx], s=100, 
                   edgecolors='black', linewidths=2, marker='s')
    
    # Plot edges (loop closures)
    for id1, id2 in edges:
        if id1 in vertices and id2 in vertices:
            group1 = id1 // 10000000000000000
            group2 = id2 // 10000000000000000
            
            if group1 != group2 or abs(id1 - id2) > 1:
                x1, y1, z1 = vertices[id1]
                x2, y2, z2 = vertices[id2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Pose Graph - 3D View', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    return ax


def plot_three_planes(vertices, edges):
    """Plot the pose graph in three 2D projections: XY, XZ, and YZ planes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group vertices by their ID prefix
    vertex_groups = {}
    for vid, (x, y, z) in vertices.items():
        group_id = vid // 10000000000000000
        if group_id not in vertex_groups:
            vertex_groups[group_id] = []
        vertex_groups[group_id].append((vid, x, y, z))
    
    # Plot each trajectory with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(vertex_groups)))
    
    # Define the three planes: (axis1_idx, axis2_idx, xlabel, ylabel, title)
    planes = [
        (0, 1, 'X (m)', 'Y (m)', 'XY Plane (Top View)'),
        (0, 2, 'X (m)', 'Z (m)', 'XZ Plane (Side View)'),
        (1, 2, 'Y (m)', 'Z (m)', 'YZ Plane (Front View)')
    ]
    
    for ax, (axis1, axis2, xlabel, ylabel, title) in zip(axes, planes):
        # Plot trajectories
        for idx, (group_id, group_vertices) in enumerate(sorted(vertex_groups.items())):
            group_vertices.sort(key=lambda v: v[0])
            coords = [(v[1], v[2], v[3]) for v in group_vertices]
            
            # Extract the appropriate axes
            axis1_vals = [c[axis1] for c in coords]
            axis2_vals = [c[axis2] for c in coords]
            
            # Plot trajectory
            ax.plot(axis1_vals, axis2_vals, 'o-', color=colors[idx], 
                   label=f'Robot {group_id}', markersize=4, linewidth=1.5, alpha=0.7)
            
            # Mark start and end
            ax.plot(axis1_vals[0], axis2_vals[0], 'o', color=colors[idx], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.plot(axis1_vals[-1], axis2_vals[-1], 's', color=colors[idx], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Plot edges (loop closures and inter-robot connections)
        for id1, id2 in edges:
            if id1 in vertices and id2 in vertices:
                group1 = id1 // 10000000000000000
                group2 = id2 // 10000000000000000
                
                # Only plot edges that are not sequential within the same robot
                if group1 != group2 or abs(id1 - id2) > 1:
                    x1, y1, z1 = vertices[id1]
                    x2, y2, z2 = vertices[id2]
                    coords1 = [x1, y1, z1]
                    coords2 = [x2, y2, z2]
                    ax.plot([coords1[axis1], coords2[axis1]], 
                           [coords1[axis2], coords2[axis2]], 
                           'r-', alpha=0.8, linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot g2o pose graph file')
    parser.add_argument('g2o_file', type=str, help='Path to g2o file')
    parser.add_argument('--only-2d', action='store_true', help='Only show 2D plot (XY plane)')
    parser.add_argument('--only-3d', action='store_true', help='Only show 3D plot')
    parser.add_argument('--three-planes', action='store_true', help='Show XY, XZ, and YZ planes')
    parser.add_argument('--save', type=str, help='Save plot to file instead of showing')
    
    args = parser.parse_args()
    
    # Parse the g2o file
    print(f"Parsing {args.g2o_file}...")
    vertices, edges = parse_g2o_file(args.g2o_file)
    print(f"Found {len(vertices)} vertices and {len(edges)} edges")
    
    # Determine number of robots
    num_robots = len(set(vid // 10000000000000000 for vid in vertices.keys()))
    print(f"Number of trajectories/robots: {num_robots}")
    
    # Create plots
    if args.only_3d:
        plot_pose_graph_3d(vertices, edges)
    elif args.only_2d:
        plot_pose_graph_2d(vertices, edges)
    elif args.three_planes:
        plot_three_planes(vertices, edges)
    else:
        # Default: show three planes
        plot_three_planes(vertices, edges)
    
    # Save or show
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
