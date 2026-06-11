import streamlit as st
import numpy as np
import plotly.graph_objects as go

def go_to(page: str):
    st.session_state.page = page
    st.rerun()


def rotation_matrix(angle_x, angle_y, angle_z):
    """Create a 3D rotation matrix from angles in degrees (ZYX convention).

    Args:
        angle_x: Rotation around X axis in degrees
        angle_y: Rotation around Y axis in degrees
        angle_z: Rotation around Z axis in degrees

    Returns:
        3x3 numpy rotation matrix
    """
    x, y, z = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    # Combined rotation (ZYX order)
    return Rz @ Ry @ Rx


def rotate_point(point, angle_x, angle_y, angle_z):
    """Rotate a 3D point around origin using Euler angles.

    Args:
        point: 3D point as tuple or array
        angle_x: Rotation around X axis in degrees
        angle_y: Rotation around Y axis in degrees
        angle_z: Rotation around Z axis in degrees

    Returns:
        Rotated point as numpy array
    """
    R = rotation_matrix(angle_x, angle_y, angle_z)
    return R @ np.array(point)


def make_magnet_mesh(cx, cy, cz, radius=3, half_height=5, n=30):
    """Returns (north_mesh, south_mesh) Mesh3d traces for a cylindrical magnet."""
    theta = np.linspace(0, 2 * np.pi, n)

    # Cylinder side vertices (two caps + side)
    # North half (top): z from 0 to +half_height
    # South half (bottom): z from -half_height to 0

    def cylinder_mesh(z_bot, z_top, color, name):
        # Build vertices for a closed cylinder
        x_ring = cx + radius * np.cos(theta)
        y_ring = cy + radius * np.sin(theta)

        # Vertices: bottom ring, top ring, bottom center, top center
        x_verts = np.concatenate([x_ring, x_ring, [cx], [cx]])
        y_verts = np.concatenate([y_ring, y_ring, [cy], [cy]])
        z_verts = np.concatenate(
            [
                np.full(n, cz + z_bot),
                np.full(n, cz + z_top),
                [cz + z_bot],
                [cz + z_top],
            ]
        )

        i_list, j_list, k_list = [], [], []
        bot_center = 2 * n
        top_center = 2 * n + 1

        for idx in range(n):
            nxt = (idx + 1) % n
            # Side quad -> 2 triangles
            i_list += [idx, idx]
            j_list += [nxt, nxt + n]
            k_list += [idx + n, idx]
            # Rearranged for correct winding:
            i_list += [idx, nxt]
            j_list += [nxt, nxt + n]
            k_list += [idx + n, idx + n]
            # Bottom cap
            i_list.append(bot_center)
            j_list.append(nxt)
            k_list.append(idx)
            # Top cap
            i_list.append(top_center)
            j_list.append(idx + n)
            k_list.append(nxt + n)

        return go.Mesh3d(
            x=x_verts,
            y=y_verts,
            z=z_verts,
            i=i_list,
            j=j_list,
            k=k_list,
            color=color,
            opacity=1.0,
            name=name,
            showlegend=False,
            flatshading=False,
        )

    north = cylinder_mesh(0, half_height, "red", "N pole")
    south = cylinder_mesh(-half_height, 0, "blue", "S pole")
    return north, south