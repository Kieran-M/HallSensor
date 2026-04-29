import streamlit as st
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="3D Magnetic Field Calculator")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Magnet Settings")

    shape = st.selectbox("Magnet Shape", ["Cylinder", "Cuboid"])

    magnet_type = st.selectbox(
        "Magnet Type", ["N42", "N35", "N52", "Ceramic", "Custom"]
    )

    remanence_presets = {
        "N35": 11700,
        "N42": 13000,
        "N52": 14500,
        "Ceramic": 3800,
        "Custom": 1000,
    }

    default_rem = remanence_presets.get(magnet_type, 1000)
    disabled_rem = magnet_type != "Custom"
    remanence_g = st.number_input(
        "Remanence (Gauss)", value=float(default_rem), disabled=disabled_rem
    )

    height = st.number_input(
        "Height (mm)", value=5.0, min_value=0.1, step=0.5
    )

    if shape == "Cylinder":
        diameter = st.number_input(
            "Diameter (mm)", value=5.0, min_value=0.1, step=0.5
        )
    else:
        width = st.number_input("Width (mm)", value=10.0)
        length = st.number_input("Length (mm)", value=10.0)

    z_air_gap = st.number_input(
        "Z, Air Gap (mm)", value=2.0, min_value=0.0, step=0.1
    )

    st.markdown("---")
    st.header("Animation Settings")

    motion_type = st.selectbox(
        "Motion Path",
        [
            "Linear X-Sweep",
            "Linear Y-Sweep",
            "Linear Z-Sweep",
            "Circular XY",
            "Hinge (Door/Lid)",
            "Custom Path",
        ],
    )

    num_frames = st.slider(
        "Number of Frames",
        min_value=10,
        max_value=200,
        value=60,
        step=10,
    )

    if motion_type in [
        "Linear X-Sweep",
        "Linear Y-Sweep",
        "Linear Z-Sweep",
    ]:
        sweep_range = st.number_input(
            "Sweep Range (mm, ±)",
            value=15.0,
            min_value=1.0,
            step=1.0,
        )

    elif motion_type == "Circular XY":
        orbit_radius = st.number_input(
            "Orbit Radius (mm)",
            value=10.0,
            min_value=1.0,
            step=1.0,
        )
        orbit_z_offset = st.number_input(
            "Z Offset from sensor (mm)",
            value=5.0,
            min_value=0.0,
            step=0.5,
        )

    elif motion_type == "Hinge (Door/Lid)":
        st.caption(
            "Simulates a magnet on a door/lid that rotates around "
            "a hinge axis. The sensor sits near the hinge."
        )
        hinge_radius = st.number_input(
            "Hinge Arm Length (mm)",
            value=15.0,
            min_value=1.0,
            step=1.0,
            help="Distance from hinge pivot to the magnet center.",
        )
        hinge_angle_open = st.slider(
            "Open Angle (°)",
            min_value=0,
            max_value=180,
            value=90,
            step=5,
            help="Maximum opening angle of the door/lid.",
        )
        hinge_angle_close = st.slider(
            "Closed Angle (°)",
            min_value=0,
            max_value=180,
            value=0,
            step=5,
            help="Angle when the door/lid is fully closed.",
        )
        hinge_plane = st.selectbox(
            "Hinge Rotation Plane",
            ["XZ (side hinge)", "YZ (top hinge)", "XY (flat spin)"],
        )
        hinge_sensor_offset = st.number_input(
            "Sensor Offset from Pivot (mm)",
            value=2.0,
            min_value=0.0,
            step=0.5,
            help=(
                "How far the sensor is from the hinge pivot point "
                "along the perpendicular axis."
            ),
        )
        hinge_bounce = st.checkbox(
            "Bounce (close → open → close)",
            value=True,
            help="Animate a full open-and-close cycle.",
        )

    elif motion_type == "Custom Path":
        st.caption(
            "Define waypoints (one per line: x,y,z in mm). "
            "The magnet interpolates between them."
        )
        custom_waypoints_str = st.text_area(
            "Waypoints",
            value="0,0,5\n10,0,5\n10,10,5\n0,10,5\n0,0,5",
        )


# --- Calculation Logic ---
def create_magnet(shape_type, h, d_or_dim, rem_gauss):
    polarization_mt = rem_gauss / 10.0
    if shape_type == "Cylinder":
        magnet = magpy.magnet.Cylinder(
            polarization=(0, 0, polarization_mt),
            dimension=(d_or_dim, h),
            position=(0, 0, 0),
        )
    else:
        w, l = d_or_dim
        magnet = magpy.magnet.Cuboid(
            polarization=(0, 0, polarization_mt),
            dimension=(l, w, h),
            position=(0, 0, 0),
        )
    return magnet


def get_magnet_and_sensor(shape_type, h, d_or_dim, rem_gauss, gap):
    magnet = create_magnet(shape_type, h, d_or_dim, rem_gauss)
    sensor_pos_z = (h / 2) + gap
    sensor = magpy.Sensor(position=(0, 0, sensor_pos_z))
    return magnet, sensor


def calculate_field(magnet, sensor):
    b_vec = magpy.getB(magnet, sensor)
    return abs(b_vec[2] * 10.0)


def generate_curve(magnet, h):
    gaps = np.linspace(0, 15, 50)
    zs = (h / 2) + gaps
    path = np.column_stack(
        [np.zeros_like(zs), np.zeros_like(zs), zs]
    )
    res = []
    for p in path:
        s = magpy.Sensor(position=p)
        res.append(magpy.getB(magnet, s))
    b_fields = np.array(res)
    bz_curve_gauss = np.abs(b_fields[:, 2]) * 10.0
    return gaps, bz_curve_gauss


def generate_animation_path(motion, n_frames, **kwargs):
    if motion == "Linear X-Sweep":
        r = kwargs["sweep_range"]
        xs = np.linspace(-r, r, n_frames)
        ys = np.zeros(n_frames)
        zs = np.full(n_frames, kwargs["z_offset"])
        return np.column_stack([xs, ys, zs])

    elif motion == "Linear Y-Sweep":
        r = kwargs["sweep_range"]
        xs = np.zeros(n_frames)
        ys = np.linspace(-r, r, n_frames)
        zs = np.full(n_frames, kwargs["z_offset"])
        return np.column_stack([xs, ys, zs])

    elif motion == "Linear Z-Sweep":
        r = kwargs["sweep_range"]
        xs = np.zeros(n_frames)
        ys = np.zeros(n_frames)
        zs = np.linspace(0.5, r, n_frames)
        return np.column_stack([xs, ys, zs])

    elif motion == "Circular XY":
        radius = kwargs["orbit_radius"]
        z_off = kwargs["z_offset"]
        angles = np.linspace(
            0, 2 * np.pi, n_frames, endpoint=False
        )
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        zs = np.full(n_frames, z_off)
        return np.column_stack([xs, ys, zs])

    elif motion == "Hinge (Door/Lid)":
        radius = kwargs["hinge_radius"]
        a_open = np.radians(kwargs["angle_open"])
        a_close = np.radians(kwargs["angle_close"])
        plane = kwargs["plane"]
        bounce = kwargs["bounce"]

        if bounce:
            half = n_frames // 2
            angles_out = np.linspace(a_close, a_open, half)
            angles_back = np.linspace(
                a_open, a_close, n_frames - half
            )
            angles = np.concatenate([angles_out, angles_back])
        else:
            angles = np.linspace(a_close, a_open, n_frames)

        if plane == "XZ (side hinge)":
            xs = radius * np.sin(angles)
            ys = np.zeros(n_frames)
            zs = radius * np.cos(angles)
        elif plane == "YZ (top hinge)":
            xs = np.zeros(n_frames)
            ys = radius * np.sin(angles)
            zs = radius * np.cos(angles)
        else:  # XY flat spin
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            zs = np.full(n_frames, kwargs.get("sensor_offset", 2.0))

        return np.column_stack([xs, ys, zs])

    elif motion == "Custom Path":
        waypoints = kwargs["waypoints"]
        if len(waypoints) < 2:
            return np.tile(waypoints[0], (n_frames, 1))
        total_seg = np.sqrt(
            np.sum(np.diff(waypoints, axis=0) ** 2, axis=1)
        )
        cumulative = np.concatenate([[0], np.cumsum(total_seg)])
        total_length = cumulative[-1]
        if total_length == 0:
            return np.tile(waypoints[0], (n_frames, 1))
        t_uniform = np.linspace(0, total_length, n_frames)
        path = np.zeros((n_frames, 3))
        for i in range(3):
            path[:, i] = np.interp(
                t_uniform, cumulative, waypoints[:, i]
            )
        return path

    return np.zeros((n_frames, 3))


def compute_animation_fields(magnet, path_positions):
    sensor_pos = (0, 0, 0)
    fields = []
    for pos in path_positions:
        magnet.position = pos
        b = magpy.getB(magnet, sensor_pos)
        fields.append(b * 10.0)
    magnet.position = (0, 0, 0)
    return np.array(fields)


def parse_custom_waypoints(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    pts = []
    for line in lines:
        parts = line.split(",")
        if len(parts) == 3:
            pts.append([float(p) for p in parts])
    return np.array(pts) if pts else np.array([[0, 0, 5]])


# --- Execute Static Logic ---
dims = diameter if shape == "Cylinder" else (width, length)
magnet_obj, sensor_obj = get_magnet_and_sensor(
    shape, height, dims, remanence_g, z_air_gap
)
result_gauss = calculate_field(magnet_obj, sensor_obj)
curve_gaps, curve_b = generate_curve(magnet_obj, height)

# --- Sidebar Result ---
with st.sidebar:
    st.markdown("### Result (Bz)")
    st.metric(
        label="Magnetic Flux Density", value=f"{result_gauss:.1f} G"
    )
    df = pd.DataFrame({"AirGap_mm": curve_gaps, "Bz_Gauss": curve_b})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv, "magnet_data.csv", "text/csv")

# --- Tabs ---
tab_static, tab_anim = st.tabs(
    ["Static Analysis", "Animation"]
)

# --- Static Tab ---
with tab_static:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Device")
        st.caption("Left click to rotate. Scroll to zoom.")

        fig_3d = magpy.show(
            magnet_obj,
            sensor_obj,
            backend="plotly",
            return_fig=True,
            style_path_frames=10,
        )
        fig_3d.update_layout(
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Field Strength vs Gap")
        fig_2d = go.Figure()
        fig_2d.add_trace(
            go.Scatter(
                x=curve_gaps, y=curve_b, mode="lines", name="Bz vs Gap"
            )
        )
        fig_2d.add_trace(
            go.Scatter(
                x=[z_air_gap],
                y=[result_gauss],
                mode="markers",
                name="Current Pos",
                marker=dict(size=10, color="red"),
            )
        )
        fig_2d.update_layout(
            xaxis_title="Air Gap (mm)",
            yaxis_title="Magnetic Field (Gauss)",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_2d, use_container_width=True)

# --- Animation Tab ---
with tab_anim:
    st.subheader("Animated Magnet Motion")

    anim_magnet = create_magnet(shape, height, dims, remanence_g)
    z_offset_default = (height / 2) + z_air_gap

    path_kwargs = {}
    if motion_type in ["Linear X-Sweep", "Linear Y-Sweep"]:
        path_kwargs["sweep_range"] = sweep_range
        path_kwargs["z_offset"] = z_offset_default
    elif motion_type == "Linear Z-Sweep":
        path_kwargs["sweep_range"] = sweep_range
    elif motion_type == "Circular XY":
        path_kwargs["orbit_radius"] = orbit_radius
        path_kwargs["z_offset"] = orbit_z_offset
    elif motion_type == "Hinge (Door/Lid)":
        path_kwargs["hinge_radius"] = hinge_radius
        path_kwargs["angle_open"] = hinge_angle_open
        path_kwargs["angle_close"] = hinge_angle_close
        path_kwargs["plane"] = hinge_plane
        path_kwargs["bounce"] = hinge_bounce
        path_kwargs["sensor_offset"] = hinge_sensor_offset
    elif motion_type == "Custom Path":
        path_kwargs["waypoints"] = parse_custom_waypoints(
            custom_waypoints_str
        )

    anim_path = generate_animation_path(
        motion_type, num_frames, **path_kwargs
    )

    with st.spinner("Computing fields along path..."):
        anim_fields = compute_animation_fields(anim_magnet, anim_path)

    bx = anim_fields[:, 0]
    by = anim_fields[:, 1]
    bz = anim_fields[:, 2]
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    frame_idx = np.arange(num_frames)

    # Compute angle axis for hinge mode
    if motion_type == "Hinge (Door/Lid)":
        if hinge_bounce:
            half = num_frames // 2
            angles_deg_out = np.linspace(
                hinge_angle_close, hinge_angle_open, half
            )
            angles_deg_back = np.linspace(
                hinge_angle_open,
                hinge_angle_close,
                num_frames - half,
            )
            angles_deg = np.concatenate(
                [angles_deg_out, angles_deg_back]
            )
        else:
            angles_deg = np.linspace(
                hinge_angle_close, hinge_angle_open, num_frames
            )

    # Data export
    df_anim = pd.DataFrame(
        {
            "Frame": frame_idx,
            "Magnet_X": anim_path[:, 0],
            "Magnet_Y": anim_path[:, 1],
            "Magnet_Z": anim_path[:, 2],
            "Bx_Gauss": bx,
            "By_Gauss": by,
            "Bz_Gauss": bz,
            "B_Total_Gauss": b_mag,
        }
    )
    if motion_type == "Hinge (Door/Lid)":
        df_anim.insert(1, "Angle_deg", angles_deg)

    csv_anim = df_anim.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Animation Data (CSV)",
        csv_anim,
        "animation_data.csv",
        "text/csv",
    )

    col_3d, col_2d = st.columns([1, 1])

    with col_3d:
        st.caption("3D magnet path and sensor position")

        fig_path = go.Figure()

        # Trace 0: Path line
        fig_path.add_trace(
            go.Scatter3d(
                x=anim_path[:, 0],
                y=anim_path[:, 1],
                z=anim_path[:, 2],
                mode="lines",
                line=dict(color="royalblue", width=3),
                name="Magnet Path",
            )
        )

        # Trace 1: Sensor
        fig_path.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers",
                marker=dict(size=6, color="green", symbol="diamond"),
                name="Sensor (origin)",
            )
        )

        # Trace 2: Hinge arm or placeholder
        if motion_type == "Hinge (Door/Lid)":
            fig_path.add_trace(
                go.Scatter3d(
                    x=[0, anim_path[0, 0]],
                    y=[0, anim_path[0, 1]],
                    z=[0, anim_path[0, 2]],
                    mode="lines",
                    line=dict(color="orange", width=2),
                    name="Hinge Arm",
                    showlegend=False,
                )
            )
        else:
            fig_path.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="none",
                    showlegend=False,
                )
            )

        # Trace 3: Magnet marker (animated)
        fig_path.add_trace(
            go.Scatter3d(
                x=[anim_path[0, 0]],
                y=[anim_path[0, 1]],
                z=[anim_path[0, 2]],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Magnet",
            )
        )

        # Build frames — only update traces 2 and 3
        frames_3d = []
        slider_steps = []
        for i in range(num_frames):
            frame_data = []

            # Update trace 2: hinge arm or placeholder
            if motion_type == "Hinge (Door/Lid)":
                frame_data.append(
                    go.Scatter3d(
                        x=[0, anim_path[i, 0]],
                        y=[0, anim_path[i, 1]],
                        z=[0, anim_path[i, 2]],
                        mode="lines",
                        line=dict(color="orange", width=2),
                        showlegend=False,
                    )
                )
            else:
                frame_data.append(
                    go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode="none",
                        showlegend=False,
                    )
                )

            # Update trace 3: magnet position
            frame_data.append(
                go.Scatter3d(
                    x=[anim_path[i, 0]],
                    y=[anim_path[i, 1]],
                    z=[anim_path[i, 2]],
                    mode="markers",
                    marker=dict(size=8, color="red"),
                    name="Magnet",
                )
            )

            frames_3d.append(
                go.Frame(
                    data=frame_data,
                    traces=[2, 3],
                    name=str(i),
                )
            )

            if motion_type == "Hinge (Door/Lid)":
                label = f"{angles_deg[i]:.0f}°"
            else:
                label = str(i)
            slider_steps.append(
                dict(
                    args=[
                        [str(i)],
                        dict(
                            frame=dict(duration=50, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                    label=label,
                    method="animate",
                )
            )

        fig_path.frames = frames_3d

        all_coords = np.concatenate([anim_path, [[0, 0, 0]]])
        center = all_coords.mean(axis=0)
        max_range = np.abs(all_coords - center).max()
        eye_dist = 1.5


        fig_path.update_layout(
            uirevision="constant",
            scene=dict(
                camera=dict(
                    eye=dict(
                        x=eye_dist * 0.8,
                        y=eye_dist * 1.0,
                        z=eye_dist * 0.6,
                    ),
                ),
                aspectmode="auto",
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0, x=0.5, xanchor="center",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=slider_steps,
                    currentvalue=dict(
                        prefix=(
                            "Angle: "
                            if motion_type == "Hinge (Door/Lid)"
                            else "Frame: "
                        ),
                        visible=True,
                    ),
                    pad=dict(t=50),
                )
            ],
        )

        st.plotly_chart(fig_path, use_container_width=True)

    with col_2d:
        if motion_type == "Hinge (Door/Lid)":
            st.caption("Field components at sensor vs. hinge angle")
            x_axis_data = angles_deg
            x_axis_label = "Hinge Angle (°)"
        else:
            st.caption("Field components at sensor vs. frame")
            x_axis_data = frame_idx
            x_axis_label = "Frame"

        fig_field = go.Figure()

        fig_field.add_trace(
            go.Scatter(
                x=x_axis_data,
                y=bx,
                mode="lines",
                name="Bx",
                line=dict(color="red"),
            )
        )
        fig_field.add_trace(
            go.Scatter(
                x=x_axis_data,
                y=by,
                mode="lines",
                name="By",
                line=dict(color="green"),
            )
        )
        fig_field.add_trace(
            go.Scatter(
                x=x_axis_data,
                y=bz,
                mode="lines",
                name="Bz",
                line=dict(color="blue"),
            )
        )
        fig_field.add_trace(
            go.Scatter(
                x=x_axis_data,
                y=b_mag,
                mode="lines",
                name="|B|",
                line=dict(color="black", dash="dash"),
            )
        )

        fig_field.add_trace(
            go.Scatter(
                x=[x_axis_data[0]],
                y=[b_mag[0]],
                mode="markers",
                marker=dict(size=10, color="orange"),
                name="Current",
                showlegend=False,
            )
        )

        field_frames = []
        for i in range(num_frames):
            field_frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=x_axis_data,
                            y=bx,
                            mode="lines",
                            name="Bx",
                            line=dict(color="red"),
                        ),
                        go.Scatter(
                            x=x_axis_data,
                            y=by,
                            mode="lines",
                            name="By",
                            line=dict(color="green"),
                        ),
                        go.Scatter(
                            x=x_axis_data,
                            y=bz,
                            mode="lines",
                            name="Bz",
                            line=dict(color="blue"),
                        ),
                        go.Scatter(
                            x=x_axis_data,
                            y=b_mag,
                            mode="lines",
                            name="|B|",
                            line=dict(
                                color="yellow", dash="dash"
                            ),
                        ),
                        go.Scatter(
                            x=[x_axis_data[i]],
                            y=[b_mag[i]],
                            mode="markers",
                            marker=dict(
                                size=12, color="orange"
                            ),
                            name="Current",
                            showlegend=False,
                        ),
                    ],
                    name=str(i),
                )
            )

        fig_field.frames = field_frames

        fig_field.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title="Field (Gauss)",
            height=500,
            margin=dict(l=20, r=20, t=30, b=80),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=50, redraw=True
                                    ),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(
                                        duration=0, redraw=False
                                    ),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=slider_steps,
                    currentvalue=dict(
                        prefix=(
                            "Angle: "
                            if motion_type == "Hinge (Door/Lid)"
                            else "Frame: "
                        ),
                        visible=True,
                    ),
                    pad=dict(t=50),
                )
            ],
        )

        st.plotly_chart(fig_field, use_container_width=True)

    # --- Summary ---
    st.subheader("Animation Summary")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Max |B|", f"{b_mag.max():.1f} G")
    with summary_cols[1]:
        st.metric("Min |B|", f"{b_mag.min():.1f} G")
    with summary_cols[2]:
        st.metric("Max Bz", f"{np.max(np.abs(bz)):.1f} G")
    with summary_cols[3]:
        path_len = np.sum(
            np.sqrt(
                np.sum(np.diff(anim_path, axis=0) ** 2, axis=1)
            )
        )
        st.metric("Path Length", f"{path_len:.1f} mm")

    with st.expander("View all animation data"):
        st.dataframe(df_anim, use_container_width=True)


# --- Part Matching ---
st.markdown("---")
st.subheader("Find Matching Parts")
c1, c2, c3 = st.columns(3)
with c1:
    dev_type = st.selectbox(
        "Device Type", ["Omnipolar", "Unipolar", "Bipolar"]
    )
with c2:
    out_type = st.selectbox(
        "Output Type", ["Open Drain", "Push-Pull"]
    )
with c3:
    volt_type = st.selectbox(
        "Operating Voltage", ["1.6 to 5.5", "3.0 to 24"]
    )

parts_db = [
    {
        "Part Number": "AH1921",
        "Type": "Omnipolar",
        "Out": "Open Drain",
        "Bop(Min)": 30,
        "Bop(Max)": 90,
    },
    {
        "Part Number": "AH180",
        "Type": "Omnipolar",
        "Out": "Push-Pull",
        "Bop(Min)": 40,
        "Bop(Max)": 110,
    },
    {
        "Part Number": "AH337",
        "Type": "Unipolar",
        "Out": "Open Drain",
        "Bop(Min)": 90,
        "Bop(Max)": 140,
    },
]
df_parts = pd.DataFrame(parts_db)
filtered_df = df_parts[
    (df_parts["Type"] == dev_type) & (df_parts["Out"] == out_type)
]
st.dataframe(filtered_df, use_container_width=True)

if not filtered_df.empty:
    req = filtered_df.iloc[0]["Bop(Max)"]
    if result_gauss > req:
        st.success(
            f"Success! {result_gauss:.1f}G > {req}G trigger point."
        )
    else:
        st.error(
            f"Too Weak. {result_gauss:.1f}G < {req}G trigger point."
        )