import streamlit as st
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

FRAMERATE = 60
FRAME_DURATION = int(1000 / FRAMERATE)

DISCLAIMER = (
    "Magnetic calculations are performed via Magpylib (BSD 2-Clause, "
    "© 2019-2025). Calculation results are provided for reference only. "
    "Please contact us if you have more advanced simulation requirement."
)


# --- Calculation Logic ---
def create_magnet(shape, h, d_or_dim, rem_gauss):
    polarization_mt = rem_gauss / 10.0
    shape_type = SHAPE_TYPE_MAP[shape]

    if shape_type == "Cylinder":
        radial = shape == "Cylinder (Radially Magnetized)"
        polarization = (polarization_mt, 0, 0) if radial else (0, 0, polarization_mt)
        return magpy.magnet.Cylinder(
            polarization=polarization,
            dimension=(d_or_dim, h),
            position=(0, 0, 0),
            style_magnetization_color_mode="bicolor",
            style_magnetization_color_north="r",
            style_magnetization_color_south="b",
            style_magnetization_color_transition=0,
        )
    elif shape_type == "Sphere":
        return magpy.magnet.Sphere(
            polarization=(0, 0, polarization_mt),
            diameter=h,
            position=(0, 0, 0),
            style_magnetization_color_mode="bicolor",
            style_magnetization_color_north="r",
            style_magnetization_color_south="b",
            style_magnetization_color_transition=0,
        )
    elif shape_type == "Ring":
        inner_r = (d_or_dim * 0.3) / 2
        outer_r = d_or_dim / 2
        return magpy.magnet.CylinderSegment(
            magnetization=(polarization_mt * 1e3, 0, 0),
            dimension=(inner_r, outer_r, h, 0, 360),
            position=(0, 0, 0),
            style_magnetization_color_mode="bicolor",
            style_magnetization_color_north="r",
            style_magnetization_color_south="b",
            style_magnetization_color_transition=0,
        )
    else:
        w, l = d_or_dim
        return magpy.magnet.Cuboid(
            polarization=(0, 0, polarization_mt),
            dimension=(l, w, h),
            position=(0, 0, 0),
            style_magnetization_color_mode="bicolor",
            style_magnetization_color_north="r",
            style_magnetization_color_south="b",
            style_magnetization_color_transition=0,
        )


def make_magnet_traces(
    shape, h, d_or_dim, rem_gauss, cx, cy, cz, angle_x=0, angle_y=0, angle_z=0
):
    from scipy.spatial.transform import Rotation as R

    magnet = create_magnet(shape, h, d_or_dim, rem_gauss)
    magnet.position = (cx, cy, cz)
    magnet.orientation = R.from_euler("xyz", [0, 0, 0], degrees=True)

    if angle_x != 0 or angle_y != 0 or angle_z != 0:
        magnet.rotate_from_euler([angle_x, angle_y, angle_z], "xyz", degrees=True)

    fig = magnet.show(backend="plotly", return_fig=True)

    traces = []
    for trace in fig.data:
        trace.update(showlegend=False)
        traces.append(trace)

    return traces


def get_magnet_and_sensor(shape, h, d_or_dim, rem_gauss, gap):
    magnet = create_magnet(shape, h, d_or_dim, rem_gauss)
    sensor_pos_z = (h / 2) + gap
    sensor = magpy.Sensor(position=(0, 0, sensor_pos_z))
    return magnet, sensor


def calculate_field(magnet, sensor):
    b_vec = magpy.getB(magnet, sensor)
    return abs(b_vec[2] * 10.0)


def generate_curve(shape, h, d_or_dim, rem_gauss):
    magnet = create_magnet(shape, h, d_or_dim, rem_gauss)
    gaps = np.linspace(0, 15, 50)
    zs = (h / 2) + gaps
    path = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    res = []
    for p in path:
        s = magpy.Sensor(position=p)
        res.append(magpy.getB(magnet, s))
    b_fields = np.array(res)
    return gaps, np.abs(b_fields[:, 2]) * 10.0


def generate_animation_path(motion, **kwargs):
    if motion == "Position-Based":
        start_pos = np.array(kwargs["start_pos"])
        end_pos = np.array(kwargs["end_pos"])
        return np.array([
            start_pos + t * (end_pos - start_pos)
            for t in np.linspace(0, 1, FRAMERATE)
        ])
    elif motion == "Hinge (Door/Lid)":
        radius = kwargs["hinge_radius"]
        a_open = np.radians(kwargs["angle_open"])
        a_close = np.radians(kwargs["angle_close"])
        plane = kwargs["plane"]
        bounce = kwargs["bounce"]
        hinge_origin = np.array(kwargs.get("hinge_origin", (0, 0, 0)))

        if bounce:
            half = FRAMERATE // 2
            angles = np.concatenate([
                np.linspace(a_close, a_open, half),
                np.linspace(a_open, a_close, FRAMERATE - half),
            ])
        else:
            angles = np.linspace(a_close, a_open, FRAMERATE)

        if plane == "XZ (side hinge)":
            local_path = np.column_stack([
                radius * np.sin(angles),
                np.zeros(FRAMERATE),
                radius * np.cos(angles),
            ])
        elif plane == "YZ (top hinge)":
            local_path = np.column_stack([
                np.zeros(FRAMERATE),
                radius * np.sin(angles),
                radius * np.cos(angles),
            ])
        else:
            local_path = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(FRAMERATE, kwargs.get("sensor_offset", 2.0)),
            ])

        return local_path + hinge_origin
    return np.zeros((FRAMERATE, 3))


def compute_animation_fields(
    magnet,
    path_positions,
    rot_x_arr=None,
    rot_y_arr=None,
    rot_z_arr=None,
    sensor_pos=(0, 0, 0),
):
    from scipy.spatial.transform import Rotation as R

    if rot_x_arr is None:
        rot_x_arr = np.zeros(FRAMERATE)
    if rot_y_arr is None:
        rot_y_arr = np.zeros(FRAMERATE)
    if rot_z_arr is None:
        rot_z_arr = np.zeros(FRAMERATE)

    fields = []
    original_pos = magnet.position.copy()
    identity_orientation = R.from_euler("xyz", [0, 0, 0], degrees=True)

    for i, pos in enumerate(path_positions):
        magnet.position = pos
        magnet.orientation = identity_orientation

        if rot_x_arr[i] != 0 or rot_y_arr[i] != 0 or rot_z_arr[i] != 0:
            magnet.rotate_from_euler(
                [rot_x_arr[i], rot_y_arr[i], rot_z_arr[i]], "xyz", degrees=True
            )

        fields.append(magpy.getB(magnet, sensor_pos) * 10.0)

    magnet.position = original_pos
    magnet.orientation = identity_orientation

    return np.array(fields)


# --- Constants ---
SIMULATIONS = [
    {
        "key": "slide_by",
        "name": "Linear, Slide-By",
        "function": "Linear",
        "magnet_shape": "Axial Cylinder",
        "image": "assets/slide_by.png",
    },
    {
        "key": "rotation_radial",
        "name": "Rotation or Spin",
        "function": "Rotation",
        "magnet_shape": "Radial Cylinder",
        "image": "assets/cylinder_spin.png",
    },
    {
        "key": "arc",
        "name": "Arc (Hinge)",
        "function": "Arc",
        "magnet_shape": "Axial Cylinder",
        "image": "assets/arc_hinge.png",
    },
    {
        "key": "head_on",
        "name": "Linear, Head-On",
        "function": "Linear",
        "magnet_shape": "Axial Cylinder",
        "image": "assets/head_on.png",
    },
    {
        "key": "rotation_ring",
        "name": "Rotation or Spin (Ring)",
        "function": "Rotation",
        "magnet_shape": "Ring",
        "image": "assets/rotation_spin.png",
    },
]

REMANENCE_PRESETS = {
    "N35": 11700, "N38": 12200, "N40": 12500, "N42": 13000,
    "N45": 13500, "N48": 14000, "N50": 14200, "N52": 14500,
    "N55": 15000, "Custom": 1000,
}

PRESET_CONFIGS = {
    "slide_by": {
        "name": "Linear, Slide-By",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": -10.0, "start_y": 0.0, "start_z": 7.0,
        "end_x": 10.0, "end_y": 0.0, "end_z": 7.0,
        "rot_x_start": 0.0, "rot_y_start": 0.0, "rot_z_start": 0.0,
        "rot_x_end": 0.0, "rot_y_end": 0.0, "rot_z_end": 0.0,
    },
    "rotation_radial": {
        "name": "Rotation or Spin",
        "shape": "Cylinder (Radially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0, "start_y": 0.0, "start_z": 7.0,
        "end_x": 0.0, "end_y": 0.0, "end_z": 7.0,
        "rot_x_start": 0.0, "rot_y_start": 0.0, "rot_z_start": 0.0,
        "rot_x_end": 0.0, "rot_y_end": 0.0, "rot_z_end": 359.9,
    },
    "arc": {
        "name": "Arc (Hinge)",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Hinge (Door/Lid)",
        "hinge_origin_x": 0.0, "hinge_origin_y": 0.0, "hinge_origin_z": 0.0,
        "sensor_x": 0.0, "sensor_y": 0.0, "sensor_z": 0.0,
        "hinge_radius": 7.0,
        "hinge_angle_open": 90.0,
        "hinge_angle_close": 0.0,
        "hinge_plane": "XZ (side hinge)",
        "hinge_bounce": True,
        "rot_x_start": 0.0, "rot_y_start": 0.0, "rot_z_start": 0.0,
        "rot_x_end": 0.0, "rot_y_end": 90.0, "rot_z_end": 0.0,
    },
    "head_on": {
        "name": "Linear, Head-On",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0, "start_y": 0.0, "start_z": 3.0,
        "end_x": 0.0, "end_y": 0.0, "end_z": 15.0,
        "rot_x_start": 0.0, "rot_y_start": 0.0, "rot_z_start": 0.0,
        "rot_x_end": 0.0, "rot_y_end": 0.0, "rot_z_end": 0.0,
    },
    "rotation_ring": {
        "name": "Rotation or Spin (Ring)",
        "shape": "Ring (Hollow Cylinder, Radial)",
        "magnet_type": "N42",
        "height": 1.0,
        "diameter": 2.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0, "start_y": 0.0, "start_z": 5.0,
        "end_x": 0.0, "end_y": 0.0, "end_z": 5.0,
        "rot_x_start": 0.0, "rot_y_start": 0.0, "rot_z_start": 0.0,
        "rot_x_end": 0.0, "rot_y_end": 0.0, "rot_z_end": 359.9,
    },
}

SHAPE_OPTIONS = [
    "Cylinder (Axially Magnetized)",
    "Cylinder (Radially Magnetized)",
    "Cuboid (Rectangular)",
    "Ring (Hollow Cylinder, Radial)",
    "Sphere (Spherical)",
]

SHAPE_TYPE_MAP = {
    "Cylinder (Axially Magnetized)": "Cylinder",
    "Cylinder (Radially Magnetized)": "Cylinder",
    "Cuboid (Rectangular)": "Cuboid",
    "Ring (Hollow Cylinder, Radial)": "Ring",
    "Sphere (Spherical)": "Sphere",
}

MOTION_OPTIONS = ["Position-Based", "Hinge (Door/Lid)"]

# --- Page Config ---
st.set_page_config(layout="wide", page_title="3D Magnetic Field Calculator")

# --- Session State ---
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = None

# --- CSS ---
st.markdown("""
    <style>
    .card {
        border: 1px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
        margin-bottom: 4px;
    }
    .card-image {
        background-color: #fde8e8;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 24px;
        min-height: 140px;
    }
    .card-image img {
        max-height: 100px;
        object-fit: contain;
    }
    .card-body {
        padding: 12px 16px;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #111;
    }
    .card-meta {
        font-size: 0.82rem;
        display: flex;
        justify-content: space-between;
        margin-bottom: 2px;
    }
    .card-meta .fn { color: #c0392b; }
    .card-meta .shape { color: #555; }
    div[data-testid="stButton"] button {
        border-radius: 4px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- Cards ---
st.title("Magnetic Field Calculator")
st.write("")

card_cols = st.columns(len(SIMULATIONS))

for col, sim in zip(card_cols, SIMULATIONS):
    with col:
        st.markdown(f"""
            <div class="card">
                <div class="card-image">
                    <img src="app/static/{sim['image']}" />
                </div>
                <div class="card-body">
                    <div class="card-title">{sim['name']}</div>
                    <div class="card-meta">
                        <span class="fn">Function: {sim['function']}</span>
                        <span class="shape">Magnet: {sim['magnet_shape']}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Start", key=f"btn_{sim['key']}", type="primary"):
            st.session_state.selected_preset = sim["key"]
            st.session_state.fig_params = None  # Force rebuild
            st.rerun()

st.divider()

# --- Sidebar ---
with st.sidebar:
    preset_config = None
    if st.session_state.selected_preset:
        preset_config = PRESET_CONFIGS.get(st.session_state.selected_preset)
        if preset_config:
            st.info(f"📋 Preset: {preset_config['name']}")
            if st.button("Clear Preset"):
                st.session_state.selected_preset = None
                st.session_state.fig_params = None
                st.rerun()

    st.header("Magnet Settings")

    shape = st.selectbox(
        "Magnet Shape",
        SHAPE_OPTIONS,
        index=SHAPE_OPTIONS.index(preset_config["shape"]) if preset_config and preset_config.get("shape") in SHAPE_OPTIONS else 0,
    )
    shape_type = SHAPE_TYPE_MAP[shape]

    magnet_type = st.selectbox(
        "Magnet Type",
        list(REMANENCE_PRESETS.keys()),
        index=(
            list(REMANENCE_PRESETS.keys()).index(preset_config["magnet_type"])
            if preset_config
            else 0
        ),
    )
    default_rem = REMANENCE_PRESETS[magnet_type]
    remanence_g = st.number_input(
        "Remanence (Gauss)",
        value=(
            float(REMANENCE_PRESETS.get(preset_config["magnet_type"], default_rem))
            if preset_config
            else default_rem
        ),
        disabled=magnet_type != "Custom",
    )

    height = st.number_input(
        "Height (mm)",
        value=preset_config["height"] if preset_config else 2.0,
        min_value=0.1,
        step=0.5,
    )

    if shape_type in ("Cylinder", "Ring", "Sphere"):
        diameter = st.number_input(
            "Diameter (mm)",
            value=preset_config["diameter"] if preset_config else 2.0,
            min_value=0.1,
            step=0.5,
        )
        dims = diameter
    else:
        width = st.number_input("Width (mm)", value=2.0)
        length = st.number_input("Length (mm)", value=2.0)
        dims = (width, length)

    z_air_gap = st.number_input(
        "Z Air Gap (mm)",
        value=preset_config["z_air_gap"] if preset_config else 2.0,
        min_value=0.0,
        step=0.1,
    )

    st.markdown("---")
    st.header("Animation Settings")

    st.subheader("Magnet Rotation (Start → End)")
    col_rot1, col_rot2 = st.columns(2)
    with col_rot1:
        st.write("**Starting Rotation**")
        rot_x_start = st.number_input(
            "Start Rotate X (°)",
            value=preset_config["rot_x_start"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )
        rot_y_start = st.number_input(
            "Start Rotate Y (°)",
            value=preset_config["rot_y_start"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )
        rot_z_start = st.number_input(
            "Start Rotate Z (°)",
            value=preset_config["rot_z_start"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )
    with col_rot2:
        st.write("**Ending Rotation**")
        rot_x_end = st.number_input(
            "End Rotate X (°)",
            value=preset_config["rot_x_end"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )
        rot_y_end = st.number_input(
            "End Rotate Y (°)",
            value=preset_config["rot_y_end"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )
        rot_z_end = st.number_input(
            "End Rotate Z (°)",
            value=preset_config["rot_z_end"] if preset_config else 0.0,
            min_value=-360.0, max_value=360.0, step=5.0,
        )

    st.markdown("---")
    st.subheader("Magnet Position")
    col_pos1, col_pos2 = st.columns(2)
    with col_pos1:
        st.write("**Starting Position**")
        start_x = st.number_input(
            "Start X (mm)",
            value=preset_config.get("start_x", 0.0) if preset_config else 0.0, step=0.5,
        )
        start_y = st.number_input(
            "Start Y (mm)",
            value=preset_config.get("start_y", 0.0) if preset_config else 0.0, step=0.5,
        )
        start_z = st.number_input(
            "Start Z (mm)",
            value=preset_config.get("start_z", 5.0) if preset_config else 5.0, step=0.5,
        )
    with col_pos2:
        st.write("**Ending Position**")
        end_x = st.number_input(
            "End X (mm)",
            value=preset_config.get("end_x", 15.0) if preset_config else 15.0, step=0.5,
        )
        end_y = st.number_input(
            "End Y (mm)",
            value=preset_config.get("end_y", 0.0) if preset_config else 0.0, step=0.5,
        )
        end_z = st.number_input(
            "End Z (mm)",
            value=preset_config.get("end_z", 5.0) if preset_config else 5.0, step=0.5,
        )

    st.markdown("---")
    st.subheader("Motion Path")

    motion_type = st.selectbox(
        "Motion Path",
        MOTION_OPTIONS,
        index=(
            MOTION_OPTIONS.index(preset_config["motion_type"])
            if preset_config and preset_config.get("motion_type") in MOTION_OPTIONS
            else 0
        ),
    )

    # Hinge-specific params (only shown when relevant)
    hinge_radius = hinge_angle_open = hinge_angle_close = None
    hinge_plane = hinge_sensor_offset = hinge_bounce = None
    hinge_origin_x = hinge_origin_y = hinge_origin_z = None
    sensor_x = sensor_y = sensor_z = None

    if motion_type == "Hinge (Door/Lid)":
        st.markdown("---")
        st.subheader("Hinge Origin Position")
        hinge_origin_x = st.number_input(
            "Hinge Origin X (mm)",
            value=preset_config.get("hinge_origin_x", 0.0) if preset_config else 0.0, step=0.5,
        )
        hinge_origin_y = st.number_input(
            "Hinge Origin Y (mm)",
            value=preset_config.get("hinge_origin_y", 0.0) if preset_config else 0.0, step=0.5,
        )
        hinge_origin_z = st.number_input(
            "Hinge Origin Z (mm)",
            value=preset_config.get("hinge_origin_z", 0.0) if preset_config else 0.0, step=0.5,
        )

        st.markdown("---")
        st.subheader("Sensor Position")
        sensor_x = st.number_input(
            "Sensor X (mm)",
            value=preset_config.get("sensor_x", 0.0) if preset_config else 0.0, step=0.5,
        )
        sensor_y = st.number_input(
            "Sensor Y (mm)",
            value=preset_config.get("sensor_y", 0.0) if preset_config else 0.0, step=0.5,
        )
        sensor_z = st.number_input(
            "Sensor Z (mm)",
            value=preset_config.get("sensor_z", 0.0) if preset_config else 0.0, step=0.5,
        )

        st.markdown("---")
        st.subheader("Hinge Parameters")
        hinge_radius = st.number_input(
            "Hinge Arm Length (mm)",
            value=preset_config.get("hinge_radius", 15.0) if preset_config else 15.0,
            min_value=1.0, step=1.0,
        )
        hinge_angle_open = st.slider(
            "Open Angle (°)", 0, 180,
            int(preset_config.get("hinge_angle_open", 90)) if preset_config else 90,
            step=5,
        )
        hinge_angle_close = st.slider(
            "Closed Angle (°)", 0, 180,
            int(preset_config.get("hinge_angle_close", 0)) if preset_config else 0,
            step=5,
        )
        hinge_plane = st.selectbox(
            "Hinge Rotation Plane",
            ["XZ (side hinge)", "YZ (top hinge)", "XY (flat spin)"],
            index=["XZ (side hinge)", "YZ (top hinge)", "XY (flat spin)"].index(
                preset_config.get("hinge_plane", "XZ (side hinge)")
            ) if preset_config and preset_config.get("hinge_plane") in ["XZ (side hinge)", "YZ (top hinge)", "XY (flat spin)"] else 0,
        )
        hinge_bounce = st.checkbox(
            "Bounce (close → open → close)",
            value=preset_config.get("hinge_bounce", True) if preset_config else True,
        )

# --- Static Calculations ---
magnet_obj, sensor_obj = get_magnet_and_sensor(shape, height, dims, remanence_g, z_air_gap)
result_gauss = calculate_field(magnet_obj, sensor_obj)
curve_gaps, curve_b = generate_curve(shape, height, dims, remanence_g)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Result (Bz)")
    st.metric("Magnetic Flux Density", f"{result_gauss:.1f} G")
    df_static = pd.DataFrame({"AirGap_mm": curve_gaps, "Bz_Gauss": curve_b})
    csv_data = df_static.to_csv(index=False) + f"\n{DISCLAIMER}"
    st.download_button(
        "Export CSV",
        csv_data.encode(),
        "magnet_data.csv",
        "text/csv",
    )

st.subheader("Animated Magnet Motion")

anim_magnet = create_magnet(shape, height, dims, remanence_g)

if motion_type == "Position-Based":
    path_kwargs = {
        "start_pos": (start_x, start_y, start_z),
        "end_pos": (end_x, end_y, end_z),
    }
    sensor_world_pos = (0, 0, 0)
else:  # Hinge (Door/Lid)
    path_kwargs = {
        "hinge_radius": hinge_radius or 15.0,
        "angle_open": hinge_angle_open or 90,
        "angle_close": hinge_angle_close or 0,
        "plane": hinge_plane or "XZ (side hinge)",
        "bounce": hinge_bounce if hinge_bounce is not None else True,
        "hinge_origin": (
            hinge_origin_x or 0.0,
            hinge_origin_y or 0.0,
            hinge_origin_z or 0.0,
        ),
    }
    sensor_world_pos = (
        sensor_x or 0.0,
        sensor_y or 0.0,
        sensor_z or 0.0,
    )

try:
    anim_path = generate_animation_path(motion_type, **path_kwargs)
    rot_x_interp = np.linspace(rot_x_start, rot_x_end, FRAMERATE)
    rot_y_interp = np.linspace(rot_y_start, rot_y_end, FRAMERATE)
    rot_z_interp = np.linspace(rot_z_start, rot_z_end, FRAMERATE)
    anim_fields = compute_animation_fields(
        anim_magnet, anim_path,
        rot_x_interp, rot_y_interp, rot_z_interp,
        sensor_pos=sensor_world_pos,
    )
except Exception as e:
    st.error(f"Animation error: {e}")
    anim_path = None
    anim_fields = None

if anim_path is None or anim_fields is None:
    st.warning("Could not compute animation with current settings.")
else:
    bx = anim_fields[:, 0]
    by = anim_fields[:, 1]
    bz = anim_fields[:, 2]
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    frame_idx = np.arange(FRAMERATE)

    if motion_type == "Hinge (Door/Lid)":
        half = FRAMERATE // 2
        if hinge_bounce:
            angles_deg = np.concatenate([
                np.linspace(hinge_angle_close, hinge_angle_open, half),
                np.linspace(hinge_angle_open, hinge_angle_close, FRAMERATE - half),
            ])
        else:
            angles_deg = np.linspace(hinge_angle_close, hinge_angle_open, FRAMERATE)
        x_axis_data, x_axis_label = angles_deg, "Hinge Angle (°)"
    else:  # Position-Based
        path_distances = np.zeros(FRAMERATE)
        for i in range(1, FRAMERATE):
            path_distances[i] = path_distances[i - 1] + np.linalg.norm(
                anim_path[i] - anim_path[i - 1]
            )
        total_rot = np.sqrt(
            (rot_x_end - rot_x_start) ** 2
            + (rot_y_end - rot_y_start) ** 2
            + (rot_z_end - rot_z_start) ** 2
        )
        if path_distances[-1] < 0.01 and total_rot > 0.01:
            rot_sweep = np.linspace(0, total_rot, FRAMERATE)
            x_axis_data, x_axis_label = rot_sweep, "Rotation Angle (°)"
        else:
            x_axis_data, x_axis_label = path_distances, "Distance (mm)"

    if motion_type == "Hinge (Door/Lid)":
        step_label = lambda i: f"{x_axis_data[i]:.0f}°"
    else:
        step_label = lambda i: f"{x_axis_data[i]:.0f} mm"

    x_min = float(x_axis_data[0])
    x_max = float(x_axis_data[-1])
    x_range = x_max - x_min

    # Define meaningful notch intervals based on the range
    if motion_type == "Hinge (Door/Lid)":
        # For angles: use 5° or 10° intervals depending on range
        if x_range <= 45:
            notch_interval = 5.0
        elif x_range <= 90:
            notch_interval = 10.0
        else:
            notch_interval = 15.0
        suffix = "°"
        prefix = "Angle: "
    else:
        # For distance: use 1mm intervals for short moves, 5mm for longer
        if x_range <= 10:
            notch_interval = 1.0
        elif x_range <= 30:
            notch_interval = 1.0
        elif x_range <= 60:
            notch_interval = 5.0
        else:
            notch_interval = 10.0
        suffix = " mm"
        prefix = "Position: "

    # Generate notch positions
    notch_positions = np.arange(
        np.ceil(x_min / notch_interval) * notch_interval,
        x_max + notch_interval * 0.5,
        notch_interval
    )

    # For each notch, find the closest frame index
    def find_nearest_frame(target_value):
        """Find the frame index whose x_axis_data value is closest to target."""
        idx = np.argmin(np.abs(x_axis_data - target_value))
        return int(idx)

    # Build slider steps - one per notch
    slider_steps = []
    for notch_val in notch_positions:
        frame_idx = find_nearest_frame(notch_val)

        # Format label: show value at meaningful intervals
        if motion_type == "Hinge (Door/Lid)":
            label_text = f"{notch_val:.0f}"
        else:
            # For distance, show integer mm values
            label_text = f"{notch_val:.0f}"

        slider_steps.append(
            dict(
                args=[
                    [str(frame_idx)],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
                label=label_text,
                method="animate",
            )
        )

    slider_layout = [
        dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(
                prefix=prefix,
                suffix=suffix,
                visible=True,
            ),
            pad=dict(t=50),
        )
    ]

    play_pause_buttons = [
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
                            frame=dict(duration=FRAME_DURATION, redraw=True),
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
    ]

    df_anim = pd.DataFrame({
        "Frame": frame_idx,
        "Magnet_X": anim_path[:, 0],
        "Magnet_Y": anim_path[:, 1],
        "Magnet_Z": anim_path[:, 2],
        "Rotate_X": rot_x_interp,
        "Rotate_Y": rot_y_interp,
        "Rotate_Z": rot_z_interp,
        "Bx_Gauss": bx,
        "By_Gauss": by,
        "Bz_Gauss": bz,
        "B_Total_Gauss": b_mag,
    })
    if motion_type == "Hinge (Door/Lid)":
        df_anim.insert(1, "Angle_deg", angles_deg)
    else:
        df_anim.insert(1, "Distance_mm", x_axis_data)

    col_3d, col_2d = st.columns(2)

    with col_3d:
        st.caption("3D magnet path and sensor position")

        # Calculate axis ranges for consistent view
        all_points = np.vstack([anim_path, [list(sensor_world_pos)]])
        if motion_type == "Hinge (Door/Lid)":
            all_points = np.vstack([all_points, [[hinge_origin_x, hinge_origin_y, hinge_origin_z]]])
        all_x, all_y, all_z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        z_mid = (all_z.max() + all_z.min()) / 2
        magnet_dim = max(height, dims if isinstance(dims, (int, float)) else max(dims))
        padding = magnet_dim * 1.5
        half_range = max(
            (all_x.max() - all_x.min()) / 2,
            (all_y.max() - all_y.min()) / 2,
            (all_z.max() - all_z.min()) / 2,
            magnet_dim,
        ) + padding

        # Create the base figure
        fig_path = go.Figure()

        fig_path.add_trace(go.Scatter3d(
            x=anim_path[:, 0], y=anim_path[:, 1], z=anim_path[:, 2],
            mode="lines", line=dict(color="royalblue", width=3), name="Magnet Path",
        ))
        fig_path.add_trace(go.Scatter3d(
            x=[sensor_world_pos[0]], y=[sensor_world_pos[1]], z=[sensor_world_pos[2]],
            mode="markers", marker=dict(size=8, color="green", symbol="diamond"),
            name="Sensor",
        ))

        # Hinge origin marker
        if motion_type == "Hinge (Door/Lid)":
            fig_path.add_trace(go.Scatter3d(
                x=[hinge_origin_x], y=[hinge_origin_y], z=[hinge_origin_z],
                mode="markers", marker=dict(size=6, color="orange", symbol="x"),
                name="Hinge Origin",
            ))

        if motion_type == "Hinge (Door/Lid)":
            fig_path.add_trace(go.Scatter3d(
                x=[hinge_origin_x, anim_path[0, 0]],
                y=[hinge_origin_y, anim_path[0, 1]],
                z=[hinge_origin_z, anim_path[0, 2]],
                mode="lines", line=dict(color="orange", width=2), showlegend=False,
            ))
        else:
            fig_path.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="none", showlegend=False,
            ))

        init_traces = make_magnet_traces(
            shape, height, dims, remanence_g,
            anim_path[0, 0], anim_path[0, 1], anim_path[0, 2],
            rot_x_start, rot_y_start, rot_z_start,
        )
        for t in init_traces:
            fig_path.add_trace(t)

        magnet_trace_indices = list(range(4 if motion_type == "Hinge (Door/Lid)" else 3, (4 if motion_type == "Hinge (Door/Lid)" else 3) + len(init_traces)))

        # Build animation frames
        frames_3d = []
        for i in range(FRAMERATE):
            arm = (
                go.Scatter3d(
                    x=[hinge_origin_x, anim_path[i, 0]],
                    y=[hinge_origin_y, anim_path[i, 1]],
                    z=[hinge_origin_z, anim_path[i, 2]],
                    mode="lines", line=dict(color="orange", width=2), showlegend=False,
                )
                if motion_type == "Hinge (Door/Lid)"
                else go.Scatter3d(x=[None], y=[None], z=[None], mode="none", showlegend=False)
            )
            magnet_traces = make_magnet_traces(
                shape, height, dims, remanence_g,
                anim_path[i, 0], anim_path[i, 1], anim_path[i, 2],
                rot_x_interp[i], rot_y_interp[i], rot_z_interp[i],
            )
            frames_3d.append(go.Frame(
                data=[arm] + magnet_traces,
                traces=[3] + magnet_trace_indices,
                name=str(i),
            ))

        fig_path.frames = frames_3d


        fig_path.update_layout(
            scene=dict(
                aspectmode="cube",
                xaxis=dict(title="X (mm)", range=[x_mid - half_range, x_mid + half_range]),
                yaxis=dict(title="Y (mm)", range=[y_mid - half_range, y_mid + half_range]),
                zaxis=dict(title="Z (mm)", range=[z_mid - half_range, z_mid + half_range]),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            updatemenus=play_pause_buttons,
            sliders=slider_layout,
        )

        st.plotly_chart(fig_path, use_container_width=True, key="plotly_3d")

    with col_2d:
        st.caption(f"Field components vs {x_axis_label}")
        fig_field = go.Figure()
        for y_data, name, color in [
            (bx, "Bx", "red"),
            (by, "By", "green"),
            (bz, "Bz", "blue"),
            (b_mag, "|B|", "black"),
        ]:
            fig_field.add_trace(go.Scatter(
                x=x_axis_data, y=y_data, mode="lines", name=name,
                line=dict(color=color, dash="dash" if name == "|B|" else "solid"),
            ))
        fig_field.add_trace(go.Scatter(
            x=[x_axis_data[0]], y=[b_mag[0]],
            mode="markers", marker=dict(size=10, color="orange"),
            name="Current", showlegend=False,
        ))

        field_frames = []
        for i in range(FRAMERATE):
            field_frames.append(go.Frame(
                data=[
                    go.Scatter(x=x_axis_data, y=bx, mode="lines", line=dict(color="red")),
                    go.Scatter(x=x_axis_data, y=by, mode="lines", line=dict(color="green")),
                    go.Scatter(x=x_axis_data, y=bz, mode="lines", line=dict(color="blue")),
                    go.Scatter(
                        x=x_axis_data, y=b_mag,
                        mode="lines", line=dict(color="black", dash="dash"),
                    ),
                    go.Scatter(
                        x=[x_axis_data[i]], y=[b_mag[i]],
                        mode="markers", marker=dict(size=12, color="orange"),
                        showlegend=False,
                    ),
                ],
                name=str(i),
            ))

        fig_field.frames = field_frames
        fig_field.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title="Field (Gauss)",
            height=500,
            margin=dict(l=20, r=20, t=30, b=80),
            updatemenus=play_pause_buttons,
            sliders=slider_layout,
            uirevision="keep"
        )
        st.plotly_chart(fig_field, use_container_width=True)

    st.subheader("Animation Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max |B|", f"{b_mag.max():.1f} G")
    s2.metric("Min |B|", f"{b_mag.min():.1f} G")
    s3.metric("Max Bz", f"{np.max(np.abs(bz)):.1f} G")
    path_len = np.sum(np.sqrt(np.sum(np.diff(anim_path, axis=0) ** 2, axis=1)))
    s4.metric("Path Length", f"{path_len:.1f} mm")

    csv_content = df_anim.to_csv(index=False) + f"\n{DISCLAIMER}"

    with st.expander("View all animation data"):
        st.dataframe(df_anim, use_container_width=True)
    st.text(DISCLAIMER)
    st.download_button(
        "Export Animation Data (CSV)",
        csv_content.encode(),
        "animation_data.csv",
        "text/csv",
    )