import streamlit as st
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#from utils import make_magnet_mesh, rotation_matrix, rotate_point

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

def make_magnet_traces(shape, h, d_or_dim, rem_gauss, cx, cy, cz, angle_x=0, angle_y=0, angle_z=0):
    """Create a fresh magnet, position/rotate it, extract plotly traces."""
    from scipy.spatial.transform import Rotation as R

    magnet = create_magnet(shape, h, d_or_dim, rem_gauss)
    magnet.position = (cx, cy, cz)

    identity = R.from_euler('xyz', [0, 0, 0], degrees=True)
    magnet.orientation = identity

    if angle_x != 0 or angle_y != 0 or angle_z != 0:
        magnet.rotate_from_euler([angle_x, angle_y, angle_z], 'xyz', degrees=True)

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


def generate_animation_path(motion, n_frames, **kwargs):
    if motion == "Position-Based":
        start_pos = np.array(kwargs["start_pos"])
        end_pos = np.array(kwargs["end_pos"])
        return np.array([
            start_pos + t * (end_pos - start_pos)
            for t in np.linspace(0, 1, n_frames)
        ])
    if motion == "Linear X-Sweep":
        r = kwargs["sweep_range"]
        xs = np.linspace(-r, r, n_frames)
        return np.column_stack([xs, np.zeros(n_frames), np.full(n_frames, kwargs["z_offset"])])
    elif motion == "Linear Y-Sweep":
        r = kwargs["sweep_range"]
        ys = np.linspace(-r, r, n_frames)
        return np.column_stack([np.zeros(n_frames), ys, np.full(n_frames, kwargs["z_offset"])])
    elif motion == "Linear Z-Sweep":
        r = kwargs["sweep_range"]
        zs = np.linspace(0.5, r, n_frames)
        return np.column_stack([np.zeros(n_frames), np.zeros(n_frames), zs])
    elif motion == "Circular XY":
        radius = kwargs["orbit_radius"]
        z_off = kwargs["z_offset"]
        angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_frames, z_off)])
    elif motion == "Hinge (Door/Lid)":
        radius = kwargs["hinge_radius"]
        a_open = np.radians(kwargs["angle_open"])
        a_close = np.radians(kwargs["angle_close"])
        plane = kwargs["plane"]
        bounce = kwargs["bounce"]
        if bounce:
            half = n_frames // 2
            angles = np.concatenate([
                np.linspace(a_close, a_open, half),
                np.linspace(a_open, a_close, n_frames - half),
            ])
        else:
            angles = np.linspace(a_close, a_open, n_frames)
        if plane == "XZ (side hinge)":
            return np.column_stack([radius * np.sin(angles), np.zeros(n_frames), radius * np.cos(angles)])
        elif plane == "YZ (top hinge)":
            return np.column_stack([np.zeros(n_frames), radius * np.sin(angles), radius * np.cos(angles)])
        else:
            return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_frames, kwargs.get("sensor_offset", 2.0))])
    elif motion == "Custom Path":
        waypoints = kwargs["waypoints"]
        if len(waypoints) < 2:
            return np.tile(waypoints[0], (n_frames, 1))
        total_seg = np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(total_seg)])
        total_length = cumulative[-1]
        if total_length == 0:
            return np.tile(waypoints[0], (n_frames, 1))
        t_uniform = np.linspace(0, total_length, n_frames)
        path = np.zeros((n_frames, 3))
        for i in range(3):
            path[:, i] = np.interp(t_uniform, cumulative, waypoints[:, i])
        return path
    return np.zeros((n_frames, 3))


def compute_animation_fields(magnet, path_positions, rot_x_arr=None, rot_y_arr=None, rot_z_arr=None, sensor_pos=(0, 0, 0)):
    """Compute magnetic field at sensor location as magnet moves along path."""
    from scipy.spatial.transform import Rotation as R

    n_frames = len(path_positions)

    if rot_x_arr is None:
        rot_x_arr = np.zeros(n_frames)
    if rot_y_arr is None:
        rot_y_arr = np.zeros(n_frames)
    if rot_z_arr is None:
        rot_z_arr = np.zeros(n_frames)

    fields = []
    original_pos = magnet.position.copy()
    identity_orientation = R.from_euler('xyz', [0, 0, 0], degrees=True)

    for i, pos in enumerate(path_positions):
        magnet.position = pos
        magnet.orientation = identity_orientation

        if rot_x_arr[i] != 0 or rot_y_arr[i] != 0 or rot_z_arr[i] != 0:
            magnet.rotate_from_euler(
                [rot_x_arr[i], rot_y_arr[i], rot_z_arr[i]], 'xyz', degrees=True
            )

        fields.append(magpy.getB(magnet, sensor_pos) * 10.0)

    magnet.position = original_pos
    magnet.orientation = identity_orientation

    return np.array(fields)


def parse_custom_waypoints(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    pts = []
    for line in lines:
        parts = line.split(",")
        if len(parts) == 3:
            pts.append([float(p) for p in parts])
    return np.array(pts) if pts else np.array([[0, 0, 5]])


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
    # Magnet moves laterally past the sensor (X axis)
    "slide_by": {
        "name": "Linear, Slide-By",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": -10.0,
        "start_y": 0.0,
        "start_z": 7.0,
        "end_x": 10.0,
        "end_y": 0.0,
        "end_z": 7.0,
        "rot_x_start": 0.0,
        "rot_y_start": 0.0,
        "rot_z_start": 0.0,
        "rot_x_end": 0.0,
        "rot_y_end": 0.0,
        "rot_z_end": 0.0,
        "num_frames": 60,
    },
    # Radially magnetized cylinder spins above sensor
    "rotation_radial": {
        "name": "Rotation or Spin",
        "shape": "Cylinder (Radially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0,
        "start_y": 0.0,
        "start_z": 7.0,
        "end_x": 0.0,
        "end_y": 0.0,
        "end_z": 7.0,
        "rot_x_start": 0.0,
        "rot_y_start": 0.0,
        "rot_z_start": 0.0,
        "rot_x_end": 0.0,
        "rot_y_end": 0.0,
        "rot_z_end": 359.9,
        "num_frames": 60,
    },
    # Axial cylinder swings in an arc (like a hinge/door)
    "arc": {
        "name": "Arc (Hinge)",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Hinge (Door/Lid)",
        "start_x": -10.0,
        "start_y": 0.0,
        "start_z": 7.0,
        "end_x": 10.0,
        "end_y": 0.0,
        "end_z": 7.0,
        "rot_x_start": 0.0,
        "rot_y_start": 0.0,
        "rot_z_start": 0.0,
        "rot_x_end": 0.0,
        "rot_y_end": 90.0,
        "rot_z_end": 0.0,
        "num_frames": 60,
    },
    # Magnet moves directly toward/away from sensor (Z axis)
    "head_on": {
        "name": "Linear, Head-On",
        "shape": "Cylinder (Axially Magnetized)",
        "magnet_type": "N42",
        "height": 5.0,
        "diameter": 5.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0,
        "start_y": 0.0,
        "start_z": 3.0,
        "end_x": 0.0,
        "end_y": 0.0,
        "end_z": 15.0,
        "rot_x_start": 0.0,
        "rot_y_start": 0.0,
        "rot_z_start": 0.0,
        "rot_x_end": 0.0,
        "rot_y_end": 0.0,
        "rot_z_end": 0.0,
        "num_frames": 60,
    },
    # Ring magnet spins above sensor
    "rotation_ring": {
        "name": "Rotation or Spin (Ring)",
        "shape": "Ring (Hollow Cylinder, Radial)",
        "magnet_type": "N42",
        "height": 1.0,
        "diameter": 2.0,
        "z_air_gap": 2.0,
        "motion_type": "Position-Based",
        "start_x": 0.0,
        "start_y": 0.0,
        "start_z": 5.0,
        "end_x": 0.0,
        "end_y": 0.0,
        "end_z": 5.0,
        "rot_x_start": 0.0,
        "rot_y_start": 0.0,
        "rot_z_start": 0.0,
        "rot_x_end": 0.0,
        "rot_y_end": 0.0,
        "rot_z_end": 359.9,
        "num_frames": 60,
    },
}
SHAPE_OPTIONS = [
    "Cylinder (Axially Magnetized)",
    "Cylinder (Radially Magnetized)",
    "Cuboid (Rectangular)",
    "Ring (Hollow Cylinder, Radial)",
    "Sphere (Spherical)"
]

# Map display names to internal type
SHAPE_TYPE_MAP = {
    "Cylinder (Axially Magnetized)": "Cylinder",
    "Cylinder (Radially Magnetized)": "Cylinder",
    "Cuboid (Rectangular)": "Cuboid",
    "Ring (Hollow Cylinder, Radial)": "Ring",
    "Sphere (Spherical)": "Sphere",
}

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
    .selected-card {
        border: 2px solid #1a5f6a !important;
    }
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
            st.session_state.selected_preset = sim['key']
            st.rerun()

st.divider()

# --- Sidebar ---
with st.sidebar:
    # Load preset if one is selected
    preset_config = None
    if st.session_state.selected_preset:
        preset_config = PRESET_CONFIGS.get(st.session_state.selected_preset)
        if preset_config:
            st.info(f"📋 Preset: {preset_config['name']}")
            if st.button("Clear Preset"):
                st.session_state.selected_preset = None
                st.rerun()

    st.header("Magnet Settings")

    shape = st.selectbox(
        "Magnet Shape",
        SHAPE_OPTIONS,
        index=SHAPE_OPTIONS.index(preset_config["shape"]) if preset_config else 0
    )
    shape_type = SHAPE_TYPE_MAP[shape]

    magnet_type = st.selectbox(
        "Magnet Type",
        list(REMANENCE_PRESETS.keys()),
        index=list(REMANENCE_PRESETS.keys()).index(preset_config["magnet_type"]) if preset_config else 0
    )
    default_rem = REMANENCE_PRESETS[magnet_type]
    remanence_g = st.number_input(
        "Remanence (Gauss)",
        value=float(REMANENCE_PRESETS.get(preset_config["magnet_type"], default_rem)) if preset_config else default_rem,
        disabled=magnet_type != "Custom",
    )

    height = st.number_input(
        "Height (mm)",
        value=preset_config["height"] if preset_config else 2.0,
        min_value=0.1,
        step=0.5
    )

    if shape_type in ("Cylinder", "Ring"):
        diameter = st.number_input(
            "Diameter (mm)",
            value=preset_config["diameter"] if preset_config else 2.0,
            min_value=0.1,
            step=0.5
        )
        dims = diameter
    elif shape_type == "Sphere":
        diameter = st.number_input(
            "Diameter (mm)",
            value=preset_config["diameter"] if preset_config else 2.0,
            min_value=0.1,
            step=0.5
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
        step=0.1
    )

    st.markdown("---")
    st.header("Animation Settings")

    st.subheader("Magnet Rotation (Start → End)")
    col_rot1, col_rot2 = st.columns(2)
    with col_rot1:
        st.write("**Starting Rotation**")
        rot_x_start = st.number_input("Start Rotate X (°)", value=preset_config["rot_x_start"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)
        rot_y_start = st.number_input("Start Rotate Y (°)", value=preset_config["rot_y_start"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)
        rot_z_start = st.number_input("Start Rotate Z (°)", value=preset_config["rot_z_start"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)
    with col_rot2:
        st.write("**Ending Rotation**")
        rot_x_end = st.number_input("End Rotate X (°)", value=preset_config["rot_x_end"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)
        rot_y_end = st.number_input("End Rotate Y (°)", value=preset_config["rot_y_end"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)
        rot_z_end = st.number_input("End Rotate Z (°)", value=preset_config["rot_z_end"] if preset_config else 0.0, min_value=-360.0, max_value=360.0, step=5.0)

    st.markdown("---")
    st.subheader("Magnet Position")
    col_pos1, col_pos2 = st.columns(2)
    with col_pos1:
        st.write("**Starting Position**")
        start_x = st.number_input("Start X (mm)", value=preset_config["start_x"] if preset_config else 0.0, step=0.5)
        start_y = st.number_input("Start Y (mm)", value=preset_config["start_y"] if preset_config else 0.0, step=0.5)
        start_z = st.number_input("Start Z (mm)", value=preset_config["start_z"] if preset_config else 5.0, step=0.5)
    with col_pos2:
        st.write("**Ending Position**")
        end_x = st.number_input("End X (mm)", value=preset_config["end_x"] if preset_config else 15.0, step=0.5)
        end_y = st.number_input("End Y (mm)", value=preset_config["end_y"] if preset_config else 0.0, step=0.5)
        end_z = st.number_input("End Z (mm)", value=preset_config["end_z"] if preset_config else 5.0, step=0.5)

    st.markdown("---")
    st.subheader("Motion Path")

    motion_type = st.selectbox(
        "Motion Path",
        ["Position-Based", "Linear X-Sweep", "Linear Y-Sweep", "Linear Z-Sweep", "Circular XY", "Hinge (Door/Lid)", "Custom Path"],
        index=["Position-Based", "Linear X-Sweep", "Linear Y-Sweep", "Linear Z-Sweep", "Circular XY", "Hinge (Door/Lid)", "Custom Path"].index(preset_config["motion_type"]) if preset_config else 0
    )
    num_frames = st.slider(
        "Number of Frames",
        min_value=10,
        max_value=200,
        value=preset_config["num_frames"] if preset_config else 60,
        step=10
    )

    # Motion-specific params
    sweep_range = orbit_radius = orbit_z_offset = None
    hinge_radius = hinge_angle_open = hinge_angle_close = None
    hinge_plane = hinge_sensor_offset = hinge_bounce = None
    custom_waypoints_str = None

    if motion_type in ["Linear X-Sweep", "Linear Y-Sweep", "Linear Z-Sweep"]:
        sweep_range = st.number_input("Sweep Range (mm, ±)", value=15.0, min_value=1.0, step=1.0)
    elif motion_type == "Circular XY":
        orbit_radius = st.number_input("Orbit Radius (mm)", value=10.0, min_value=1.0, step=1.0)
        orbit_z_offset = st.number_input("Z Offset (mm)", value=5.0, min_value=0.0, step=0.5)
    elif motion_type == "Hinge (Door/Lid)":
        hinge_radius = st.number_input("Hinge Arm Length (mm)", value=15.0, min_value=1.0, step=1.0)
        hinge_angle_open = st.slider("Open Angle (°)", 0, 180, 90, step=5)
        hinge_angle_close = st.slider("Closed Angle (°)", 0, 180, 0, step=5)
        hinge_plane = st.selectbox("Hinge Rotation Plane", ["XZ (side hinge)", "YZ (top hinge)", "XY (flat spin)"])
        hinge_sensor_offset = st.number_input("Sensor Offset from Pivot (mm)", value=2.0, min_value=0.0, step=0.5)
        hinge_bounce = st.checkbox("Bounce (close → open → close)", value=True)
    elif motion_type == "Custom Path":
        custom_waypoints_str = st.text_area("Waypoints", value="0,0,5\n10,0,5\n10,10,5\n0,10,5\n0,0,5")

# --- Static Calculations ---
magnet_obj, sensor_obj = get_magnet_and_sensor(shape, height, dims, remanence_g, z_air_gap)
result_gauss = calculate_field(magnet_obj, sensor_obj)
curve_gaps, curve_b = generate_curve(shape, height, dims, remanence_g)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Result (Bz)")
    st.metric("Magnetic Flux Density", f"{result_gauss:.1f} G")
    df_static = pd.DataFrame({"AirGap_mm": curve_gaps, "Bz_Gauss": curve_b})
    st.download_button("Export CSV", df_static.to_csv(index=False).encode(), "magnet_data.csv", "text/csv")

st.subheader("Animated Magnet Motion")

if shape_type == "Cylinder (Radially Magnetized)":
    radial = True
else:
    radial = False
anim_magnet = create_magnet(shape, height, dims, remanence_g)
z_offset_default = (height / 2) + z_air_gap

path_kwargs = {}
if motion_type == "Position-Based":
    path_kwargs = {
        "start_pos": (start_x, start_y, start_z),
        "end_pos": (end_x, end_y, end_z),
    }
elif motion_type in ["Linear X-Sweep", "Linear Y-Sweep"]:
    path_kwargs = {"sweep_range": sweep_range or 15.0, "z_offset": z_offset_default}
elif motion_type == "Linear Z-Sweep":
    path_kwargs = {"sweep_range": sweep_range or 15.0}
elif motion_type == "Circular XY":
    path_kwargs = {"orbit_radius": orbit_radius or 10.0, "z_offset": orbit_z_offset or 5.0}
elif motion_type == "Hinge (Door/Lid)":
    path_kwargs = {
        "hinge_radius": hinge_radius or 15.0,
        "angle_open": hinge_angle_open or 90,
        "angle_close": hinge_angle_close or 0,
        "plane": hinge_plane or "XZ (side hinge)",
        "bounce": hinge_bounce if hinge_bounce is not None else True,
        "sensor_offset": hinge_sensor_offset or 2.0,
    }
elif motion_type == "Custom Path":
    path_kwargs = {"waypoints": parse_custom_waypoints(custom_waypoints_str or "0,0,5")}

sensor_world_pos = (0, 0, 0)

try:
    anim_path = generate_animation_path(motion_type, num_frames, **path_kwargs)
    rot_x_interp = np.linspace(rot_x_start, rot_x_end, num_frames)
    rot_y_interp = np.linspace(rot_y_start, rot_y_end, num_frames)
    rot_z_interp = np.linspace(rot_z_start, rot_z_end, num_frames)
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
    frame_idx = np.arange(num_frames)

    if motion_type == "Hinge (Door/Lid)":
        half = num_frames // 2
        if hinge_bounce:
            angles_deg = np.concatenate([
                np.linspace(hinge_angle_close, hinge_angle_open, half),
                np.linspace(hinge_angle_open, hinge_angle_close, num_frames - half),
            ])
        else:
            angles_deg = np.linspace(hinge_angle_close, hinge_angle_open, num_frames)
        x_axis_data, x_axis_label = angles_deg, "Hinge Angle (°)"
        slider_prefix = "Angle: "
    elif motion_type == "Position-Based":
        path_distances = np.zeros(num_frames)
        for i in range(1, num_frames):
            path_distances[i] = path_distances[i - 1] + np.linalg.norm(
                anim_path[i] - anim_path[i - 1]
            )

        # If magnet doesn't move, use total rotation angle as x-axis instead
        total_rot = np.sqrt(
            (rot_x_end - rot_x_start) ** 2
            + (rot_y_end - rot_y_start) ** 2
            + (rot_z_end - rot_z_start) ** 2
        )
        if path_distances[-1] < 0.01 and total_rot > 0.01:
            # Pure rotation — use Z rotation angle (or whichever axis is sweeping)
            rot_sweep = np.linspace(0, total_rot, num_frames)
            x_axis_data, x_axis_label = rot_sweep, "Rotation Angle (°)"
            slider_prefix = "Angle: "
        else:
            x_axis_data, x_axis_label = path_distances, "Distance (mm)"
            slider_prefix = "Position: "

    slider_steps = [
        dict(
            args=[
                [str(i)],
                dict(
                    frame=dict(duration=50, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=f"{x_axis_data[i]:.0f}" + ("°" if motion_type == "Hinge (Door/Lid)" else ""),
            method="animate",
        )
        for i in range(num_frames)
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
    ]

    slider_layout = [
        dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(prefix=slider_prefix, visible=True),
            pad=dict(t=50),
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
    elif motion_type == "Position-Based":
        df_anim.insert(1, "Distance_mm", x_axis_data)

    col_3d, col_2d = st.columns(2)

    with col_3d:
        st.caption("3D magnet path and sensor position")
        fig_path = go.Figure()

        # Magnet path (trace 0)
        fig_path.add_trace(go.Scatter3d(
            x=anim_path[:, 0], y=anim_path[:, 1], z=anim_path[:, 2],
            mode="lines", line=dict(color="royalblue", width=3), name="Magnet Path",
        ))

        # Sensor (trace 1)
        fig_path.add_trace(go.Scatter3d(
            x=[sensor_world_pos[0]], y=[sensor_world_pos[1]], z=[sensor_world_pos[2]],
            mode="markers", marker=dict(size=8, color="green", symbol="diamond"),
            name="Sensor",
        ))

        # Hinge arm (trace 2)
        if motion_type == "Hinge (Door/Lid)":
            fig_path.add_trace(go.Scatter3d(
                x=[0, anim_path[0, 0]], y=[0, anim_path[0, 1]], z=[0, anim_path[0, 2]],
                mode="lines", line=dict(color="orange", width=2), showlegend=False,
            ))
        else:
            fig_path.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="none", showlegend=False,
            ))

        init_traces = make_magnet_traces(
            shape, height, dims, remanence_g,
            anim_path[0, 0], anim_path[0, 1], anim_path[0, 2],
            rot_x_start, rot_y_start, rot_z_start
        )
        for t in init_traces:
            fig_path.add_trace(t)

        magnet_trace_indices = list(range(3, 3 + len(init_traces)))

        # Interpolate rotation angles for each frame
        rot_x_interp = np.linspace(rot_x_start, rot_x_end, num_frames)
        rot_y_interp = np.linspace(rot_y_start, rot_y_end, num_frames)
        rot_z_interp = np.linspace(rot_z_start, rot_z_end, num_frames)

        # Frames
        frames_3d = []
        for i in range(num_frames):
            arm = (
                go.Scatter3d(
                    x=[0, anim_path[i, 0]], y=[0, anim_path[i, 1]], z=[0, anim_path[i, 2]],
                    mode="lines", line=dict(color="orange", width=2), showlegend=False,
                )
                if motion_type == "Hinge (Door/Lid)"
                else go.Scatter3d(x=[None], y=[None], z=[None], mode="none", showlegend=False)
            )
            magnet_traces = make_magnet_traces(
                shape, height, dims, remanence_g,
                anim_path[i, 0], anim_path[i, 1], anim_path[i, 2],
                rot_x_interp[i], rot_y_interp[i], rot_z_interp[i]
            )
            frames_3d.append(go.Frame(
                data=[arm] + magnet_traces,
                traces=[2] + magnet_trace_indices,
                name=str(i),
            ))

        fig_path.frames = frames_3d

        all_points = np.vstack([
            anim_path,
            [[sensor_world_pos[0], sensor_world_pos[1], sensor_world_pos[2]]]
        ])
        all_x = all_points[:, 0]
        all_y = all_points[:, 1]
        all_z = all_points[:, 2]

        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        z_mid = (all_z.max() + all_z.min()) / 2

        # Single padding value based on magnet dimensions only
        magnet_dim = max(height, dims if isinstance(dims, (int, float)) else max(dims))
        padding = magnet_dim * 1.5

        # Half-range is the largest axis span / 2 + padding (equal on all axes for cube mode)
        x_half = (all_x.max() - all_x.min()) / 2
        y_half = (all_y.max() - all_y.min()) / 2
        z_half = (all_z.max() - all_z.min()) / 2
        half_range = max(x_half, y_half, z_half, magnet_dim) + padding

        x_range = [x_mid - half_range, x_mid + half_range]
        y_range = [y_mid - half_range, y_mid + half_range]
        z_range = [z_mid - half_range, z_mid + half_range]

        fig_path.update_layout(
            scene=dict(
                aspectmode="cube",  # Keep cube aspect to prevent distortion
                xaxis=dict(title="X (mm)", range=x_range),
                yaxis=dict(title="Y (mm)", range=y_range),
                zaxis=dict(title="Z (mm)", range=z_range),
            ),
            scene_camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.9),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            updatemenus=play_pause_buttons,
            sliders=slider_layout,
        )
        st.plotly_chart(fig_path, use_container_width=True)

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
        for i in range(num_frames):
            field_frames.append(go.Frame(
                data=[
                    go.Scatter(x=x_axis_data, y=bx, mode="lines", line=dict(color="red")),
                    go.Scatter(x=x_axis_data, y=by, mode="lines", line=dict(color="green")),
                    go.Scatter(x=x_axis_data, y=bz, mode="lines", line=dict(color="blue")),
                    go.Scatter(x=x_axis_data, y=b_mag, mode="lines", line=dict(color="black", dash="dash")),
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
        )
        st.plotly_chart(fig_field, use_container_width=True)

    st.subheader("Animation Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max |B|", f"{b_mag.max():.1f} G")
    s2.metric("Min |B|", f"{b_mag.min():.1f} G")
    s3.metric("Max Bz", f"{np.max(np.abs(bz)):.1f} G")
    path_len = np.sum(np.sqrt(np.sum(np.diff(anim_path, axis=0) ** 2, axis=1)))
    s4.metric("Path Length", f"{path_len:.1f} mm")

    disclaimer = (
    "Magnetic calculations are performed via Magpylib (BSD 2-Clause, "
    "© 2019-2025). Calculation results are provided for reference only. "
    "Please contact us if you have more advanced simulation requirement."
    )

    csv_content = df_anim.to_csv(index=False) + f"\n{disclaimer}"
    with st.expander("View all animation data"):
        st.dataframe(df_anim, use_container_width=True)
    st.text(disclaimer)
    st.download_button(
        "Export Animation Data (CSV)",
        csv_content.encode(),
        "animation_data.csv",
        "text/csv",
    )
