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

    # Magnet Shape
    shape = st.selectbox("Magnet Shape", ["Cylinder", "Cuboid"])

    # Magnet Type (Preset Remanence values)
    magnet_type = st.selectbox("Magnet Type", ["N42", "N35", "N52", "Ceramic", "Custom"])

    remanence_presets = {
        "N35": 11700, "N42": 13000, "N52": 14500, "Ceramic": 3800, "Custom": 1000
    }

    default_rem = remanence_presets.get(magnet_type, 1000)
    disabled_rem = magnet_type != "Custom"
    remanence_g = st.number_input("Remanence (Gauss)", value=float(default_rem), disabled=disabled_rem)

    # Geometry Inputs
    height = st.number_input("Height (mm)", value=5.0, min_value=0.1, step=0.5)

    if shape == "Cylinder":
        diameter = st.number_input("Diameter (mm)", value=5.0, min_value=0.1, step=0.5)
    else:
        width = st.number_input("Width (mm)", value=10.0)
        length = st.number_input("Length (mm)", value=10.0)

    # Air Gap Input
    z_air_gap = st.number_input("Z, Air Gap (mm)", value=2.0, min_value=0.0, step=0.1)

# --- Calculation Logic ---

def get_magnet_and_sensor(shape_type, h, d_or_dim, rem_gauss, gap):
    """
    Returns the magpylib Magnet object and Sensor object
    """
    # Convert Gauss to milliTesla
    polarization_mt = rem_gauss / 10.0

    if shape_type == "Cylinder":
        magnet = magpy.magnet.Cylinder(
            polarization=(0, 0, polarization_mt),
            dimension=(d_or_dim, h),
            position=(0, 0, 0)
        )
    else:
        # For cuboid d_or_dim is actually a tuple (width, length)
        w, l = d_or_dim
        magnet = magpy.magnet.Cuboid(
            polarization=(0, 0, polarization_mt),
            dimension=(l, w, h),
            position=(0, 0, 0)
        )

    # Sensor position: Center of magnet is 0. Top surface is h/2.
    sensor_pos_z = (h / 2) + gap
    sensor = magpy.Sensor(position=(0, 0, sensor_pos_z))

    return magnet, sensor

def calculate_field(magnet, sensor):
    b_vec = magpy.getB(magnet, sensor)

    # Return Bz in Gauss (b_vec is in mT)
    return abs(b_vec[2] * 10.0)

def generate_curve(magnet, h):
    gaps = np.linspace(0, 15, 50)
    zs = (h / 2) + gaps
    path = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])

    res = []
    for p in path:
        s = magpy.Sensor(position=p)
        try:
            res.append(magpy.getB(magnet, s))
        except:
            res.append(s.get_B(magnet))
    b_fields = np.array(res)

    bz_curve_gauss = np.abs(b_fields[:, 2]) * 10.0
    return gaps, bz_curve_gauss

# Execute Logic
dims = diameter if shape == "Cylinder" else (width, length)
magnet_obj, sensor_obj = get_magnet_and_sensor(shape, height, dims, remanence_g, z_air_gap)
result_gauss = calculate_field(magnet_obj, sensor_obj)
curve_gaps, curve_b = generate_curve(magnet_obj, height)

# --- Sidebar Result ---
with st.sidebar:
    st.markdown("### Result (Bz)")
    st.metric(label="Magnetic Flux Density", value=f"{result_gauss:.1f} G")

    # Download CSV
    df = pd.DataFrame({"AirGap_mm": curve_gaps, "Bz_Gauss": curve_b})
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Export CSV", csv, "magnet_data.csv", "text/csv")

# --- Main 3D Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Device")
    st.caption("Left click to rotate. Scroll to zoom.")

    fig_3d = magpy.show(
        magnet_obj,
        sensor_obj,
        backend='plotly',
        return_fig=True,
        style_path_frames=10
    )

    fig_3d.update_layout(
        scene=dict(
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )

    st.plotly_chart(fig_3d, width='stretch')

with col2:
    st.subheader("Field Strength vs Gap")
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=curve_gaps, y=curve_b, mode='lines', name='B-Field'))
    fig_2d.add_trace(go.Scatter(x=[z_air_gap], y=[result_gauss], mode='markers', name='Current Pos', marker=dict(size=10, color='red')))

    fig_2d.update_layout(
        xaxis_title="Air Gap (mm)",
        yaxis_title="Magnetic Field (Gauss)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_2d, width='stretch')

# --- Part Matching ---
st.markdown("---")
st.subheader("Find Matching Parts")
c1, c2, c3 = st.columns(3)
with c1: dev_type = st.selectbox("Device Type", ["Omnipolar", "Unipolar", "Bipolar"])
with c2: out_type = st.selectbox("Output Type", ["Open Drain", "Push-Pull"])
with c3: volt_type = st.selectbox("Operating Voltage", ["1.6 to 5.5", "3.0 to 24"])

parts_db = [
    {"Part Number": "AH1921", "Type": "Omnipolar", "Out": "Open Drain", "Bop(Min)": 30, "Bop(Max)": 90},
    {"Part Number": "AH180", "Type": "Omnipolar", "Out": "Push-Pull", "Bop(Min)": 40, "Bop(Max)": 110},
    {"Part Number": "AH337", "Type": "Unipolar", "Out": "Open Drain", "Bop(Min)": 90, "Bop(Max)": 140},
]
df_parts = pd.DataFrame(parts_db)
filtered_df = df_parts[(df_parts["Type"] == dev_type) & (df_parts["Out"] == out_type)]
st.dataframe(filtered_df, width='stretch')

if not filtered_df.empty:
    req = filtered_df.iloc[0]["Bop(Max)"]
    if result_gauss > req:
        st.success(f"Success! {result_gauss:.1f}G > {req}G trigger point.")
    else:
        st.error(f"Too Weak. {result_gauss:.1f}G < {req}G trigger point.")