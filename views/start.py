import streamlit as st
from views.utils import go_to

SIMULATIONS = [
    {
        "key": "angle_encoding",
        "name": "Angle Encoding",
        "function": "Rotation",
        "magnet_shape": "Diametric Cylinder",
        "image": "assets/magnet.jpg",
    },
    {
        "key": "slide_by",
        "name": "Slide-By",
        "function": "Linear",
        "magnet_shape": "Axial Cylinder",
        "image": "assets/magnet.jpg",
    },
    {
        "key": "incremental_encoding",
        "name": "Incremental Encoding",
        "function": "Rotation",
        "magnet_shape": "Ring",
        "image": "assets/magnet.jpg",
    },
]

def render():
    st.markdown("""
        <style>
        .card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 0;
            overflow: hidden;
            background: #fff;
            margin-bottom: 8px;
        }
        .card-image {
            background-color: #fde8e8;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }
        .card-image img {
            max-height: 120px;
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
        .card-meta .label {
            color: #c0392b;
        }
        .card-meta .value {
            color: #555;
        }
        div[data-testid="stButton"] button {
            border-radius: 4px;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #1a5f6a;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Designs")
    st.write("")

    cols = st.columns(3)

    for col, sim in zip(cols, SIMULATIONS):
        with col:
            st.markdown(f"""
                <div class="card">
                    <div class="card-image">
                        <img src="app/static/{sim['image']}" />
                    </div>
                    <div class="card-body">
                        <div class="card-title">{sim['name']}</div>
                        <div class="card-meta">
                            <span class="label">Function: {sim['function']}</span>
                            <span class="value">Magnet shape: {sim['magnet_shape']}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.button("Start", key=f"start_{sim['key']}", type="secondary", on_click=go_to("scene"), args=(sim['key']))