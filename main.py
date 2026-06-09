import streamlit as st
from views import start, scene

if "page" not in st.session_state:
    st.session_state.page = "start"
PAGES = {
    "start": start,
    "scene": scene,
}

#PAGES[st.session_state.page].render()

#scene.render()