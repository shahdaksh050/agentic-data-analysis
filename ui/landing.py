"""
Landing Page (Phase 2 Narrative)
Serves the custom Streamlit component for the scroll-driven landing experience.
"""
from __future__ import annotations

import os

import streamlit.components.v1 as components

# Determine the absolute path to the component directory
_component_dir = os.path.join(os.path.dirname(__file__), "landing_component")

# Declare the component
_landing_component = components.declare_component("landing", path=_component_dir)

def show_landing_page() -> bool:
    """Renders the full-screen narrative landing page.

    Returns True if the user clicked the CTA button to enter the workspace, otherwise False.
    """
    import streamlit as st

    theme = st.session_state.get("theme", "night")
    bg_color = "#f7eedd" if theme == "day" else "#130f0b"

    # Hide Streamlit UI completely and force iframe to be fixed full-screen
    st.markdown(f"""
        <style>
            header, footer, [data-testid="stSidebar"] {{ display: none !important; }}
            .block-container {{ padding: 0 !important; max-width: 100% !important; margin: 0 !important; }}
            .stApp {{ background: {bg_color} !important; }}

            iframe {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                border: none !important;
                z-index: 999999 !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Render the component. The JS will pass `true` when the button is clicked.
    clicked = _landing_component(theme=theme, key="landing_narrative")

    return bool(clicked)
