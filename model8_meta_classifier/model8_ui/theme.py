def apply_theme(st):
    st.set_page_config(
        page_title="Model 8 — Bitcoin Intelligence",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .big {font-size:26px; font-weight:700;}
        .metric {font-size:20px;}
        .good {color:#2ecc71;}
        .bad {color:#e74c3c;}
        .neutral {color:#f1c40f;}
        </style>
    """, unsafe_allow_html=True)
