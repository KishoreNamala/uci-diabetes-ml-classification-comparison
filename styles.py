import streamlit as st

def inject_global_css():
    st.markdown(
        """
        <style>
          h1, h2, h3 { margin-top: 0.4rem; margin-bottom: 0.4rem; }
          .big-title {
            font-size: 2.0rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
          }
          .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
          }
          .body-text {
            font-size: 1.0rem;
            line-height: 1.4;
          }
          .small-note {
            font-size: 0.9rem;
            opacity: 0.75;
          }
          .card {
            padding: 10x 12px;
            border-radius: 14px;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(127,127,127,0.04);
            margin-bottom: 10px;
          }
          .block-container {
            padding-top: 1.4rem;
            padding-bottom: 0.8rem;
          }          
        </style>
        """,
        unsafe_allow_html=True,
    )
