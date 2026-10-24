import sys
from pathlib import Path

# Ensure project root is in sys.path for absolute module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.ui import apply_custom_css, render_provenance_footer

# Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="PlaceWise - Campus Placement Predictor & Governance Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified design system
apply_custom_css()

# Sidebar Branding and System Status
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 18px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🎓</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">
                        PlaceWise AI
                    </h3>
                    <div style="font-size: 0.78rem; color: #818CF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                        Explainability & Governance
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; margin-bottom: 20px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 8px;">
                System Health Status
            </div>
            <div style="font-size: 0.82rem; color: #34D399; margin-bottom: 4px;">● Tuned XGBoost: <b>Online</b></div>
            <div style="font-size: 0.82rem; color: #34D399; margin-bottom: 4px;">● Platt Calibration: <b>Active</b></div>
            <div style="font-size: 0.82rem; color: #34D399; margin-bottom: 4px;">● SHAP TreeExplainer: <b>Loaded</b></div>
            <div style="font-size: 0.82rem; color: #34D399;">● Evidently Monitoring: <b>Surveilling</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page Navigation Setup
APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"

pages = {
    "Candidate Assessment": [
        st.Page(
            str(PAGES_DIR / "1_🎯_Predict.py"),
            title="Predict & What-If",
            icon="🎯",
            default=True,
        ),
    ],
    "Model Diagnostics": [
        st.Page(
            str(PAGES_DIR / "2_📊_Model_Insights.py"),
            title="Model Insights & SHAP",
            icon="📊",
        ),
    ],
    "Algorithmic Governance": [
        st.Page(
            str(PAGES_DIR / "3_⚖️_Fairness_Dashboard.py"),
            title="Fairness & Mitigation",
            icon="⚖️",
        ),
        st.Page(
            str(PAGES_DIR / "5_📈_Monitoring.py"),
            title="Surveillance & Drift",
            icon="📈",
        ),
        st.Page(
            str(PAGES_DIR / "4_ℹ️_About.py"),
            title="About & Provenance",
            icon="ℹ️",
        ),
    ],
}

nav = st.navigation(pages)
nav.run()

# Global Provenance Footer
render_provenance_footer()

with st.sidebar:
    st.markdown("---")
    st.caption("PlaceWise AI v0.1.0 • Built with Streamlit, FastAPI, XGBoost, SHAP, & Fairlearn.")
