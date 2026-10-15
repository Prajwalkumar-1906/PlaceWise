import sys
from pathlib import Path

# Ensure project root is in sys.path for absolute module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.ui import apply_custom_css

# Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="Campus Placement Predictor & Fairness Audit",
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
                        Placement Predictor
                    </h3>
                    <div style="font-size: 0.78rem; color: #818CF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                        Explainability & Fairness
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
            <div style="font-size: 0.82rem; color: #34D399; margin-bottom: 4px;">● SHAP TreeExplainer: <b>Active</b></div>
            <div style="font-size: 0.82rem; color: #34D399;">● Fairlearn Auditor: <b>Loaded</b></div>
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
            str(PAGES_DIR / "4_ℹ️_About.py"),
            title="About & Provenance",
            icon="ℹ️",
        ),
    ],
}

nav = st.navigation(pages)
nav.run()

with st.sidebar:
    st.markdown("---")
    st.caption(
        "Campus Placement Predictor v0.1.0 • Built with Streamlit, XGBoost, SHAP, & Fairlearn."
    )
