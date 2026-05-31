# app.py — Premium Telco Churn Dashboard
"""
Production-grade, highly interactive Streamlit application for Telco Customer Churn.
Features a premium custom glassmorphic dark theme, interactive Plotly visualizations,
dynamic training configurations, a cached comparative model leaderboard, and an
intelligent, factor-driven prediction engine with an indicator gauge.
"""

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

from utils import (
    load_data,
    compute_kpis,
    get_model_registry,
    train_model,
    train_all_models,
    predict_single,
    load_saved_model,
    get_feature_importance,
    get_feature_target,
    save_model,
    PALETTE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SMOTE_AVAILABLE,
    XGBOOST_AVAILABLE,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Configuration & Global Theme Settings
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Sentry — Advanced Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark theme injection (Glassmorphism & Gradient accents)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Overrides */
    html, body, [class*='css'] {
        font-family: 'Outfit', sans-serif;
        background-color: #06020c;
        color: #e2e8f0;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #15092a 0%, #06020c 100%);
    }
    
    /* Hide Default Header/Footer elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0517 !important;
        border-right: 1px solid rgba(189, 0, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stButton {
        margin-bottom: 20px;
    }
    
    /* Glassmorphic Container Panels */
    .glass-panel {
        background: rgba(18, 9, 36, 0.5);
        border: 1px solid rgba(189, 0, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px) saturate(180%);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
    }
    
    .glass-card {
        background: linear-gradient(135deg, rgba(29, 13, 58, 0.6) 0%, rgba(18, 9, 36, 0.6) 100%);
        border: 1px solid rgba(189, 0, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(189, 0, 255, 0.4);
        box-shadow: 0 12px 24px rgba(189, 0, 255, 0.25);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #bd00ff 0%, #ff007f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #9d8fb3;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Styled Headers */
    .glowing-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ffffff 0%, #d8b4fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        border-left: 5px solid #bd00ff;
        padding-left: 15px;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #bd00ff 0%, #ff007f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Badges & Accents */
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    
    .badge-purple { background: rgba(189, 0, 255, 0.15); color: #bd00ff; border: 1px solid rgba(189, 0, 255, 0.3); }
    .badge-pink { background: rgba(255, 0, 127, 0.15); color: #ff007f; border: 1px solid rgba(255, 0, 127, 0.3); }
    .badge-cyan { background: rgba(0, 240, 255, 0.15); color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.3); }
    
    /* Custom buttons and fields */
    div.stButton > button {
        background: linear-gradient(135deg, #bd00ff 0%, #ff007f 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(189, 0, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(189, 0, 255, 0.5) !important;
    }
    
    /* Standardized margins */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Shared Helper Functions for Styling & Plots
# ──────────────────────────────────────────────
def format_plotly_figure(fig):
    """Apply standard premium dark styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family="Outfit, sans-serif",
        font_color="#cbd5e1",
        title_font_size=16,
        title_font_color="#ffffff",
        title_font_family="Outfit, sans-serif",
        legend_bgcolor='rgba(18, 9, 36, 0.6)',
        legend_bordercolor='rgba(255, 255, 255, 0.05)',
        legend_borderwidth=1,
        colorway=["#bd00ff", "#ff007f", "#00f0ff", "#8b5cf6", "#e0aaff", "#ffd700", "#ff6600"],
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor='rgba(189, 0, 255, 0.07)',
            zerolinecolor='rgba(189, 0, 255, 0.1)',
            tickfont=dict(color="#9d8fb3"),
            title=dict(font=dict(color="#cbd5e1"))
        ),
        yaxis=dict(
            gridcolor='rgba(189, 0, 255, 0.07)',
            zerolinecolor='rgba(189, 0, 255, 0.1)',
            tickfont=dict(color="#9d8fb3"),
            title=dict(font=dict(color="#cbd5e1"))
        ),
        hoverlabel=dict(
            bgcolor='#120924',
            bordercolor='#bd00ff',
            font_size=13,
            font_family="Outfit, sans-serif"
        )
    )
    return fig

def kpi_card_html(label, value, prefix="", suffix="", tooltip=""):
    """Render a premium glassmorphic KPI card."""
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""
    return f"""
    <div class="glass-card" {tooltip_attr}>
        <div class="kpi-value">{prefix}{value}{suffix}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """

# ──────────────────────────────────────────────
# Sidebar Navigation Menu
# ──────────────────────────────────────────────
def sidebar_menu():
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 1px;">📡 TELCO <span class="gradient-text">SENTRY</span></h2>
            <p style="color: #64748b; font-size: 0.8rem; margin: 4px 0 20px;">AI-Driven Churn Intelligence Hub</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    pages = {
        "🏠 Home & Impact": "page_home",
        "📂 Dataset Explorer": "page_explorer",
        "📊 EDA Dashboard": "page_eda",
        "⚙️ Model Factory": "page_training",
        "🏆 Model Comparison": "page_comparison",
        "🔮 Prediction System": "page_prediction"
    }
    
    choice = st.sidebar.selectbox("Navigate System Pages", list(pages.keys()))
    st.sidebar.markdown("---")
    
    # Show active configurations in Sidebar
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(18, 9, 36, 0.4); border-radius: 8px; padding: 12px; border: 1px solid rgba(189,0,255,0.05);">
            <h4 style="margin: 0 0 8px; color: #9d8fb3; font-size: 0.8rem; text-transform: uppercase;">Pipeline Engine Info</h4>
            <div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.6;">
                ⚡ SMOTE Oversampler: <span style="color: {'#00f0ff' if SMOTE_AVAILABLE else '#ff007f'};">{'Active' if SMOTE_AVAILABLE else 'Unavailable'}</span><br>
                🚀 XGBoost Library: <span style="color: {'#00f0ff' if XGBOOST_AVAILABLE else '#ff007f'};">{'Installed' if XGBOOST_AVAILABLE else 'Unavailable'}</span><br>
                📡 Platform Model Status: <span style="color: #bd00ff;">{'Loaded' if load_saved_model() is not None else 'Unregistered'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return pages[choice]

# ──────────────────────────────────────────────
# Page 1: Home Page & Business Hub
# ──────────────────────────────────────────────
def page_home():
    st.markdown('<div class="glowing-header">Home & Business Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #9d8fb3; font-size: 0.9rem; margin-top: -10px; margin-bottom: 20px; letter-spacing: 0.5px;">Platform Engineered & Developed by <b style="color: #bd00ff;">Sufyan</b></p>', unsafe_allow_html=True)
    
    # Load dataset & compute overall statistics
    try:
        df = load_data()
        kpis = compute_kpis(df)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # Visual Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card_html("Total Customers Analysed", f"{kpis['total_customers']:,}", tooltip="Total rows in customer database"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card_html("Overall Churn Rate", kpis['churn_rate'], suffix="%", tooltip="Percentage of lost subscribers"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card_html("Avg Monthly Revenue/Sub", kpis['avg_monthly_charges'], prefix="$", tooltip="Average monthly fee billed per account"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card_html("Average Tenure length", kpis['avg_tenure'], suffix=" mo", tooltip="Mean subscription length in months"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Description & Interactive Impact Calculator
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown(
            """
            <div class="glass-panel" style="height: 100%;">
                <h3 style="margin-top: 0; color: #ffffff;">📡 Enterprise Churn Mitigation Engine</h3>
                <p>Welcome to <b>Telco Sentry</b>, a fully integrated machine learning system designed to analyze client behavior, predict active churn risks, and provide key business metrics to boost subscriber retention.</p>
                <h4 style="color: #bd00ff;">Analytical Pipeline Overview</h4>
                <ul style="padding-left: 20px; line-height: 1.7; color: #cbd5e1;">
                    <li><b>Data Hub</b>: Load and clean structured IBM Telco Data. Handles data anomalies, cleans billing columns, and encodes categories.</li>
                    <li><b>Deep EDA</b>: Interactive graphs highlighting patterns. Compare features, extract Pearson correlations, and view feature distributions.</li>
                    <li><b>Model Factory</b>: Train high-caliber classifiers. Tune pipelines using standard features and class balancing algorithms like <b>SMOTE</b>.</li>
                    <li><b>Leaderboard</b>: Run comparison benchmarks with one click. Compares F1, Recall, Precision, Accuracy, and plots dual curves (ROC & Precision-Recall).</li>
                    <li><b>Real-Time Engine</b>: Predict risk in seconds using a robust interactive GUI. Fully explains key driving forces behind individual client churn scores.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with right_col:
        st.markdown(
            """
            <div class="glass-panel" style="height: 100%;">
                <h3 style="margin-top: 0; color: #ffffff;">💵 Financial Impact Calculator</h3>
                <p style="font-size: 0.85rem; color: #94a3b8;">Estimate the financial benefits of deploying Telco Sentry to retain accounts before they churn.</p>
            """,
            unsafe_allow_html=True
        )
        
        # User input factors for Calculator
        annual_val = st.slider("Average Annual Customer Value ($)", min_value=100, max_value=3000, value=900, step=50)
        target_retention = st.slider("Target Churn Retention Rate (%)", min_value=5, max_value=80, value=25, step=5)
        incentive_cost = st.slider("Retention Campaign Cost / Customer ($)", min_value=10, max_value=500, value=120, step=10)
        
        # Calculations
        churned_est = int(kpis["total_customers"] * (kpis["churn_rate"] / 100.0))
        saved_customers = int(churned_est * (target_retention / 100.0))
        gross_savings = saved_customers * annual_val
        campaign_expense = saved_customers * incentive_cost
        net_savings = gross_savings - campaign_expense
        
        st.markdown(
            f"""
                <hr style="border-color: rgba(255,255,255,0.05); margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #cbd5e1;">Annual Churned Subscribers (Est):</span>
                    <span style="font-weight: 600; color: #ffffff;">{churned_est:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #cbd5e1;">Subscribers Rescued:</span>
                    <span style="font-weight: 600; color: #00f0ff;">+{saved_customers:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #cbd5e1;">Campaign Expenses:</span>
                    <span style="font-weight: 600; color: #ff007f;">${campaign_expense:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 1.1rem; font-weight: 700;">
                    <span style="color: #ffffff;">Net Revenue Saved:</span>
                    <span style="color: #bd00ff;">${net_savings:,}</span>
                </div>
                <p style="font-size: 0.75rem; color: #64748b; margin: 0; line-height: 1.3;">*Calculations are estimated based on active database totals and average monthly charges.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────────
# Page 2: Dataset Explorer Page
# ──────────────────────────────────────────────
def page_explorer():
    st.markdown('<div class="glowing-header">Dataset Explorer & Profiler</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown(
        f"""
        <div class="glass-panel">
            <h3 style="margin-top: 0; color: #ffffff;">📊 Core Database Characteristics</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Inspect raw columns, schema data structures, missing elements, and individual distributions.</p>
            <span class="badge badge-cyan">Total Records: {df.shape[0]}</span>
            <span class="badge badge-purple">Total Features: {df.shape[1]}</span>
            <span class="badge badge-pink">Missing Value Ratio: 0.00%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Columns grid search
    search_term = st.text_input("🔍 Search Database (matches columns, values or client IDs)", placeholder="Type to filter rows...")
    
    if search_term:
        mask = df.apply(lambda col: col.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered = df[mask]
        st.write(f"Matches Found: **{filtered.shape[0]}** matching rows")
    else:
        filtered = df

    max_rows = st.slider("Select maximum records to preview", min_value=5, max_value=100, value=15, step=5)
    
    # Styled Grid View
    st.dataframe(filtered.head(max_rows), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grid splits for Schema details
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📋 Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
        
    with right_col:
        st.subheader("💡 Schema Features & Categories")
        types_df = pd.DataFrame({
            "DataType": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Unique Cardinality": df.nunique()
        })
        st.dataframe(types_df, use_container_width=True)

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
    
    # Dynamic Profile plot
    st.subheader("🎯 Column Distribution Profiler")
    selected_col = st.selectbox("Select a column to chart distribution profile:", df.columns, index=df.columns.get_loc("Contract"))
    
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        # Histogram & Boxplot
        fig = px.histogram(
            df, 
            x=selected_col, 
            color_discrete_sequence=[PALETTE["primary"]],
            marginal="box",
            title=f"Distribution Profile of Numerical Field: {selected_col}"
        )
    else:
        # Bar counts
        vc = df[selected_col].value_counts().reset_index()
        vc.columns = [selected_col, "Count"]
        fig = px.bar(
            vc, 
            x=selected_col, 
            y="Count", 
            color=selected_col,
            color_discrete_sequence=[PALETTE["primary"], PALETTE["purple"], PALETTE["danger"], PALETTE["success"]],
            title=f"Category Distribution Counts of Field: {selected_col}"
        )
        
    fig = format_plotly_figure(fig)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Page 3: EDA Dashboard Page
# ──────────────────────────────────────────────
def page_eda():
    st.markdown('<div class="glowing-header">Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    tab_uni, tab_bi, tab_corr, tab_imp = st.tabs([
        "📊 Univariate Analysis", 
        "⚖️ Churn Correlates (Bivariate)", 
        "🔥 Correlation Heatmap", 
        "🧬 Feature Importance Insights"
    ])
    
    # Tab 1: Univariate
    with tab_uni:
        st.markdown("<h3 style='color: #ffffff; margin-top: 10px;'>Feature Distribution Analysis</h3>", unsafe_allow_html=True)
        selected_feature = st.selectbox(
            "Pick Feature for Univariate Distribution Plot:",
            ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "InternetService"],
            key="eda_uni_select"
        )
        
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            fig = px.histogram(
                df, 
                x=selected_feature, 
                color_discrete_sequence=[PALETTE["primary"]],
                marginal="violin",
                title=f"Histogram & Violin Distribution of {selected_feature}"
            )
        else:
            fig = px.pie(
                df, 
                names=selected_feature, 
                hole=0.4,
                color_discrete_sequence=[PALETTE["primary"], PALETTE["purple"], PALETTE["success"], PALETTE["yellow"]],
                title=f"Percentage Makeup of {selected_feature}"
            )
            
        fig = format_plotly_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Bivariate (vs Churn)
    with tab_bi:
        st.markdown("<h3 style='color: #ffffff; margin-top: 10px;'>Comparing Variables Against Customer Churn</h3>", unsafe_allow_html=True)
        selected_bivar = st.selectbox(
            "Select Variable to Compare with Customer Churn Status:",
            ["Contract", "PaymentMethod", "InternetService", "gender", "SeniorCitizen", "tenure", "MonthlyCharges"],
            key="eda_bi_select"
        )
        
        if pd.api.types.is_numeric_dtype(df[selected_bivar]):
            fig = px.box(
                df, 
                x="Churn", 
                y=selected_bivar, 
                color="Churn",
                color_discrete_map={"No": PALETTE["success"], "Yes": PALETTE["danger"]},
                points="all",
                title=f"Bivariate Distribution of {selected_bivar} by Churn Status"
            )
        else:
            fig = px.histogram(
                df, 
                x=selected_bivar, 
                color="Churn", 
                barmode="group",
                color_discrete_map={"No": PALETTE["success"], "Yes": PALETTE["danger"]},
                title=f"Churn Split Counts per {selected_bivar} Category"
            )
            
        fig = format_plotly_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Correlation Matrix
    with tab_corr:
        st.markdown("<h3 style='color: #ffffff; margin-top: 10px;'>Numerical Feature Linear Correlation Matrix</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Examine linear relations between tenure, monthly fees, total accumulated charges, and the binary Churn target.</p>", unsafe_allow_html=True)
        
        numeric_cols = NUMERIC_FEATURES + ["Churn_Binary"]
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[[0, PALETTE["danger"]], [0.5, "#120924"], [1, PALETTE["success"]]],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="white", size=14)
        ))
        
        fig.update_layout(height=450, title="Interactive Pearson Heatmap Matrix")
        fig = format_plotly_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Feature Importance
    with tab_imp:
        st.markdown("<h3 style='color: #ffffff; margin-top: 10px;'>Top-Ranked Feature Predictors</h3>", unsafe_allow_html=True)
        model = load_saved_model()
        
        if model is not None:
            feat_imp = get_feature_importance(model, top_n=12)
            if not feat_imp.empty:
                fig = px.bar(
                    feat_imp, 
                    x="Importance", 
                    y="Feature", 
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, PALETTE["purple"]], [1, PALETTE["primary"]]],
                    title="Top 12 Extracted Feature Importances from Registered Champion Model"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                fig = format_plotly_figure(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance is not supported or extractable for this model class.")
        else:
            st.warning("⚠️ No active champion model found registered on disk.")
            st.info("Please navigate to the 'Model Factory' page and train a new model pipeline to view feature importance data.")

# ──────────────────────────────────────────────
# Page 4: Model Training Factory
# ──────────────────────────────────────────────
def page_training():
    st.markdown('<div class="glowing-header">Model Factory & Pipeline Trainer</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown(
            """
            <div class="glass-panel">
                <h3 style="margin-top: 0; color: #ffffff;">⚙️ Training Settings</h3>
                <p style="color: #94a3b8; font-size: 0.85rem;">Configure the dataset splitting, select the base algorithm, and toggle oversampling pipelines.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        model_options = list(get_model_registry().keys())
        selected_model = st.selectbox("Select Core Algorithm:", model_options)
        
        test_size = st.slider("Validation Holdout Ratio (Test Split):", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        
        use_smote = st.checkbox("Balance Classes using SMOTE oversampling", value=True, disabled=not SMOTE_AVAILABLE)
        if not SMOTE_AVAILABLE:
            st.caption("⚠️ SMOTE balance disabled. Install imbalanced-learn package.")
            
        train_btn = st.button("🔥 Train Classifier Pipeline")
        
    with right_col:
        st.markdown(
            """
            <div class="glass-panel" style="min-height: 380px;">
                <h3 style="margin-top: 0; color: #ffffff;">📊 Pipeline Execution Console</h3>
                <div id="status-console" style="color: #cbd5e1; font-size: 0.9rem;">
            """,
            unsafe_allow_html=True
        )
        
        if train_btn:
            with st.spinner("Executing data ingestion, building scikit-learn transformers, oversampling minority churn classes, and running fits..."):
                try:
                    # Run training logic
                    metrics, cm, fpr, tpr, prec, rec, pipeline = train_model(
                        df, 
                        selected_model, 
                        test_size=test_size
                    )
                    
                    # Store variables in streamlit session state for model saving
                    st.session_state["trained_pipeline"] = pipeline
                    st.session_state["trained_model_name"] = selected_model
                    st.session_state["trained_metrics"] = metrics
                    st.session_state["trained_cm"] = cm
                    st.session_state["trained_curves"] = (fpr, tpr, prec, rec)
                    
                    st.success(f"Pipeline Successfully Fitted! Algorithm: {selected_model}")
                except Exception as e:
                    st.error(f"Error encountered during pipeline execution: {e}")
                    return
            
            # Show Metrics
            st.markdown("<h4>📈 Validation Set Evaluation Metrics</h4>", unsafe_allow_html=True)
            
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.markdown(kpi_card_html("Accuracy", f"{metrics['Accuracy']*100:.1f}", suffix="%"), unsafe_allow_html=True)
            with m_col2:
                st.markdown(kpi_card_html("Precision", f"{metrics['Precision']*100:.1f}", suffix="%"), unsafe_allow_html=True)
            with m_col3:
                st.markdown(kpi_card_html("Recall", f"{metrics['Recall']*100:.1f}", suffix="%"), unsafe_allow_html=True)
            with m_col4:
                st.markdown(kpi_card_html("F1 Score", f"{metrics['F1 Score']:.3f}"), unsafe_allow_html=True)
            with m_col5:
                st.markdown(kpi_card_html("ROC-AUC", f"{metrics['ROC-AUC']:.3f}"), unsafe_allow_html=True)
            
            # Interactive Confusion Matrix using Plotly
            st.markdown("<h4>📋 Confusion Matrix Breakdown</h4>", unsafe_allow_html=True)
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted No Churn", "Predicted Churn"],
                    y=["True No Churn", "True Churn"],
                    colorscale=[[0, "#120924"], [1, PALETTE["primary"]]],
                    showscale=False,
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(color="white", size=18)
                ))
                fig_cm.update_layout(width=340, height=260, margin=dict(l=10, r=10, t=10, b=10))
                fig_cm = format_plotly_figure(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=False)
                
            with c_right:
                st.markdown(
                    f"""
                    <div style="background: rgba(18, 9, 36, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(189, 0, 255, 0.1); font-size: 0.85rem; height: 100%;">
                        <b style="color: #ffffff;">Classifier Diagnostics</b><br><br>
                        • <b>True Negatives</b> (Accurate retention prediction): <b>{cm[0][0]}</b><br>
                        • <b>False Positives</b> (Type-I False alarms): <b>{cm[0][1]}</b><br>
                        • <b>False Negatives</b> (Undetected high churn risks): <b>{cm[1][0]}</b><br>
                        • <b>True Positives</b> (Accurately captured churn events): <b>{cm[1][1]}</b><br><br>
                        <i>Recall (Sensitivity) is crucial here since undetected churn represents direct loss of recurring revenue.</i>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Offer model saving
            st.markdown("---")
            if st.button("💾 Register as Live Champion Model"):
                try:
                    save_model(st.session_state["trained_pipeline"])
                    st.success("🎉 Excellent! Pipeline exported and registered successfully. It is now serving the live Prediction System.")
                except Exception as e:
                    st.error(f"Failed to export pipeline: {e}")
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 60px 0; color: #64748b;">
                    <span style="font-size: 3rem;">⚙️</span>
                    <p style="margin-top: 10px;">Select parameters on the left pane and hit 'Train Classifier Pipeline' to execute model training.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("</div></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Page 5: Model Comparison & Benchmark Leaderboard
# ──────────────────────────────────────────────
def page_comparison():
    st.markdown('<div class="glowing-header">System Leaderboard & Benchmark comparisons</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown(
        """
        <div class="glass-panel">
            <h3 style="margin-top: 0; color: #ffffff;">🏆 Multi-Model Competitive Analysis</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Benchmarking several candidate pipelines side-by-side using holdout sets. Compares accuracy, precision, recall, and plotting custom metrics curves.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Execute full comparison
    if st.button("🥇 Launch Multi-Model Benchmarks"):
        with st.spinner("Benchmarking all pipelines in registry (Logistic Regression, Decision Trees, RandomForest, SVM, KNN, GradientBoost, XGBoost)..."):
            try:
                # We cache/retrieve results
                results = train_all_models(df)
            except Exception as e:
                st.error(f"Execution Error during benchmarks: {e}")
                return
                
        # Build Results DataFrame
        rows = []
        for name, (metrics, _) in results.items():
            rows.append({
                "Algorithm": name,
                "F1 Score": metrics["F1 Score"],
                "Recall (Sensitivity)": metrics["Recall"],
                "Precision": metrics["Precision"],
                "Accuracy": metrics["Accuracy"],
                "ROC-AUC": metrics["ROC-AUC"]
            })
            
        leaderboard_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
        # Add rank emojis
        leaderboard_df.insert(0, "Rank", ["🥇 First Place", "🥈 Second", "🥉 Third"] + [""] * max(0, len(leaderboard_df) - 3))
        
        # Display Leaderboard
        st.markdown("<h3>🎯 Candidate Performance Leaderboard</h3>", unsafe_allow_html=True)
        st.dataframe(
            leaderboard_df.style.background_gradient(subset=["F1 Score", "Recall (Sensitivity)", "ROC-AUC"], cmap="Purples"),
            use_container_width=True
        )
        
        # Display Best Model Badge
        champion_name = leaderboard_df.iloc[0]["Algorithm"]
        champion_f1 = leaderboard_df.iloc[0]["F1 Score"]
        champion_recall = leaderboard_df.iloc[0]["Recall (Sensitivity)"]
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(212,175,55,0.15) 0%, rgba(18,9,36,0.6) 100%); border-radius: 12px; border: 2px solid #ffd700; box-shadow: 0 0 15px rgba(255, 215, 0, 0.2); padding: 20px; margin: 20px 0; text-align: center;">
                <span style="font-size: 2.2rem;">🏆</span>
                <h3 style="margin: 8px 0; color: #ffd700;">CHAMPION CLASSIFIER: {champion_name}</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0;">Holds first place benchmark with <b>F1: {champion_f1:.4f}</b> and <b>Recall: {champion_recall*100:.1f}%</b>. Excellent balance between precision and capturing churn leaks.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Dual ROC / PR plots side-by-side
        st.markdown("<h3>📈 Dynamic Classifier Curves</h3>", unsafe_allow_html=True)
        
        curve_left, curve_right = st.columns([1, 1])
        
        # Split Data to generate probability plots
        X, y = get_feature_target(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        with curve_left:
            roc_fig = go.Figure()
            for name, (_, pipeline) in results.items():
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
                
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color="#475569"), showlegend=False))
            roc_fig.update_layout(
                title="Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450
            )
            roc_fig = format_plotly_figure(roc_fig)
            st.plotly_chart(roc_fig, use_container_width=True)
            
        with curve_right:
            pr_fig = go.Figure()
            for name, (_, pipeline) in results.items():
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=name))
                
            pr_fig.update_layout(
                title="Precision-Recall Curve (PR)",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (Reliability)",
                height=450
            )
            pr_fig = format_plotly_figure(pr_fig)
            st.plotly_chart(pr_fig, use_container_width=True)
            
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 80px 0; color: #64748b;">
                <span style="font-size: 3.5rem;">🏆</span>
                <p style="margin-top: 15px;">Click the button below to train all candidate models and benchmark them side-by-side.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────────
# Page 6: Live Prediction System Page
# ──────────────────────────────────────────────
def page_prediction():
    st.markdown('<div class="glowing-header">Live Predictive Intelligence Form</div>', unsafe_allow_html=True)
    
    # Load Registered Model from disk
    model = load_saved_model()
    
    if model is None:
        st.markdown(
            """
            <div class="glass-panel" style="text-align: center; padding: 50px 20px;">
                <span style="font-size: 3rem;">⚠️</span>
                <h3 style="color: #ff6b35;">Prediction System Offline</h3>
                <p style="color: #cbd5e1; max-width: 600px; margin: 10px auto;">
                    There is currently no champion pipeline model registered on disk under <code>models/best_churn_model.pkl</code>.
                </p>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Please proceed to the <b>Model Factory</b> page, configure training settings, hit train, and save/register the model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Visual Layout
    st.markdown("<p style='color: #cbd5e1; margin-bottom: 20px;'>Input a customer's demographic profile, connectivity settings, and billing charges below to predict their risk of churning.</p>", unsafe_allow_html=True)
    
    with st.form("churn_predict_form"):
        col_demo, col_service, col_contract = st.columns(3)
        
        with col_demo:
            st.markdown("<h4 style='color: #bd00ff; border-bottom: 1px solid rgba(189,0,255,0.1); padding-bottom: 5px; margin-top:0;'>Demographics</h4>", unsafe_allow_html=True)
            gender = st.selectbox("Gender:", ["Male", "Female"])
            senior = st.selectbox("Is Senior Citizen:", ["No", "Yes"])
            partner = st.selectbox("Has Partner:", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents:", ["No", "Yes"])
            tenure = st.slider("Tenure length (months):", min_value=1, max_value=72, value=12)
            
        with col_service:
            st.markdown("<h4 style='color: #8b5cf6; border-bottom: 1px solid rgba(189,0,255,0.1); padding-bottom: 5px; margin-top:0;'>Connectivity & Services</h4>", unsafe_allow_html=True)
            phoneservice = st.selectbox("Phone Service Status:", ["Yes", "No"])
            multiplelines = st.selectbox("Multiple Phone Lines:", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service Type:", ["Fiber optic", "DSL", "No"])
            
            # Sub-options conditional styling logic
            onlinesec = st.selectbox("Online Security Feature:", ["No", "Yes", "No internet service"])
            onlinebackup = st.selectbox("Online Backup Storage:", ["Yes", "No", "No internet service"])
            deviceprot = st.selectbox("Device Protection Insurance:", ["No", "Yes", "No internet service"])
            techsupport = st.selectbox("Premium Technical Support:", ["No", "Yes", "No internet service"])
            
        with col_contract:
            st.markdown("<h4 style='color: #00f0ff; border-bottom: 1px solid rgba(189,0,255,0.1); padding-bottom: 5px; margin-top:0;'>Billing & Contract</h4>", unsafe_allow_html=True)
            contract = st.selectbox("Contract Terms:", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing:", ["Yes", "No"])
            payment = st.selectbox("Billing Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            streamingtv = st.selectbox("Streaming TV Add-on:", ["No", "Yes", "No internet service"])
            streamingmovies = st.selectbox("Streaming Movies Add-on:", ["No", "Yes", "No internet service"])
            
            monthly = st.slider("Monthly Charges billed ($):", min_value=15.0, max_value=130.0, value=75.0, step=0.5)
            total_input = st.text_input("Total Charges Accumulated ($):", placeholder="Auto-calculated if blank...")

        st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Predict Subscription Churn Risk")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if submit:
        # Standard fallback for total charges
        if not total_input.strip():
            total = float(monthly * tenure)
        else:
            try:
                total = float(total_input)
            except ValueError:
                total = float(monthly * tenure)
                
        # Parse customer input dict
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phoneservice,
            "MultipleLines": multiplelines,
            "InternetService": internet,
            "OnlineSecurity": onlinesec,
            "OnlineBackup": onlinebackup,
            "DeviceProtection": deviceprot,
            "TechSupport": techsupport,
            "StreamingTV": streamingtv,
            "StreamingMovies": streamingmovies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": total
        }
        
        # Fetch Prediction
        with st.spinner("Analyzing risk factors, executing inference pipeline..."):
            try:
                label, proba = predict_single(model, input_dict)
                proba_pct = float(proba * 100)
            except Exception as e:
                st.error(f"Inference Failure: {e}")
                return
                
        # Split display into score layout
        out_left, out_right = st.columns([1, 1])
        
        with out_left:
            # Color code outputs
            if proba >= 0.65:
                status_color = PALETTE["danger"]
                status_header = "🔴 HIGH CHURN RISK"
                status_text = "This customer shows critical signs of leaving. Recommended immediate contact with premium retention incentives."
            elif proba >= 0.35:
                status_color = PALETTE["yellow"]
                status_header = "🟡 MEDIUM CHURN WARNING"
                status_text = "This account is exhibiting warnings. Monitor monthly bills and check online service satisfaction."
            else:
                status_color = PALETTE["success"]
                status_header = "🟢 SAFE / LOW RISK"
                status_text = "This subscriber shows strong account health and is highly likely to remain active."
                
            st.markdown(
                f"""
                <div class="glass-panel" style="border-left: 6px solid {status_color}; height: 100%;">
                    <h3 style="color: {status_color}; margin-top: 0;">{status_header}</h3>
                    <p style="font-size: 1.15rem; font-weight: 500;">Predictive Score: <b>{proba_pct:.1f}% Risk Probability</b></p>
                    <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">{status_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with out_right:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CHURN RISK METER", 'font': {'size': 16, 'family': 'Outfit, sans-serif', 'color': '#ffffff'}},
                number={'suffix': "%", 'font': {'size': 32, 'family': 'Outfit, sans-serif', 'color': '#ffffff'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': status_color},
                    'bgcolor': "rgba(18, 9, 36, 0.6)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(189, 0, 255, 0.1)",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(0, 240, 255, 0.07)'},
                        {'range': [35, 65], 'color': 'rgba(224, 170, 255, 0.07)'},
                        {'range': [65, 100], 'color': 'rgba(255, 0, 127, 0.07)'}
                    ]
                }
            ))
            
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            fig_gauge = format_plotly_figure(fig_gauge)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dynamic Risk Factor Breakdown
        st.markdown("### 🧬 Personalized Risk Drivers Breakdown")
        
        drivers = []
        
        if contract == "Month-to-month":
            drivers.append(("🚨 Month-to-Month Contract", "Month-to-month contracts have a strong empirical link with high subscriber churn. Consider offering an annual discount incentive.", "High Impact"))
        elif contract == "One year":
            drivers.append(("ℹ️ One-Year Agreement", "Subscribers on a yearly agreement show medium stability. Track expiration and renewals.", "Low Impact"))
            
        if internet == "Fiber optic":
            drivers.append(("🚨 Fiber Optic Line", "Subscribers on fiber lines have higher overall monthly charges and exhibit elevated churn trends. Validate connection stability.", "Medium Impact"))
            
        if tenure <= 12:
            drivers.append(("🚨 First-Year Lifecycle Stage", "Subscribers with less than 12 months tenure are highly susceptible to early termination. Customer onboarding support is vital.", "High Impact"))
        elif tenure >= 48:
            drivers.append(("🟢 High Long-Term Tenure", "Subscribers with more than 4 years tenure show solid platform brand loyalty. Low risk factor.", "Protective"))
            
        if techsupport == "No" and internet != "No":
            drivers.append(("⚠️ Lack of Technical Support add-on", "Accounts without premium tech support show a 40% higher rate of churn. Upsell support services.", "Medium Impact"))
            
        if onlinesec == "No" and internet != "No":
            drivers.append(("⚠️ Lack of Online Security add-on", "Security add-ons increase customer lock-in and platform dependency. Recommend safety package upgrades.", "Medium Impact"))
            
        if payment == "Electronic check":
            drivers.append(("⚠️ Manual Electronic Checks billing", "Manual monthly electronic checks lead to frequent transaction touchpoints and higher churn. Transition to autopay.", "Low Impact"))
        elif "automatic" in payment:
            drivers.append(("🟢 Autopay Enrolled", "Subscribers on automatic credit card or bank draft transfer exhibit minimal payment friction.", "Protective"))

        if not drivers:
            st.info("No standout high risk driver tags identified for this profile. Maintained account looks extremely stable.")
        else:
            grid_cols = st.columns(min(len(drivers), 3))
            for idx, (title, desc, impact) in enumerate(drivers):
                with grid_cols[idx % 3]:
                    color_tag = PALETTE["danger"] if "High" in impact else (PALETTE["yellow"] if "Medium" in impact else (PALETTE["purple"] if "Low" in impact else PALETTE["success"]))
                    st.markdown(
                        f"""
                        <div style="background: rgba(18, 9, 36, 0.4); border: 1px solid rgba(189, 0, 255, 0.1); border-radius: 10px; padding: 16px; height: 100%;">
                            <span style="font-weight:600; color:#ffffff; font-size: 0.95rem;">{title}</span>
                            <span class="badge" style="background: rgba(255, 255, 255, 0.05); color: {color_tag}; border: 1px solid {color_tag}; font-size: 0.7rem; float: right;">{impact}</span>
                            <p style="color:#cbd5e1; font-size: 0.82rem; margin: 10px 0 0; line-height: 1.45;">{desc}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ──────────────────────────────────────────────
# Main Application Router
# ──────────────────────────────────────────────
if __name__ == "__main__":
    page_func_name = sidebar_menu()
    globals()[page_func_name]()
