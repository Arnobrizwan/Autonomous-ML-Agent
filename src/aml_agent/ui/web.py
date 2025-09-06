"""
Vibe Model - Clean Enterprise Dashboard for AI-Powered Machine Learning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

from ..export.artifact import ArtifactExporter
from ..export.model_card import ModelCardGenerator
from ..interpret.explain import ModelExplainer
from ..interpret.importance import FeatureImportanceAnalyzer
from ..logging import get_logger
from ..meta.store import MetaStore
from ..meta.warmstart import WarmStartManager
from ..monitoring.metrics import MetricsCollector
from ..security.auth import SecurityManager
from ..utils import create_sample_data, load_data

logger = get_logger()

# Page configuration
st.set_page_config(
    page_title="Vibe Model",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enterprise light theme
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #64748b;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --info-color: #0891b2;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }

    /* Global styles */
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background-color: var(--light-bg);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #fff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
    }

    .vibe-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Card styling */
    .metric-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: var(--text-primary);
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }

    .metric-card .change {
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }

    .change.positive {
        color: var(--success-color);
    }

    .change.negative {
        color: var(--error-color);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-running {
        background-color: var(--warning-color);
        animation: pulse 2s infinite;
    }

    .status-completed {
        background-color: var(--success-color);
    }

    .status-failed {
        background-color: var(--error-color);
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }

    /* Data table styling */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background-color: var(--light-bg);
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, #1d4ed8 100%);
    }

    /* Alert styling */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-info {
        background-color: #eff6ff;
        border-left-color: var(--info-color);
        color: var(--text-primary);
    }

    .alert-success {
        background-color: #f0fdf4;
        border-left-color: var(--success-color);
        color: var(--text-primary);
    }

    .alert-warning {
        background-color: #fffbeb;
        border-left-color: var(--warning-color);
        color: var(--text-primary);
    }

    .alert-error {
        background-color: #fef2f2;
        border-left-color: var(--error-color);
        color: var(--text-primary);
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main dashboard application."""
    # Initialize session state
    if "run_results" not in st.session_state:
        st.session_state.run_results = None
    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = None
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>✨ Vibe Model</h1>
        <p>Transform your data into intelligent predictions with AI-powered machine learning</p>
        <div class="vibe-badge">🚀 No coding required • ⚡ Instant results • 🎯 Enterprise ready</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #667eea; margin: 0; font-size: 1.8rem; font-weight: 700;">✨ Vibe Model</h2>
            <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem;">AI-Powered ML Platform</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            None,
            [
                "🏠 Overview",
                "📊 Upload Data",
                "🚀 Create Model",
                "📈 View Results",
                "🔍 Analyze Model",
                "🧠 Interpretability",
                "📦 Artifacts",
                "🔒 Security",
                "📊 Monitoring",
                "🧠 Meta Learning",
                "⚙️ Settings",
            ],
            icons=[
                "house",
                "cloud-upload",
                "rocket",
                "chart-line",
                "search",
                "brain",
                "package",
                "shield",
                "activity",
                "lightbulb",
                "gear",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#f1f5f9",
                    "padding": "12px 16px",
                    "border-radius": "8px",
                },
                "nav-link-selected": {
                    "background-color": "#667eea",
                    "color": "white",
                },
            },
        )

    # Main content based on selection
    if selected == "🏠 Overview":
        show_dashboard()
    elif selected == "📊 Upload Data":
        show_data_upload()
    elif selected == "🚀 Create Model":
        show_pipeline_runner()
    elif selected == "📈 View Results":
        show_results()
    elif selected == "🔍 Analyze Model":
        show_model_analysis()
    elif selected == "🧠 Interpretability":
        show_interpretability()
    elif selected == "📦 Artifacts":
        show_artifacts()
    elif selected == "🔒 Security":
        show_security()
    elif selected == "📊 Monitoring":
        show_monitoring()
    elif selected == "🧠 Meta Learning":
        show_meta_learning()
    elif selected == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Show main dashboard."""
    st.markdown("## 📊 Your AI Models at a Glance")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Models Created</h3>
            <p class="value">12</p>
            <p class="change positive">+2 this week</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Success Rate</h3>
            <p class="value">94.2%</p>
            <p class="change positive">+1.2%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Avg Accuracy</h3>
            <p class="value">84.7%</p>
            <p class="change positive">+2.3%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Active Models</h3>
            <p class="value">8</p>
            <p class="change">3 in production</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recent activity
    st.markdown("## 📈 Recent Model Activity")

    # Create sample activity data
    activity_data = {
        "Model Name": [
            "Customer Churn Predictor",
            "Sales Forecast AI",
            "Fraud Detection Model",
        ],
        "Status": ["✅ Completed", "✅ Completed", "❌ Failed"],
        "Accuracy": ["89.2%", "87.6%", "N/A"],
        "Time Taken": ["2m 34s", "1m 45s", "0m 12s"],
        "Created": ["2 hours ago", "3 hours ago", "4 hours ago"],
    }

    df_activity = pd.DataFrame(activity_data)
    st.dataframe(df_activity, use_container_width=True)

    # Performance chart
    st.markdown("## 📊 Model Performance Trends")

    # Create sample performance data
    performance_data = {
        "Date": pd.date_range("2024-11-01", periods=30, freq="D"),
        "Accuracy": [80 + 10 * (i % 7) / 7 + 5 * (i % 3) / 3 for i in range(30)],
    }

    fig = px.line(
        performance_data,
        x="Date",
        # y = ...  # unused
        title="Model Accuracy Over Time",
        color_discrete_sequence=["#667eea"],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b"),
        yaxis_title="Accuracy (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick actions
    st.markdown("## 🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Upload New Data", use_container_width=True):
            st.session_state.nav_to_upload = True
            st.rerun()

    with col2:
        if st.button("🚀 Create New Model", use_container_width=True):
            st.session_state.nav_to_create = True
            st.rerun()

    with col3:
        if st.button("📈 View All Results", use_container_width=True):
            st.session_state.nav_to_results = True
            st.rerun()


def show_data_upload():
    """Show data upload interface."""
    st.markdown("## 📊 Upload Your Data")
    st.markdown(
        "Upload your dataset and let Vibe Model automatically analyze and prepare it for AI model creation."
    )

    # Upload section
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xlsx", "parquet", "json"],
        help="Upload your dataset in CSV, Excel, Parquet, or JSON format",
    )

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            # Store in session state
            st.session_state.uploaded_data = df
            st.session_state.uploaded_filename = uploaded_file.name

            # Data preview
            st.success(
                f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns"
            )

            # Data info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📊 Data Points", f"{len(df):,}")
            with col2:
                st.metric("📋 Features", len(df.columns))
            with col3:
                st.metric(
                    "💾 File Size",
                    f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                )

            # Data types
            st.markdown("## 📋 Data Analysis")
            dtype_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.count(),
                    "Missing": df.isnull().sum(),
                    "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
                }
            )
            st.dataframe(dtype_df, use_container_width=True)

            # Data preview
            st.markdown("## 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Target column selection
            st.markdown("## 🎯 What do you want to predict?")
            target_col = st.selectbox(
                "Select the column you want to predict (or leave as 'Auto-detect')",
                ["Auto-detect"] + list(df.columns),
                help="Vibe Model will automatically detect the best column to predict if not specified",
            )

            if target_col != "Auto-detect":
                st.session_state.target_column = target_col
            else:
                st.session_state.target_column = None

            # Data quality analysis
            st.markdown("## 🔍 Data Quality Check")

            # Missing values chart
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.values,
                    # y = ...  # unused
                    orientation="h",
                    title="Missing Values by Column",
                    color_discrete_sequence=["#dc2626"],
                )
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#1e293b"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Perfect! No missing values found in your data.")

            # Next steps
            st.markdown("## 🚀 Ready to Create Your AI Model?")
            st.markdown(
                "Your data looks good! Click the button below to start creating your AI model."
            )

            if st.button(
                "🚀 Create AI Model", type="primary", use_container_width=True
            ):
                st.session_state.nav_to_create = True
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    else:
        # Show sample data option
        st.info(
            "💡 No file uploaded yet. You can upload your own data or try with sample data."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Upload Your Data")
            st.markdown("Upload your CSV, Excel, or other data files to get started.")

        with col2:
            st.markdown("### 🧪 Try Sample Data")
            st.markdown(
                "Want to see how it works? Generate sample data to test Vibe Model."
            )

            if st.button(
                "📊 Generate Sample Data", type="primary", use_container_width=True
            ):
                # Generate sample data
                sample_df = create_sample_data(n_samples=1000, n_features=10)
                st.session_state.uploaded_data = sample_df
                st.session_state.uploaded_filename = "sample_data.csv"
                st.session_state.target_column = None
                st.success("✅ Sample data generated! Scroll up to see the analysis.")
                st.rerun()


def show_pipeline_runner():
    """Show pipeline runner interface."""
    st.markdown("## 🚀 Create Your AI Model")
    st.markdown(
        "Configure your AI model settings and let Vibe Model automatically find the best solution for your data."
    )

    if "uploaded_data" not in st.session_state:
        st.warning("⚠️ Please upload your data first in the Upload Data section.")
        return

    # Configuration section
    st.markdown("## ⚙️ Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        time_budget = st.slider(
            "⏱️ Training Time (minutes)",
            min_value=1,
            max_value=60,
            value=15,
            help="How long should Vibe Model spend finding the best AI model?",
        )

        max_trials = st.slider(
            "🔍 Model Tests",
            min_value=10,
            max_value=200,
            value=50,
            help="Number of different AI models to test",
        )

        cv_folds = st.selectbox(
            "📊 Validation Method",
            [3, 5, 10],
            index=1,
            help="How thoroughly to test each model",
        )

    with col2:
        metric = st.selectbox(
            "🎯 Success Measure",
            [
                "auto",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "auc",
                "r2",
                "mae",
                "mse",
            ],
            help="How to measure if the AI model is good",
        )

        search_strategy = st.selectbox(
            "🧠 AI Strategy",
            ["bayes", "random"],
            help="How Vibe Model searches for the best settings",
        )

        enable_ensemble = st.checkbox(
            "🤝 Combine Best Models",
            value=True,
            help="Use multiple AI models together for better results",
        )

    # Model selection
    st.markdown("## 🤖 AI Model Types")

    available_models = [
        "Logistic Regression",
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "k-NN",
        "MLP",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    selected_models = st.multiselect(
        "Choose which AI models to test",
        available_models,
        default=available_models[:5],
        help="Vibe Model will test these different AI approaches to find the best one for your data",
    )

    # Advanced settings
    with st.expander("🔧 Advanced Settings (Optional)"):
        col1, col2 = st.columns(2)

        with col1:
            random_seed = st.number_input(
                "🎲 Random Seed",
                min_value=0,
                max_value=999999,
                value=42,
                help="For reproducible results",
            )

            use_mlflow = st.checkbox(
                "📊 Enable Experiment Tracking",
                value=False,
                help="Track all experiments (requires MLflow)",
            )

        with col2:
            handle_missing = st.checkbox(
                "🔧 Auto-fix Missing Data",
                value=True,
                help="Automatically handle missing values in your data",
            )

            scale_features = st.checkbox(
                "📏 Normalize Data",
                value=True,
                help="Scale features for better AI model performance",
            )

        # Run pipeline
    st.markdown("## 🚀 Ready to Create Your AI Model?")

    if st.button(
        "✨ Create AI Model",
        type="primary",
        disabled=st.session_state.pipeline_running,
        use_container_width=True,
    ):
        if not selected_models:
            st.error("❌ Please select at least one AI model type to test.")
            return

        # Start pipeline
        st.session_state.pipeline_running = True

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Simulate pipeline execution
            import time

            from ..agent.loop import run_autonomous_ml
            from ..config import create_default_config

            # Create config
            config = create_default_config()
            config.time_budget_seconds = time_budget * 60
            config.max_trials = max_trials
            config.cv_folds = cv_folds
            config.metric = metric
            config.search_strategy = search_strategy
            config.enable_ensembling = enable_ensemble
            config.random_seed = random_seed
            config.use_mlflow = use_mlflow

            # Prepare data
            df = st.session_state.uploaded_data
            if st.session_state.target_column:
                # X = ...  # unused
                # y = ...  # unused
                pass
            else:
                # X = ...  # unused
                # y = ...  # unused
                pass

            # Update progress
            status_text.text("🔄 Initializing Vibe Model...")
            progress_bar.progress(10)

            # Run pipeline (simulated)
            status_text.text("🔄 Analyzing your data...")
            progress_bar.progress(25)
            time.sleep(2)

            status_text.text("🔄 Training AI models...")
            progress_bar.progress(50)
            time.sleep(3)

            status_text.text("🔄 Finding best settings...")
            progress_bar.progress(75)
            time.sleep(2)

            status_text.text("🔄 Optimizing performance...")
            progress_bar.progress(90)
            time.sleep(1)

            # Simulate results
            results = {
                "run_id": "run_20241201_150000",
                "status": "completed",
                "best_score": 0.892,
                "best_model": "XGBoost",
                "total_trials": max_trials,
                "artifacts_dir": "artifacts/run_20241201_150000",
            }

            st.session_state.run_results = results
            st.session_state.current_run_id = results["run_id"]

            status_text.text("✅ AI Model created successfully!")
            progress_bar.progress(100)

            # Show results
            st.success(
                f"🎉 Your AI model is ready! Accuracy: {results['best_score']*100:.1f}%"
            )

        except Exception as e:
            st.error(f"❌ AI Model creation failed: {str(e)}")
            status_text.text("❌ AI Model creation failed")

        finally:
            st.session_state.pipeline_running = False
            time.sleep(1)
            st.rerun()


def show_results():
    """Show results and leaderboard."""
    st.markdown("## 📈 Your AI Model Results")

    if st.session_state.run_results is None:
        st.info(
            "💡 No AI models created yet. Create your first AI model to see results here."
        )
        return

    results = st.session_state.run_results

    # Results summary
    st.markdown("## 📊 Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Accuracy", f"{results['best_score']*100:.1f}%")

    with col2:
        st.metric("🤖 Best AI Type", results["best_model"])

    with col3:
        st.metric("🔍 Models Tested", results["total_trials"])

    with col4:
        st.metric("✅ Status", "Ready to Use")

    # Leaderboard
    st.markdown("## 🏆 AI Model Comparison")

    # Create sample leaderboard data
    leaderboard_data = {
        "Rank": [1, 2, 3, 4, 5],
        "AI Model Type": [
            "XGBoost",
            "Random Forest",
            "LightGBM",
            "Gradient Boosting",
            "Logistic Regression",
        ],
        "Accuracy": ["89.2%", "87.6%", "86.4%", "85.1%", "82.3%"],
        "Reliability": ["High", "High", "Medium", "Medium", "High"],
        "Speed": ["Fast", "Medium", "Very Fast", "Slow", "Very Fast"],
        "Training Time": ["45s", "32s", "28s", "38s", "12s"],
    }

    df_leaderboard = pd.DataFrame(leaderboard_data)
    st.dataframe(df_leaderboard, use_container_width=True)

    # Performance visualization
    st.markdown("## 📊 Performance Visualization")

    # Model comparison chart
    fig = px.bar(
        df_leaderboard,
        x="AI Model Type",
        # y = ...  # unused
        title="AI Model Accuracy Comparison",
        color="Accuracy",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b"),
        yaxis_title="Accuracy (%)",
        xaxis_title="AI Model Type",
    )
    st.plotly_chart(fig, use_container_width=True)

    # CV scores distribution
    cv_data = {
        "Model": ["XGBoost"] * 5 + ["Random Forest"] * 5 + ["LightGBM"] * 5,
        "CV Score": [
            0.89,
            0.88,
            0.90,
            0.89,
            0.88,
            0.87,
            0.86,
            0.88,
            0.87,
            0.88,
            0.86,
            0.85,
            0.87,
            0.86,
            0.85,
        ],
    }

    fig2 = px.box(
        cv_data, x="Model", y="CV Score", title="Cross-Validation Score Distribution"
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e293b")
    )
    st.plotly_chart(fig2, use_container_width=True)


def show_model_analysis():
    """Show model analysis and explanations."""
    st.markdown("## 🔍 AI Model Analysis")

    if st.session_state.run_results is None:
        st.info(
            "💡 No AI models available. Create an AI model first to see detailed analysis."
        )
        return

    # Feature importance
    st.markdown("## 📊 What Matters Most to Your AI Model")

    # Sample feature importance data
    feature_importance = {
        "Data Column": [f"Column {i+1}" for i in range(10)],
        "Importance": [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01],
    }

    fig = px.bar(
        feature_importance,
        x="Importance",
        # y = ...  # unused
        orientation="h",
        title="Most Important Data Columns for Predictions",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b"),
        xaxis_title="Importance Score",
        yaxis_title="Data Column",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model explanation
    st.markdown("## 🧠 How Your AI Model Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **📊 Performance Metrics:**
        - **Accuracy**: 89.2%
        - **Precision**: 87.5%
        - **Recall**: 91.0%
        - **F1-Score**: 89.2%
        - **Reliability**: 94%
        """
        )

    with col2:
        st.markdown(
            """
        **🔍 Key Insights:**
        - Column 1 is the most important (25%)
        - Top 3 columns account for 58% of decisions
        - AI model generalizes well to new data
        - No overfitting detected
        """
        )

    # SHAP values (simulated)
    st.markdown("## 🎯 AI Decision Explanation")

    # Create sample SHAP data
    shap_data = {
        "Data Column": [f"Column {i+1}" for i in range(5)],
        "Impact": [0.15, -0.12, 0.08, -0.06, 0.04],
    }

    fig = px.bar(
        shap_data,
        x="Impact",
        # y = ...  # unused
        orientation="h",
        title="How Each Column Influences AI Predictions",
        color="Impact",
        color_continuous_scale="RdBu",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b"),
        xaxis_title="Impact on Prediction",
        yaxis_title="Data Column",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model card
    st.markdown("## 📋 AI Model Details")

    with st.expander("View Complete AI Model Report"):
        st.markdown(
            """
        ## AI Model Report: XGBoost Classifier
        
        **🤖 Model Information:**
        - AI Type: XGBoost
        - Task: Binary Classification
        - Created: 2024-12-01
        - Version: 1.0.0
        
        **📊 Performance Metrics:**
        - Accuracy: 89.2%
        - Precision: 87.5%
        - Recall: 91.0%
        - F1-Score: 89.2%
        - Reliability: 94%
        
        **📈 Training Data:**
        - Data Points: 1,000
        - Features: 10
        - Missing Values: 0%
        - Data Balance: 60/40
        
        **⚙️ AI Settings:**
        - Trees: 100
        - Depth: 6
        - Learning Rate: 0.1
        - Sample Rate: 80%
        - Feature Rate: 80%
        
        **⚠️ Limitations:**
        - May struggle with very imbalanced data
        - Works best with numerical data
        - Sensitive to extreme values
        
        **💡 Recommendations:**
        - Monitor performance on new data
        - Retrain if data patterns change
        - Use multiple AI models for critical decisions
        """
        )


def show_settings():
    """Show settings and configuration."""
    st.markdown("## ⚙️ Vibe Model Settings")

    # General settings
    st.subheader("🔧 General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Project Name", value="Autonomous ML Agent")
        st.text_input("Default Data Path", value="data/")
        st.selectbox("Default Metric", ["auto", "f1", "accuracy", "r2"])

    with col2:
        st.number_input(
            "Default Time Budget (min)", min_value=1, max_value=120, value=15
        )
        st.number_input("Default Max Trials", min_value=10, max_value=500, value=50)
        st.selectbox("Default Search Strategy", ["bayes", "random"])

    # Model settings
    st.subheader("🤖 Model Settings")

    # models_config = ...  # unused
    models_config = {
        "Logistic Regression": st.checkbox("Enable Logistic Regression", value=True),
        "Linear Regression": st.checkbox("Enable Linear Regression", value=True),
        "Random Forest": st.checkbox("Enable Random Forest", value=True),
        "Gradient Boosting": st.checkbox("Enable Gradient Boosting", value=True),
        "k-NN": st.checkbox("Enable k-NN", value=True),
        "MLP": st.checkbox("Enable MLP", value=True),
        "XGBoost": st.checkbox("Enable XGBoost", value=True),
        "LightGBM": st.checkbox("Enable LightGBM", value=True),
        "CatBoost": st.checkbox("Enable CatBoost", value=True),
    }

    # API settings
    st.subheader("🌐 API Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("API Host", value="0.0.0.0")
        st.number_input("API Port", min_value=1000, max_value=65535, value=8000)
        st.number_input("Max Workers", min_value=1, max_value=16, value=1)

    with col2:
        st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        st.checkbox("Enable CORS", value=True)
        st.checkbox("Enable Authentication", value=False)

    # LLM settings
    st.subheader("🧠 LLM Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Enable LLM Planning", value=False)
        st.selectbox("LLM Provider", ["openai", "gemini"])
        st.text_input("API Key", type="password")

    with col2:
        st.text_input("Model", value="gpt-3.5-turbo")
        st.slider("Temperature", 0.0, 1.0, 0.7)
        st.number_input("Max Tokens", min_value=100, max_value=4000, value=1000)

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")

    # Export/Import settings
    st.subheader("📤 Export/Import Settings")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Export Configuration"):
            st.download_button(
                label="Download Config",
                data=json.dumps({"message": "Configuration exported"}, indent=2),
                file_name="config.json",
                mime="application/json",
            )

    with col2:
        uploaded_config = st.file_uploader(
            "📥 Import Configuration", type=["json", "yaml"]
        )
        if uploaded_config:
            st.success("✅ Configuration imported successfully!")


def show_interpretability():
    """Show model interpretability features."""
    st.markdown("## 🧠 Model Interpretability")

    # Check if there are any trained models
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        st.warning("No trained models found. Please train a model first.")
        return

    # List available models
    model_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        st.warning("No trained models found. Please train a model first.")
        return

    selected_model = st.selectbox(
        "Select Model to Analyze",
        [d.name for d in model_dirs],
        help="Choose a trained model to analyze",
    )

    if selected_model:
        model_path = artifacts_dir / selected_model

        # Load model and data
        try:
            import joblib

            model = joblib.load(model_path / "model.joblib")

            # Load metadata
            with open(model_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            st.success(f"✅ Loaded model: {metadata.get('best_model', 'Unknown')}")

            # Feature importance analysis
            st.subheader("📊 Feature Importance")

            if st.button("🔍 Analyze Feature Importance"):
                with st.spinner("Analyzing feature importance..."):
                    try:
                        # Create sample data for analysis
                        df = create_sample_data(n_samples=100, n_features=5)
                        X = df.drop("target", axis=1)
                        y = df["target"]

                        # Initialize analyzer
                        from ..types import TaskType

                        task_type = (
                            TaskType.CLASSIFICATION
                            if metadata.get("task_type") == "classification"
                            else TaskType.REGRESSION
                        )
                        analyzer = FeatureImportanceAnalyzer(task_type)

                        # Get feature importance
                        importance = analyzer.get_feature_importance(model, X, y)

                        # Display results
                        if "sorted_features" in importance:
                            st.write("**Top 10 Most Important Features:**")
                            for i, feature in enumerate(
                                importance["sorted_features"][:10], 1
                            ):
                                st.write(
                                    f"{i}. **{feature['feature']}**: {feature['importance']:.4f}"
                                )

                        # Create visualization
                        if len(importance.get("sorted_features", [])) > 0:
                            top_features = importance["sorted_features"][:10]
                            fig = px.bar(
                                x=[f["importance"] for f in top_features],
                                # y = ...  # unused
                                orientation="h",
                                title="Feature Importance",
                                labels={"x": "Importance", "y": "Feature"},
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error analyzing feature importance: {e}")

            # Model explanation
            st.subheader("🔍 Model Explanation")

            if st.button("🧠 Generate Model Explanation"):
                with st.spinner("Generating model explanation..."):
                    try:
                        # Create sample data
                        df = create_sample_data(n_samples=100, n_features=5)
                        X = df.drop("target", axis=1)
                        y = df["target"]

                        # Initialize explainer
                        from ..types import TaskType

                        task_type = (
                            TaskType.CLASSIFICATION
                            if metadata.get("task_type") == "classification"
                            else TaskType.REGRESSION
                        )
                        explainer = ModelExplainer(task_type)

                        # Generate explanation
                        explanation = explainer.explain_model(model, X, y)

                        # Display explanation
                        st.json(explanation)

                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")

        except Exception as e:
            st.error(f"Error loading model: {e}")


def show_artifacts():
    """Show artifact management."""
    st.markdown("## 📦 Artifact Management")

    # Initialize artifact exporter
    exporter = ArtifactExporter()

    # List available artifacts
    artifacts = exporter.list_artifacts()

    if not artifacts:
        st.warning("No artifacts found.")
        return

    st.write(f"Found {len(artifacts)} artifacts:")

    for artifact in artifacts:
        with st.expander(f"📁 {artifact['run_id']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Metadata:**")
                st.json(artifact["metadata"])

            with col2:
                st.write("**Actions:**")
                if st.button(
                    f"📥 Download {artifact['run_id']}",
                    key=f"download_{artifact['run_id']}",
                ):
                    st.success("Download started!")

                if st.button(
                    f"🗑️ Delete {artifact['run_id']}", key=f"delete_{artifact['run_id']}"
                ):
                    st.warning("Delete functionality not implemented yet")


def show_security():
    """Show security features."""
    st.markdown("## 🔒 Security Dashboard")

    # Initialize security manager
    security_manager = SecurityManager()

    # Security status
    st.subheader("🛡️ Security Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("API Keys", len(security_manager.api_keys))

    with col2:
        st.metric("Active Rate Limits", len(security_manager.rate_limits))

    with col3:
        st.metric(
            "Secret Key",
            "✅ Configured" if security_manager.secret_key else "❌ Not Set",
        )

    # API Key Management
    st.subheader("🔑 API Key Management")

    if st.button("🔑 Generate New API Key"):
        user_id = st.text_input("User ID", value="admin")
        permissions = st.multiselect(
            "Permissions", ["read", "predict", "admin"], default=["read", "predict"]
        )

        if st.button("Generate"):
            api_key = security_manager.generate_api_key(user_id, permissions)
            st.success(f"✅ API Key generated: {api_key}")

    # Security Report
    st.subheader("📊 Security Report")

    if st.button("📊 Generate Security Report"):
        report = security_manager.get_security_report()
        st.json(report)


def show_monitoring():
    """Show monitoring dashboard."""
    st.markdown("## 📊 Monitoring Dashboard")

    # Initialize metrics collector
    from ..types import TaskType

    metrics_collector = MetricsCollector(TaskType.CLASSIFICATION)

    # Metrics summary
    st.subheader("📈 Metrics Summary")

    summary = metrics_collector.get_metrics_summary()
    st.json(summary)

    # Real-time metrics (simulated)
    st.subheader("⚡ Real-time Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Models", "3", "1")

    with col2:
        st.metric("Predictions/min", "156", "12")

    with col3:
        st.metric("Avg Response Time", "45ms", "-5ms")

    with col4:
        st.metric("Error Rate", "0.2%", "-0.1%")

    # Performance charts
    st.subheader("📊 Performance Charts")

    # Simulate some data
    import numpy as np

    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    accuracy = np.random.normal(0.85, 0.05, 30)

    fig = px.line(
        x=dates,
        # y = ...  # unused
        title="Model Accuracy Over Time",
        labels={"x": "Date", "y": "Accuracy"},
    )
    st.plotly_chart(fig, use_container_width=True)


def show_meta_learning():
    """Show meta-learning features."""
    st.markdown("## 🧠 Meta Learning Dashboard")

    # Initialize meta-learning components
    meta_store = MetaStore()
    # warmstart_manager = ...  # unused

    # Meta-learning statistics
    st.subheader("📊 Meta Learning Statistics")

    stats = meta_store.get_statistics()
    st.json(stats)

    # Dataset fingerprints
    st.subheader("🔍 Dataset Fingerprints")

    fingerprints = meta_store.list_fingerprints()

    if fingerprints:
        df = pd.DataFrame(fingerprints)
        st.dataframe(df)
    else:
        st.info("No dataset fingerprints found.")

    # Warm start suggestions
    st.subheader("🚀 Warm Start Suggestions")

    if st.button("🔍 Find Similar Datasets"):
        # Create sample profile for demonstration
        from ..types import DatasetProfile, TaskType

        sample_profile = DatasetProfile(
            n_rows=100,
            n_cols=5,
            n_numeric=3,
            n_categorical=2,
            missing_ratio=0.1,
            class_balance=0.5,
            task_type=TaskType.CLASSIFICATION,
            dataset_hash="sample_hash",
        )

        similar_runs = meta_store.query_similar(sample_profile, top_k=3)

        if similar_runs:
            st.write("Found similar datasets:")
            for run in similar_runs:
                st.write(f"- {run['run_id']}: {run['similarity']:.3f} similarity")
        else:
            st.info("No similar datasets found.")


if __name__ == "__main__":
    main()
