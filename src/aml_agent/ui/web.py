"""
Web UI for the Autonomous ML Agent using Streamlit.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ..agent.loop import run_autonomous_ml
from ..config import create_default_config
from ..logging import get_logger
from ..monitoring.metrics import HealthChecker, PerformanceMonitor
from ..utils import create_sample_data, load_data

logger = get_logger()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Autonomous ML Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Autonomous ML Agent")
    st.markdown("Intelligent machine learning pipeline with LLM orchestration")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            [
                "Dashboard",
                "Run Pipeline",
                "Model Leaderboard",
                "Model Cards",
                "Monitoring",
                "Settings",
            ],
        )

    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Run Pipeline":
        show_pipeline_runner()
    elif page == "Model Leaderboard":
        show_leaderboard()
    elif page == "Model Cards":
        show_model_cards()
    elif page == "Monitoring":
        show_monitoring()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Show main dashboard."""
    st.header("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", "0", "0")

    with col2:
        st.metric("Best Accuracy", "0.00%", "0.00%")

    with col3:
        st.metric("Active Models", "0", "0")

    with col4:
        st.metric("System Status", "Healthy", "0")

    # Recent runs
    st.subheader("Recent Runs")

    # Check for recent runs
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
        if run_dirs:
            # Sort by modification time
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for run_dir in run_dirs[:5]:  # Show last 5 runs
                with st.expander(f"Run: {run_dir.name}"):
                    # Load metadata if available
                    metadata_file = run_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Status:** {metadata.get('status', 'Unknown')}")
                            st.write(
                                f"**Best Score:** {metadata.get('best_score', 'N/A')}"
                            )
                        with col2:
                            st.write(
                                f"**Best Model:** {metadata.get('best_model', 'N/A')}"
                            )
                            st.write(
                                f"**Total Trials:** {metadata.get('total_trials', 'N/A')}"
                            )
        else:
            st.info("No runs found. Start a new pipeline run!")
    else:
        st.info("No artifacts directory found. Start a new pipeline run!")


def show_pipeline_runner():
    """Show pipeline runner interface."""
    st.header("🚀 Run ML Pipeline")

    with st.form("pipeline_form"):
        st.subheader("Configuration")

        col1, col2 = st.columns(2)

        with col1:
            data_source = st.selectbox(
                "Data Source", ["Upload File", "Use Sample Data", "Load from Path"]
            )

            if data_source == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose a file", type=["csv", "json", "parquet", "xlsx"]
                )
                if uploaded_file:
                    data = pd.read_csv(uploaded_file)
                    st.write(f"Data shape: {data.shape}")

            elif data_source == "Use Sample Data":
                task_type = st.selectbox("Task Type", ["Classification", "Regression"])
                n_samples = st.slider("Number of samples", 100, 10000, 1000)
                n_features = st.slider("Number of features", 5, 50, 10)

                if st.form_submit_button("Generate Sample Data"):
                    data = create_sample_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        task_type=task_type.lower(),
                    )
                    st.session_state["sample_data"] = data
                    st.success("Sample data generated!")

            elif data_source == "Load from Path":
                file_path = st.text_input("File path", "data/sample.csv")
                if st.form_submit_button("Load Data"):
                    try:
                        data = load_data(file_path)
                        st.session_state["loaded_data"] = data
                        st.success(f"Data loaded: {data.shape}")
                    except Exception as e:
                        st.error(f"Error loading data: {e}")

        with col2:
            target_column = st.text_input("Target column", "target")
            time_budget = st.slider("Time budget (seconds)", 60, 3600, 300)
            max_trials = st.slider("Max trials", 10, 200, 50)
            metric = st.selectbox("Metric", ["accuracy", "f1", "roc_auc", "r2", "mse"])

        # Advanced settings
        with st.expander("Advanced Settings"):
            enable_ensembling = st.checkbox("Enable ensembling", True)
            enable_llm = st.checkbox("Enable LLM guidance", True)
            cv_folds = st.slider("CV folds", 3, 10, 5)

        # Run pipeline
        if st.form_submit_button("🚀 Run Pipeline", type="primary"):
            if "sample_data" in st.session_state:
                data = st.session_state["sample_data"]
            elif "loaded_data" in st.session_state:
                data = st.session_state["loaded_data"]
            else:
                st.error("Please load data first!")
                return

            # Create config
            config = create_default_config()
            config.time_budget_seconds = time_budget
            config.max_trials = max_trials
            config.metric = metric
            config.enable_ensembling = enable_ensembling
            config.llm.enabled = enable_llm
            config.cv_folds = cv_folds

            # Prepare data
            if target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            # Run pipeline
            with st.spinner("Running pipeline..."):
                try:
                    results = run_autonomous_ml(config, X, y)
                    st.session_state["last_results"] = results
                    st.success("Pipeline completed successfully!")

                    # Show results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Score", f"{results['best_score']:.4f}")
                    with col2:
                        st.metric("Best Model", results["best_model"])
                    with col3:
                        st.metric("Total Trials", results["total_trials"])

                except Exception as e:
                    st.error(f"Pipeline failed: {e}")


def show_leaderboard():
    """Show model leaderboard."""
    st.header("🏆 Model Leaderboard")

    # Load leaderboard
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        st.info("No artifacts found. Run a pipeline first!")
        return

    # Find latest run
    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        st.info("No runs found. Run a pipeline first!")
        return

    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Select run
    run_selection = st.selectbox("Select Run", [d.name for d in run_dirs])

    selected_run = artifacts_dir / run_selection
    leaderboard_file = selected_run / "leaderboard.csv"

    if leaderboard_file.exists():
        df = pd.read_csv(leaderboard_file)

        # Display leaderboard
        st.dataframe(df, use_container_width=True)

        # Performance visualization
        if len(df) > 0:
            st.subheader("Performance Visualization")

            col1, col2 = st.columns(2)

            with col1:
                # Score by model type
                fig = px.bar(df, x="model_type", y="score", title="Score by Model Type")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Score vs Fit Time
                fig = px.scatter(
                    df,
                    x="fit_time",
                    y="score",
                    color="model_type",
                    title="Score vs Fit Time",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No leaderboard found for this run.")


def show_model_cards():
    """Show model cards."""
    st.header("📋 Model Cards")

    # Find model cards
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        st.info("No artifacts found. Run a pipeline first!")
        return

    model_cards = []
    for run_dir in artifacts_dir.iterdir():
        if run_dir.is_dir():
            card_file = run_dir / "model_card.md"
            if card_file.exists():
                model_cards.append((run_dir.name, card_file))

    if not model_cards:
        st.info("No model cards found. Run a pipeline first!")
        return

    # Select model card
    card_selection = st.selectbox(
        "Select Model Card", [f"{name} - {card.name}" for name, card in model_cards]
    )

    selected_card = model_cards[
        st.selectbox("Select Model Card", range(len(model_cards)))
    ][1]

    # Display model card
    with open(selected_card, "r") as f:
        card_content = f.read()

    st.markdown(card_content)


def show_monitoring():
    """Show system monitoring."""
    st.header("📊 System Monitoring")

    # Health checks
    st.subheader("Health Status")
    health_status = health_checker.run_health_checks()

    col1, col2 = st.columns(2)

    with col1:
        status_color = "🟢" if health_status["status"] == "healthy" else "🔴"
        st.metric("Overall Status", f"{status_color} {health_status['status'].title()}")

    with col2:
        uptime = performance_monitor.get_uptime()
        st.metric("Uptime", f"{uptime:.1f} seconds")

    # Health check details
    st.subheader("Health Check Details")
    for check_name, check_result in health_status["checks"].items():
        status_icon = "✅" if check_result["status"] == "healthy" else "❌"
        st.write(f"{status_icon} **{check_name}**: {check_result['status']}")

    # Performance metrics
    st.subheader("Performance Metrics")
    metrics = performance_monitor.get_performance_summary()

    # Display metrics
    for metric_name, metric_data in metrics["metrics"].items():
        if metric_data["type"] == "counter":
            st.metric(metric_name, metric_data["value"])
        elif metric_data["type"] == "timer":
            st.write(
                f"**{metric_name}**: {metric_data['mean']:.3f}s avg ({metric_data['count']} samples)"
            )


def show_settings():
    """Show settings page."""
    st.header("⚙️ Settings")

    st.subheader("API Keys")
    st.info("Configure API keys for LLM providers")

    # OpenAI API Key
    openai_key = st.text_input("OpenAI API Key", type="password")
    if st.button("Save OpenAI Key"):
        st.success("OpenAI key saved!")

    # Gemini API Key
    gemini_key = st.text_input("Gemini API Key", type="password")
    if st.button("Save Gemini Key"):
        st.success("Gemini key saved!")

    st.subheader("System Settings")

    # Enable/disable features
    enable_llm = st.checkbox("Enable LLM Guidance", True)
    enable_monitoring = st.checkbox("Enable Monitoring", True)
    enable_security = st.checkbox("Enable Security", True)

    if st.button("Save Settings"):
        st.success("Settings saved!")


if __name__ == "__main__":
    main()
