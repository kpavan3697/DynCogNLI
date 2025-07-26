# app.py

import streamlit as st
import os
import sys
import torch
import pandas as pd
import io
import logging
import time  # <-- Needed for progress bar simulation

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Dynamic Commonsense Persona Inference",
    page_icon="🧠",
    layout="wide",  # Use wide layout for more space
)

# Add parent directory to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules after sys.path setup
from reasoning.multi_hop_reasoner import run_gnn_reasoning
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder

# --- Suppress Streamlit internal cache resource logs ---
streamlit_logger = logging.getLogger('streamlit')
original_log_level = streamlit_logger.level
streamlit_logger.setLevel(logging.CRITICAL)

@st.cache_resource
def initialize_encoders_silently():
    """
    Initializes and caches the Transformer and Context Encoders.
    Temporarily redirects stdout and stderr to suppress any verbose logs.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
        _ = TransformerEncoder()
        _ = ContextEncoder()
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    return True


# --- Initialization Progress Bar Shown ONLY ONCE ---

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        status_text = st.empty()
        progress_bar = st.progress(0)

        status_text.text("🔄 Initializing Persona Engine...")

        for pct in [20, 40, 60, 80]:
            time.sleep(0.2)
            progress_bar.progress(pct)

        initialize_encoders_silently()

        progress_bar.progress(100)
        time.sleep(0.3)

        progress_bar.empty()
        status_text.empty()

    st.session_state.initialized = True
else:
    # On subsequent runs (e.g., button clicks), just silently call cached function (fast, no UI)
    initialize_encoders_silently()

# Restore original log level
streamlit_logger.setLevel(original_log_level)

# --- UI & App Content ---

st.title("🧠 Dynamic Commonsense Persona Inference")

with st.expander("💡 About this Application"):
    st.write("""
        This application demonstrates a Graph Neural Network (GNN) based approach to infer a user's persona
        across four key dimensions: **Urgency**, **Emotional Distress**, **Practical Need**, and **Empathy Requirement**.

        The system works by:
        1.  **Query & Context Encoding**: Your input query and selected mood, time of day, and weather are converted into numerical embeddings.
        2.  **Knowledge Graph Generation**: A relevant subgraph is extracted from ConceptNet, a large commonsense knowledge base, based on your query.
        3.  **GNN Reasoning**: The GNN processes the knowledge graph, enriched with query and context embeddings, to predict the persona scores.
        4.  **Persona Interpretation**: The scores are interpreted into human-readable insights and a recommended approach.

        You can choose to **Ignore Context** to see the baseline persona derived purely from the query and commonsense graph.
    """)
    st.info("💡 **Tip:** Try queries like 'My car broke down on the highway' or 'I found a lost puppy'.")

st.header("📝 Enter Your Query & Context")

col_query, col_context = st.columns([3, 2])

with col_query:
    st.subheader("Your Situation/Problem")
    user_query = st.text_area(
        "Describe the situation or problem here:",
        "My laptop screen cracked.",
        height=120,
        key="user_query_input"
    )

with col_context:
    st.subheader("Contextual Factors")
    ignore_context = st.checkbox("🚫 Ignore Context (for baseline comparison)", value=False, key="ignore_context_checkbox")

    with st.container():
        st.write("**:mag_right: Environment & Mood**")
        mood_options = ["Neutral", "Happy", "Stressed", "Sad", "Angry", "Excited", "Anxious", "Frustrated"]
        selected_mood = st.selectbox("Current Mood:", mood_options, index=mood_options.index("Neutral"))

        time_of_day_options = ["Day", "Night", "Morning", "Afternoon", "Evening"]
        selected_time_of_day = st.selectbox("Time of Day:", time_of_day_options, index=time_of_day_options.index("Day"))

        weather_condition_options = ["Clear", "Rainy", "Cloudy", "Snowy", "Windy", "Stormy"]
        selected_weather_condition = st.selectbox("Weather Condition:", weather_condition_options, index=weather_condition_options.index("Clear"))

# Hide Streamlit header and menu
hide_streamlit_style = """
    <style>
    header[data-testid="stHeader"] {display: none;}

    .st-emotion-cache-zy6yx3 {
    padding-left: 5rem;
    padding-right: 5rem;
    padding-top: 1rem;
}

    #MainMenu {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Analyze Button ---
if st.button("✨ Analyze Persona", type="primary", use_container_width=True):
    if not user_query:
        st.error("Please enter a query to get persona insight.")
    else:
        with st.spinner("🧠 Analyzing query, building graph, and inferring persona..."):
            insights, raw_scores_dict, explanation_path = run_gnn_reasoning(
                user_query,
                selected_mood,
                selected_time_of_day,
                selected_weather_condition,
                ignore_context=ignore_context
            )

            st.success("Analysis Complete!")
            st.markdown("---")

            st.subheader("📊 Persona Analysis Results")
            results_col1, results_col2 = st.columns([1, 1])

            with results_col1:
                st.write("### ⭐ Persona Insight:")
                st.markdown("\n".join(insights))

                st.write("### Score Breakdown:")
                df_scores = pd.DataFrame(raw_scores_dict.items(), columns=['Dimension', 'Score'])
                df_scores['Score'] = df_scores['Score'].round(2)
                df_scores = df_scores.set_index('Dimension')
                st.bar_chart(df_scores)

                with st.expander("🔍 Raw Persona Scores (JSON)", expanded=True):
                    st.json(raw_scores_dict)

            with results_col2:
                st.write("### 🌐 Generated Knowledge Graph:")
                if explanation_path and os.path.exists(explanation_path):
                    st.image(
                        explanation_path,
                        caption=f"Knowledge Graph for: '{user_query}'",
                        use_container_width=True,
                        output_format="PNG"
                    )
                    st.download_button(
                        label="Download Graph Image",
                        data=open(explanation_path, "rb").read(),
                        file_name=os.path.basename(explanation_path),
                        mime="image/png"
                    )
                else:
                    st.info("No knowledge graph generated or found for visualization.")

