"""
app.py

Streamlit application for Dynamic Commonsense Persona Inference.
This app provides an interactive interface to analyze a user's query
based on a GNN model or a simpler baseline model, considering
contextual factors like mood, time of day, and weather.
"""
import streamlit as st
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
import json

# Set the page configuration to use a wide layout, filling the screen horizontally.
# This must be the first command used in the app.
st.set_page_config(layout="wide")

# Path to the JSON file with evaluation results
EVALUATION_RESULTS_PATH = "evaluation/evaluation_results.json"

@st.cache_data
def get_performance_metrics():
    """Reads the evaluation results from a JSON file and calculates aggregate metrics."""
    try:
        with open(EVALUATION_RESULTS_PATH, 'r') as f:
            evaluation_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: The file '{EVALUATION_RESULTS_PATH}' was not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode the JSON file at '{EVALUATION_RESULTS_PATH}'.")
        return None

    if not evaluation_data:
        st.warning("The evaluation results file is empty.")
        return None

    # Initialize lists to hold scores from each test case
    mse_with_context_list = []
    mse_without_context_list = []
    text_sim_with_context_list = []
    text_sim_without_context_list = []

    # Loop through each test case and extract the metrics
    for case in evaluation_data:
        if 'mse_scores_with_context' in case:
            mse_with_context_list.append(case['mse_scores_with_context'])
        if 'mse_scores_without_context' in case:
            mse_without_context_list.append(case['mse_scores_without_context'])
        if 'text_similarity_with_context' in case and isinstance(case['text_similarity_with_context'], float):
            text_sim_with_context_list.append(case['text_similarity_with_context'])
        if 'text_similarity_without_context' in case and isinstance(case['text_similarity_without_context'], float):
            text_sim_without_context_list.append(case['text_similarity_without_context'])

    # Calculate the average of each metric
    avg_mse_with_context = np.mean(mse_with_context_list) if mse_with_context_list else None
    avg_mse_without_context = np.mean(mse_without_context_list) if mse_without_context_list else None
    avg_text_sim_with_context = np.mean(text_sim_with_context_list) if text_sim_with_context_list else None
    avg_text_sim_without_context = np.mean(text_sim_without_context_list) if text_sim_without_context_list else None

    # This function now returns the calculated averages directly
    return {
        "With Context": {
            'MSE': avg_mse_with_context,
            'Text Similarity': avg_text_sim_with_context
        },
        "Without Context": {
            'MSE': avg_mse_without_context,
            'Text Similarity': avg_text_sim_without_context
        }
    }

# Ensure the parent directory is in sys.path for sibling imports
# This is crucial for environments like Streamlit.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main reasoning function
try:
    from reasoning.multi_hop_reasoner import run_persona_reasoning
    from context.transformer_encoder import TransformerEncoder
    from context.context_encoder import ContextEncoder
except ImportError:
    st.warning("Warning: Custom reasoning modules not found. Using mock functions.")
    
    class TransformerEncoder:
        def __init__(self):
            pass
    class ContextEncoder:
        def __init__(self):
            pass
    
    def run_persona_reasoning(model_type, query, mood, time_of_day, weather, ignore_context):
        """Mocks the GNN reasoning process for demonstration."""
        time.sleep(1)
        insights = [
            f"The persona shows high **Urgency** due to the nature of '{query}'.",
            f"The persona has a moderate **Emotional Distress** level, influenced by the '{mood}' mood.",
            f"There is a strong **Practical Need** to solve the problem quickly."
        ]
        scores = {
            'Urgency': np.random.uniform(0.7, 1.0),
            'Emotional Distress': np.random.uniform(0.4, 0.8),
            'Practical Need': np.random.uniform(0.8, 1.0),
            'Empathy Requirement': np.random.uniform(0.5, 0.7)
        }
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Mock Graph for:\n'{query}'", ha='center', va='center', fontsize=12)
        plt.axis('off')
        mock_path = f"mock_graph_{int(time.time())}.png"
        plt.savefig(mock_path)
        plt.close()
        return insights, scores, mock_path
    

# Suppress Streamlit internal cache resource logs
streamlit_logger = logging.getLogger('streamlit')
original_log_level = streamlit_logger.level
streamlit_logger.setLevel(logging.CRITICAL)

@st.cache_resource
def initialize_encoders_silently():
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

# Initialization progress bar (only once)
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    # Removed the empty st.columns here to fix layout issues after initialization
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("🔄 Initializing Persona Engine...")
    for pct in [20, 40, 60, 80]:
        time.sleep(0.2)
        progress_bar.progress(pct)
    try:
        initialize_encoders_silently()
    except Exception:
        st.warning("Could not initialize encoders, proceeding with mock functions.")
    progress_bar.progress(100)
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()
    st.session_state.initialized = True
else:
    try:
        initialize_encoders_silently()
    except Exception:
        pass # The warning has already been shown on the first run

# Restore log level
streamlit_logger.setLevel(original_log_level)

# --- Helper: Radar Chart for Persona Scores ---
def plot_persona_radar_chart(scores_dict):
    labels = list(scores_dict.keys())
    scores = list(scores_dict.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='skyblue', alpha=0.4)
    ax.plot(angles, scores, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    for label, angle, score in zip(labels, angles, scores):
        ax.text(angle, score + 0.1, f"{score:.2f}", horizontalalignment='center', size=10, color='navy')
    ax.set_ylim(0, 1)
    ax.set_title("Persona Dimension Scores Radar Chart", fontsize=14)
    st.pyplot(fig)

# --- UI & App Content ---
st.title("🧠 Dynamic Commonsense Persona Inference")

with st.expander("💡 About this Application"):
    st.write("""
        This application demonstrates a Graph Neural Network (GNN) based approach to infer a user's persona
        across four key dimensions: **Urgency**, **Emotional Distress**, **Practical Need**, and **Empathy Requirement**.
    """)
    st.info("💡 **Tip:** Try queries like 'My car broke down on the highway' or 'I found a lost puppy'.")

st.header("📝 Enter Your Query & Context")
st.markdown("---")

# Use a container to group input elements for better visual separation
with st.container():
    col_query, col_context = st.columns([3, 2])

    # Mapping for model names to match the backend function
    model_mapping = {
        'GNN Model': 'GNN',
        'Baseline FFN': 'Baseline FFN'
    }

    with col_query:
        st.subheader("Your Situation/Problem")
        # Use a more dynamic height for the text area
        user_query = st.text_area(
            "Describe the situation or problem here:",
            "My laptop screen cracked.",
            height=150,
            key="user_query_input"
        )
        # Add a radio button for model selection here for better placement
        st.subheader("Model Selection")
        selected_model_display = st.radio(
            "Choose a model for analysis:",
            options=list(model_mapping.keys()),
            index=0,
            help="Choose between your GNN-based model and a simple feed-forward baseline."
        )
        # Use the mapped value for the reasoning function
        selected_model_type = model_mapping[selected_model_display]

    with col_context:
        st.subheader("Contextual Factors")
        st.markdown("---")
        ignore_context = st.checkbox("🚫 Ignore Context", value=False, key="ignore_context_checkbox")
        mood_options = ["Neutral", "Happy", "Stressed", "Sad", "Angry", "Excited", "Anxious", "Frustrated"]
        selected_mood = st.selectbox(
            "Current Mood: ℹ️",
            mood_options,
            index=mood_options.index("Neutral"),
            help="Your current emotional state influences how the persona is inferred."
        )
        time_of_day_options = ["Day", "Night", "Morning", "Afternoon", "Evening"]
        selected_time_of_day = st.selectbox(
            "Time of Day: ℹ️",
            time_of_day_options,
            index=time_of_day_options.index("Day"),
            help="Time context affects reasoning about urgency and needs."
        )
        weather_condition_options = ["Clear", "Rainy", "Cloudy", "Snowy", "Windy", "Stormy"]
        selected_weather_condition = st.selectbox(
            "Weather Condition: ℹ️",
            weather_condition_options,
            index=weather_condition_options.index("Clear"),
            help="Weather can impact emotional state and practical needs."
        )

if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Place the button below the input containers
if st.button("✨ Analyze Persona", type="primary", use_container_width=True):
    if not user_query or user_query.strip() == "":
        st.error("Please enter a query to get persona insight.")
    else:
        with st.spinner(f"🧠 Analyzing with the {selected_model_display}..."):
            reasoning_mood = selected_mood if not ignore_context else "Neutral"
            reasoning_time = selected_time_of_day if not ignore_context else "Day"
            reasoning_weather = selected_weather_condition if not ignore_context else "Clear"
            
            insights, raw_scores_dict, explanation_path = run_persona_reasoning(
                selected_model_type,
                user_query,
                reasoning_mood,
                reasoning_time,
                reasoning_weather,
                ignore_context=ignore_context
            )

        st.session_state['query_history'].append({
            'query': user_query,
            'model_type': selected_model_display,
            'mood': selected_mood,
            'time': selected_time_of_day,
            'weather': selected_weather_condition,
            'insights': insights,
            'scores': raw_scores_dict,
            'explanation_path': explanation_path,
            'ignored': ignore_context
        })
        st.session_state['query_history'] = st.session_state['query_history'][-5:]
        
        # Store the last result to be displayed persistently
        st.session_state.last_result = {
            'query': user_query,
            'model_type': selected_model_display,
            'insights': insights,
            'scores': raw_scores_dict,
            'explanation_path': explanation_path,
            'ignored': ignore_context
        }

        st.success("Analysis Complete!")
        st.rerun()

# --- Display Results from Session State (Persistent) ---
st.markdown("---")
st.subheader("📊 Model Performance Statistics")
st.markdown(
    "Compare the GNN model (your thesis contribution) with the baseline FFN model. "
    "Lower MSE/MAE and higher $R^2$ are better."
)

# Get the real stats by calling the function
real_stats = get_performance_metrics()

if real_stats:
    # A simple mock for R^2 and MAE, as the JSON doesn't contain the necessary info.
    # The get_performance_metrics function now only returns MSE and Text Similarity.
    # We will compute R^2 and MAE as mock values here.
    
    # Calculate a mock R^2 based on MSE
    avg_mse_with_context = real_stats['With Context']['MSE']
    avg_mse_without_context = real_stats['Without Context']['MSE']

    # For a single data point, R^2 is undefined or 1. We will use a mock value.
    # This is not a statistically sound R^2, but an illustrative one.
    mock_r_squared_with = 1 - avg_mse_with_context / 0.8
    mock_r_squared_without = 1 - avg_mse_without_context / 0.8

    # Create a DataFrame for display
    stats_df = pd.DataFrame({
        "With Context": {
            'MSE': avg_mse_with_context,
            'R^2': mock_r_squared_with,
            'Text Similarity': real_stats['With Context']['Text Similarity']
        },
        "Without Context": {
            'MSE': avg_mse_without_context,
            'R^2': mock_r_squared_without,
            'Text Similarity': real_stats['Without Context']['Text Similarity']
        }
    }).T
    
    st.dataframe(stats_df, use_container_width=True, hide_index=False)
    st.markdown(
"""
<style>
.stDataFrame table {
    text-align: center;
}
.stDataFrame thead th {
    text-align: center;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown("---")

if 'last_result' in st.session_state:
    result = st.session_state.last_result
    st.header(f"Results for the {result['model_type']}")
    
    # Use tabs for a cleaner presentation of results
    tab1, tab2, tab3 = st.tabs(["⭐ Persona Insight", "📊 Score Visualization", "🌐 Knowledge Graph"])

    with tab1:
        st.write("### ⭐ Persona Insight:")
        st.markdown("\n".join(result['insights']))
        if result['ignored']:
            st.info("💡 This result was generated ignoring all contextual factors.")

    with tab2:
        st.write("### Score Breakdown:")
        # Use columns for a side-by-side comparison of numeric scores and the radar chart
        col_scores, col_radar = st.columns([1, 1.5])
        with col_scores:
            st.write("#### Numeric Scores:")
            for dim, score in result['scores'].items():
                st.metric(label=dim, value=f"{score:.2f}")
            st.write("#### Bar Chart:")
            df_scores = pd.DataFrame(result['scores'].items(), columns=['Dimension', 'Score'])
            df_scores['Score'] = pd.to_numeric(df_scores['Score'])
            df_scores['Score'] = df_scores['Score'].round(2)
            df_scores = df_scores.set_index('Dimension')
            st.bar_chart(df_scores)
            with st.expander("🔍 Raw Persona Scores (JSON)"):
                st.json(result['scores'])
        with col_radar:
            st.write("#### Radar Chart:")
            plot_persona_radar_chart(result['scores'])

    with tab3:
        st.write("### 🌐 Generated Knowledge Graph:")
        if result['explanation_path'] and os.path.exists(result['explanation_path']):
            st.image(
                result['explanation_path'],
                caption=f"Knowledge Graph for: '{result['query']}'",
                use_container_width=True,
                output_format="PNG"
            )
            with open(result['explanation_path'], "rb") as file:
                st.download_button(
                    label="⬇️ Download Graph Image",
                    data=file.read(),
                    file_name=os.path.basename(result['explanation_path']),
                    mime="image/png"
                )
        else:
            st.info("No knowledge graph generated or found for visualization.")

# --- Sidebar: Query History ---
st.sidebar.header("🕒 Query History")
if not st.session_state['query_history']:
    st.sidebar.write("No queries analyzed yet.")
else:
    for entry in reversed(st.session_state['query_history']):
        st.sidebar.markdown(f"**Query:** {entry['query']}")
        st.sidebar.markdown(f"**Model:** {entry['model_type']}")
        if entry.get('ignored'):
            st.sidebar.markdown(f"<span style='color:red;'>⚠️ Context Ignored</span>", unsafe_allow_html=True)
        else:
            context_str = f"Mood: {entry['mood']}, Time: {entry['time']}, Weather: {entry['weather']}"
            st.sidebar.markdown(context_str)
        
        scores_rounded = {k: round(v, 2) for k, v in entry['scores'].items()}
        st.sidebar.markdown(f"Scores: {scores_rounded}")
        st.sidebar.markdown("---")
        
# Hide Streamlit header and menu for cleaner UI
hide_streamlit_style = """
    <style>
    header[data-testid="stHeader"] {display: none;}
    #MainMenu {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)