import streamlit as st
import os
import sys
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Dynamic Commonsense Persona Inference",
    page_icon="🧠",
    layout="wide",
)

# Add parent directory to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking modules for demonstration if they don't exist
try:
    from reasoning.multi_hop_reasoner import run_gnn_reasoning
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
    
    def run_gnn_reasoning(query, mood, time_of_day, weather, ignore_context):
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
        mock_path = "mock_graph.png"
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
    initialize_encoders_silently()

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

if st.button("✨ Analyze Persona", type="primary", use_container_width=True):
    if not user_query or user_query.strip() == "":
        st.error("Please enter a query to get persona insight.")
    else:
        with st.spinner("🧠 Analyzing query, building graph, and inferring persona..."):
            reasoning_mood = selected_mood if not ignore_context else "Neutral"
            reasoning_time = selected_time_of_day if not ignore_context else "Day"
            reasoning_weather = selected_weather_condition if not ignore_context else "Clear"
            
            insights, raw_scores_dict, explanation_path = run_gnn_reasoning(
                user_query,
                reasoning_mood,
                reasoning_time,
                reasoning_weather,
                ignore_context=ignore_context
            )

        st.session_state['query_history'].append({
            'query': user_query,
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
            'insights': insights,
            'scores': raw_scores_dict,
            'explanation_path': explanation_path,
            'ignored': ignore_context
        }

        st.success("Analysis Complete!")
        st.rerun()

# --- Display Results from Session State (Persistent) ---
if 'last_result' in st.session_state:
    result = st.session_state.last_result
    st.markdown("---")
    st.subheader("📊 Persona Analysis Results")
    
    tab1, tab2, tab3 = st.tabs(["⭐ Persona Insight", "📊 Score Visualization", "🌐 Knowledge Graph"])

    with tab1:
        st.write("### ⭐ Persona Insight:")
        st.markdown("\n".join(result['insights']))
        if result['ignored']:
            st.info("💡 This result was generated ignoring all contextual factors.")

    with tab2:
        st.write("### Score Breakdown:")
        col_scores, col_radar = st.columns(2)
        with col_scores:
            st.write("#### Numeric Scores:")
            for dim, score in result['scores'].items():
                st.metric(label=dim, value=f"{score:.2f}")
            st.write("#### Bar Chart:")
            df_scores = pd.DataFrame(result['scores'].items(), columns=['Dimension', 'Score'])
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
                caption=f"Knowledge Graph for: '{user_query}'",
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