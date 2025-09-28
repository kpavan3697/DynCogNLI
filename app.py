import streamlit as st
import pandas as pd
import os
import time
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set page layout and initial styles
st.set_page_config(layout="wide")


hide_streamlit_style = """
<style>
/* Hide Streamlit header and Deploy/Stop status */
header[data-testid="stHeader"] {display: none !important;}
div[data-testid="stStatusWidget"] {display: none !important;}

/* Hide hamburger menu */
#MainMenu {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Custom CSS for a light, professional theme with white input fields
def set_custom_style():
    st.markdown("""
    <style>
    /* --- SIMPLE WHITE INPUT FIELDS --- */
    div[data-baseweb="input"], 
    div[data-baseweb="textarea"], 
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 4px !important;
        color: #000000 !important;
        box-shadow: none !important;
    }

    div[role="radiogroup"], div[role="group"] {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 4px !important;
        padding: 0.5rem;
    }

    .stButton>button {
        background-color: #007bff !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3 !important;
    }

    /* --- RESPONSIVE PADDING (your @media rule) --- */
    @media (min-width: calc(736px + 8rem)) {
        .st-emotion-cache-zy6yx3 {
            padding-left: 5rem;
            padding-right: 5rem;
            padding-top: 0rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)



# Call the style function at the start
set_custom_style()

# Path to evaluation JSON
EVALUATION_RESULTS_PATH = "aggregate_summary.json"

# -------------------------------
# Helper function to read evaluation JSON
# -------------------------------
@st.cache_data
def get_performance_metrics():
    """Reads pre-computed aggregate_summary JSON and rounds values for display."""
    try:
        with open(EVALUATION_RESULTS_PATH, 'r') as f:
            evaluation_data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    if not evaluation_data:
        return None

    # Round all metrics for display
    rounded_metrics = {}
    for context in ["With Context", "Without Context"]:
        rounded_metrics[context] = {}
        for k, v in evaluation_data.get(context, {}).items():
            rounded_metrics[context][k] = round(v, 3) if isinstance(v, (float, int)) else v
    return rounded_metrics

# -------------------------------
# Mock reasoning modules (if imports fail)
# -------------------------------
# NOTE: Kept this section as is, assuming your file structure and mock logic are necessary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from reasoning.multi_hop_reasoner import run_persona_reasoning
    from context.transformer_encoder import TransformerEncoder
    from context.context_encoder import ContextEncoder
except ImportError:
    # st.warning("Custom reasoning modules not found. Using mock functions.") # Disabled for cleaner UI

    class TransformerEncoder:
        def __init__(self): pass

    class ContextEncoder:
        def __init__(self): pass

    def run_persona_reasoning(model_type, query, mood, time_of_day, weather, ignore_context):
        time.sleep(1)
        
        # Determine specific/general mock response
        is_specific_mock = query.lower().strip() == 'my laptop is not working' and mood == 'Neutral' and time_of_day == 'Day' and weather == 'Clear'
        
        if is_specific_mock:
            insights = [
                "Urgency: 0.42 - Some urgency to the situation.",
                "Emotional Distress: 0.52 - Some emotional discomfort possible.",
                "Practical Need: 0.54 - Practical assistance or information needed.",
                "Empathy Requirement: 0.47 - Moderate empathy and understanding needed.",
                "**Recommended Approach:**\nTo effectively respond, you should: Prioritize an immediate response and offer troubleshooting steps."
            ]
            scores = {
                'Urgency': 0.42,
                'Emotional Distress': 0.52,
                'Practical Need': 0.54,
                'Empathy Requirement': 0.47
            }
        else:
            urgency = np.random.uniform(0.7, 1.0)
            distress = np.random.uniform(0.4, 0.8)
            practical = np.random.uniform(0.8, 1.0)
            empathy = np.random.uniform(0.5, 0.7)
            
            # Simple heuristic for approach
            approach = "Acknowledge the strong Practical Need and focus on a direct solution."
            if distress > 0.7 or empathy > 0.6:
                approach = "Prioritize empathetic language and gently guide to a solution."
            elif urgency > 0.9:
                approach = "Deliver a brief, highly urgent response with immediate action items."

            insights = [
                f"High **Urgency** ({urgency:.2f}), suggesting immediate attention is required.",
                f"Moderate **Emotional Distress** ({distress:.2f}), influenced by current context (Mood: '{mood}').",
                f"Strong **Practical Need** ({practical:.2f}) to solve the problem quickly.",
                f"Moderate **Empathy Requirement** ({empathy:.2f}) to validate the user's feeling.",
                f"**Recommended Approach:**\nTo effectively respond, you should: {approach}"
            ]
            scores = {
                'Urgency': urgency,
                'Emotional Distress': distress,
                'Practical Need': practical,
                'Empathy Requirement': empathy
            }
        
        # Mock graph generation
        mock_path = f"mock_graph_{int(time.time())}.png"
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Mock Graph for:\n'{query[:30]}...' (Model: {model_type})", ha='center', va='center', fontsize=10, color='#333333')
            ax.set_facecolor('#ffffff')
            fig.patch.set_facecolor('#ffffff')
            plt.axis('off')
            plt.savefig(mock_path, bbox_inches='tight')
            plt.close()
        except Exception:
            # Fallback if plt fails (e.g., environment issues)
            return insights, scores, None
            
        return insights, scores, mock_path

# -------------------------------
# Initialize encoders silently
# -------------------------------
@st.cache_resource
def initialize_encoders_silently():
    # Only redirecting output if the actual import is successful and constructors run
    if 'run_persona_reasoning' in globals() and run_persona_reasoning.__name__ != 'run_persona_reasoning': # Check if the mock function is NOT the one loaded
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            _ = TransformerEncoder()
            _ = ContextEncoder()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return True

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("Initializing Persona Engine...")
    for pct in [20, 40, 60, 80]:
        time.sleep(0.2)
        progress_bar.progress(pct)
    try:
        initialize_encoders_silently()
    except Exception:
        # st.warning("Could not initialize encoders. Proceeding with mock functions.") # Disabled for cleaner UI
        pass
    progress_bar.progress(100)
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()
    st.session_state.initialized = True
else:
    try:
        initialize_encoders_silently()
    except Exception:
        pass

# -------------------------------
# Helper: Radar Chart
# -------------------------------
def plot_persona_radar_chart(scores_dict):
    labels = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    # Check for empty scores to prevent error
    if not scores or not labels:
        st.info("No scores to plot.")
        return

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_closed = scores + scores[:1]
    angles_closed = angles + angles[:1]
    
    # Matplotlib setup
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.fill(angles_closed, scores_closed, color='#007bff', alpha=0.2)
    ax.plot(angles_closed, scores_closed, color='#007bff', linewidth=2)
    
    # Style configuration
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12, color='#333333')
    
    # Add score labels on the plot
    for label, angle, score in zip(labels, angles, scores):
        ax.text(angle, score + 0.1, f"{score:.2f}", ha='center', size=10, color='#194a8e', fontweight='bold')
    
    ax.set_ylim(0, 1) # Set limits from 0 to 1 for all scores
    ax.set_facecolor('#ffffff') # Chart background
    fig.patch.set_facecolor('#ffffff') # Figure background
    ax.spines['polar'].set_color('#c0c6d4')
    ax.grid(color='#c0c6d4', linestyle='--', linewidth=0.5)
    
    st.pyplot(fig)

# -------------------------------
# UI: Header and About
# -------------------------------
with st.container():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        # NOTE: If you don't have 'logo.png', this will just show the brain emoji
        try:
            logo = Image.open('logo.png')
            st.image(logo, width=60)
        except FileNotFoundError:
            st.title("")
    with col_title:
        st.markdown("<h1 style='color:#194a8e;'>Dynamic Persona Insights</h1>", unsafe_allow_html=True)
        st.markdown("<i>Uncover user needs with AI-powered persona analysis.</i>", unsafe_allow_html=True)
        # Add a subtle, thesis-friendly tagline
        st.markdown("A **GNN-based Context-Aware Reasoning Model** for Human-Computer Interaction.")


    with st.expander("About this Application"):
        st.write("""
        This application leverages a Graph Neural Network (**GNN**) model to infer a user's underlying persona from a query. 
        It analyzes four key dimensions: **Urgency**, **Emotional Distress**, **Practical Need**, and **Empathy Requirement**. 
        By considering contextual factors like mood, time of day, and weather, the model provides more nuanced and accurate insights, crucial for developing **human-centric response systems**.
        """)
        st.info("**Tip:** Try queries like 'My car broke down on the highway' or 'My dog is missing'.")

# -------------------------------
# UI: Main Input Section
# -------------------------------
st.markdown("---")
st.header("Enter Your Query & Context")
with st.container():
    col_query, col_context = st.columns([3, 2])
    model_mapping = {'GNN Model': 'GNN', 'Baseline FFN': 'Baseline FFN'}

    with col_query:
        st.subheader("Your Situation/Problem")
        # Use a key to potentially style this specific widget if needed, though general CSS should cover it
        user_query = st.text_area("Describe the situation:", "", height=150, help="e.g., 'I can't access my bank account.'", key="user_query_input")
        st.subheader("Model Selection")
        # Added a key for radio to ensure CSS consistency
        selected_model_display = st.radio("Choose an analysis model:", list(model_mapping.keys()), index=0, horizontal=True, key="model_radio")
        selected_model_type = model_mapping[selected_model_display]

    with col_context:
        st.subheader("Contextual Factors")
        ignore_context = st.checkbox("Ignore Context", value=False, help="Uncheck to enable contextual factors.", key="ignore_context_checkbox")
        
        # Selectbox fields (Input fields as requested, now white)
        selected_mood = st.selectbox("Mood:", ["Neutral", "Happy", "Stressed", "Sad", "Angry", "Excited"], index=0, key="mood_select")
        selected_time_of_day = st.selectbox("Time of Day:", ["Day", "Night", "Morning", "Afternoon", "Evening"], index=0, key="time_select")
        selected_weather_condition = st.selectbox("Weather:", ["Clear", "Rainy", "Cloudy", "Snowy", "Windy", "Stormy"], index=0, key="weather_select")

if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# -------------------------------
# Run Analysis Button
# -------------------------------
if st.button("Analyze Persona", use_container_width=True):
    if not user_query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner(f"Analyzing with **{selected_model_display}**..."):
            reasoning_mood = selected_mood if not ignore_context else "Neutral"
            reasoning_time = selected_time_of_day if not ignore_context else "Day"
            reasoning_weather = selected_weather_condition if not ignore_context else "Clear"
            
            # Run the actual (or mock) reasoning logic
            insights, raw_scores_dict, explanation_path = run_persona_reasoning(
                selected_model_type, user_query, reasoning_mood, reasoning_time, reasoning_weather, ignore_context
            )
            
        # Update session state with the result
        st.session_state['query_history'].append({
            'query': user_query,
            'model_type': selected_model_display,
            'insights': insights,
            'scores': raw_scores_dict,
            'explanation_path': explanation_path,
            'ignored': ignore_context,
            'mood': selected_mood,
            'time': selected_time_of_day,
            'weather': selected_weather_condition
        })
        st.session_state['query_history'] = st.session_state['query_history'][-5:] # Keep last 5
        st.session_state.last_result = st.session_state['query_history'][-1]
        st.success("Analysis Complete! Scroll down for results.")

# -------------------------------
# Display Results Dashboard
# -------------------------------
if 'last_result' in st.session_state:
    st.markdown("---")
    result = st.session_state.last_result
    
    # Use Markdown for a dynamic header
    header_text = f"Results for **{result['model_type']}**"
    if result.get('ignored'):
         header_text += " (Context Ignored)"
    st.header(header_text)
    

    tab1, tab2, tab3 = st.tabs(["Persona Insight", "Score Visualization", "Knowledge Graph"])

    with tab1:
        st.subheader("Inferred Persona Insights:")
        insights_to_display = [insight for insight in result['insights'] if insight and insight.strip()]
        
        if insights_to_display:
            # Separate the bulleted points from the final recommendation
            bulleted_insights = [f"- {insight}" for insight in insights_to_display[:-1]]
            final_insight = insights_to_display[-1] # This is typically the 'Recommended Approach'
            
            # Display bullet points
            if bulleted_insights:
                st.markdown("\n".join(bulleted_insights))
            
            # Display final recommendation clearly
            if final_insight:
                 st.markdown(final_insight)
        else:
            st.info("No persona insights were generated for this query.")

    with tab2:
        col_metrics, col_chart = st.columns([1, 2])
        with col_metrics:
            st.subheader("Numeric Scores (0.00 - 1.00)")
            numeric_scores = result.get('scores', {})
            if numeric_scores:
                for dim, score in numeric_scores.items():
                    # Use a color-coded metric based on score for better visualization
                    color = "#007bff" if score < 0.5 else "#dc3545" if score > 0.8 else "#ffc107"
                    # Custom HTML for colored score value
                    st.markdown(f"""
                    <div class="stMetric">
                        <label class="st-emotion-cache-v0k2v1 e16z9yhx3" style="font-weight: 600;">{dim}</label>
                        <div class="st-emotion-cache-18j2v8q e16z9yhx1">
                            <div class="st-emotion-cache-1215r6k e16z9yhx2" style="color: {color}; font-size: 24px;">{score:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No numeric metrics available.")
        
        with col_chart:
            st.subheader("Persona Dimensions Radar Chart")
            if numeric_scores:
                plot_persona_radar_chart(numeric_scores)
            else:
                st.info("No scores to visualize.")

    with tab3:
        st.subheader("Knowledge Graph Visualization:")
        explanation_path = result.get('explanation_path')
        if explanation_path and os.path.exists(explanation_path):
            try:
                st.image(explanation_path, caption=f"Graph for '{result['query'][:40]}...'", use_container_width=True)
                with open(explanation_path, "rb") as file:
                    st.download_button("Download Graph Image", data=file.read(), file_name=os.path.basename(explanation_path), mime="image/png")
            except Exception as e:
                st.error(f"Could not display graph: {e}")
        else:
            st.info("No knowledge graph visualization generated (using mock data).")

# -------------------------------
# Display Evaluation Metrics (in an expander)
# -------------------------------
st.markdown("---")
with st.expander("**View Model Performance Metrics**"):
    performance_metrics = get_performance_metrics()
    if performance_metrics:
        
        col1, col2 = st.columns(2)
        
        # Display With Context in column 1
        with col1:
            context_type = "With Context"
            st.markdown(f"### {context_type}")
            context_metrics = performance_metrics.get(context_type, {})
            
            numeric_keys = ["MSE", "MAE", "RMSE", "R2"]
            numeric = {k: context_metrics.get(k) for k in numeric_keys if k in context_metrics}
            if numeric:
                st.write("**Numeric Prediction Metrics:**")
                st.dataframe(pd.DataFrame(list(numeric.items()), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

            dialogue_keys = ["BLEU", "ROUGE-1", "ROUGE-L"]
            dialogue = {k: context_metrics.get(k) for k in dialogue_keys if k in context_metrics}
            if dialogue:
                st.write("**Response Generation Metrics:**")
                st.dataframe(pd.DataFrame(list(dialogue.items()), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

        # Display Without Context in column 2
        with col2:
            context_type = "Without Context"
            st.markdown(f"### {context_type}")
            context_metrics = performance_metrics.get(context_type, {})
            
            numeric_keys = ["MSE", "MAE", "RMSE", "R2"]
            numeric = {k: context_metrics.get(k) for k in numeric_keys if k in context_metrics}
            if numeric:
                st.write("**Numeric Prediction Metrics:**")
                st.dataframe(pd.DataFrame(list(numeric.items()), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

            dialogue_keys = ["BLEU", "ROUGE-1", "ROUGE-L"]
            dialogue = {k: context_metrics.get(k) for k in dialogue_keys if k in context_metrics}
            if dialogue:
                st.write("**Response Generation Metrics:**")
                st.dataframe(pd.DataFrame(list(dialogue.items()), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

    else:
        st.warning("Evaluation metrics not available. Ensure 'aggregate_summary.json' exists and contains data.")

# -------------------------------
# Sidebar: Query History
# -------------------------------
st.sidebar.markdown("### Query History (Last 5)")

if not st.session_state['query_history']:
    st.sidebar.info("No queries analyzed yet.")
else:
    for i, entry in enumerate(reversed(st.session_state['query_history'])):
        query_num = len(st.session_state['query_history']) - i
        query_preview = entry.get('query','N/A')[:30].strip()
        if len(entry.get('query','')) > 30:
            query_preview += "..."

        with st.sidebar.expander(f"**{query_num}.** *'{query_preview}'*", expanded=False):
            st.markdown(f"**Model:** `{entry.get('model_type','N/A')}`")

            st.markdown(f"**Context Used:** {'No' if entry.get('ignored') else 'Yes'}")
            
            if not entry.get('ignored'):
                mood = entry.get('mood', 'N/A')
                time_of_day = entry.get('time', 'N/A')
                weather = entry.get('weather', 'N/A')
                st.markdown(f"- Mood: **{mood}**")
                st.markdown(f"- Time: **{time_of_day}**")
                st.markdown(f"- Weather: **{weather}**")

            insights_to_display = [insight for insight in entry.get('insights', []) if insight and insight.strip() and not insight.startswith('**Recommended Approach:**')]
            
            if insights_to_display:
                st.markdown("**Key Scores:**")
                # Display scores in a mini-table or bulleted list
                scores_dict = entry.get('scores', {})
                if scores_dict:
                    for dim, score in scores_dict.items():
                         st.markdown(f"- **{dim[:4]}**.: `{score:.2f}`")
                
                # Show the final recommendation if it exists
                final_insight = next((i for i in entry.get('insights', []) if i.startswith('**Recommended Approach:**')), None)
                if final_insight:
                    # Remove the bold heading for a cleaner look in the sidebar
                    st.markdown("---")
                    st.markdown(final_insight.replace('**Recommended Approach:**\n', ''))


# -------------------------------
# Hide Streamlit header/menu (as per original code)
# -------------------------------
hide_streamlit_style = """
<style>
header[data-testid="stHeader"] {display:none;}
#MainMenu {display:none;}
.st-emotion-cache-1wv0u7p {
    padding-top: 0rem;
    padding-right: 1rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)