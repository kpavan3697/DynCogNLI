# Set environment variables
$env:STREAMLIT_WATCHER_TYPE = "none"
$env:OMP_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Optional: Activate Python environment
# & "$HOME\.venv\Scripts\Activate.ps1"

# Run Streamlit
streamlit run app.py --server.runOnSave false
