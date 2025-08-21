:: Set environment variables
set STREAMLIT_WATCHER_TYPE=none
set OMP_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE

:: Optional: Activate Python environment
:: call "%USERPROFILE%\.venv\Scripts\activate.bat"

:: Run Streamlit
streamlit run app.py --server.runOnSave false
