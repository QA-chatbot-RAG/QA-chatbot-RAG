@echo off
set "PROJECT_DIR=%~dp0"
call "%PROJECT_DIR%frontend\.venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_DIR%frontend"
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501
