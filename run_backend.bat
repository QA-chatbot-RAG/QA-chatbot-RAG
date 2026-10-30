@echo off
set "PROJECT_DIR=%~dp0"
call "%PROJECT_DIR%backend\.venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_DIR%backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
