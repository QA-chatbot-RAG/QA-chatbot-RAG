@echo off
set "PROJECT_DIR=%~dp0"
call "%PROJECT_DIR%.venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
cd /d "%PROJECT_DIR%"
python -m uvicorn main:app --host 127.0.0.1 --port 8001
