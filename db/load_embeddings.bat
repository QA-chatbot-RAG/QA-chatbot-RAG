@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PGCLIENTENCODING=UTF8
if "%PGBIN%"=="" set PGBIN=C:\Program Files\PostgreSQL\18\bin
if not exist medical_embeddings.csv (
  echo medical_embeddings.csv 파일을 db 폴더에 넣어주세요.
  goto fail
)
"%PGBIN%\psql.exe" -U postgres -d medical_rag -f load_embeddings.sql
if errorlevel 1 goto fail
echo 완료!
pause
exit /b 0
:fail
echo 중간에 실패했어요. 위 메시지를 확인하세요.
pause
exit /b 1
