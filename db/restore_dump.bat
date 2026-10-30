@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PGCLIENTENCODING=UTF8
if "%PGBIN%"=="" set PGBIN=C:\Program Files\PostgreSQL\18\bin
rem medical_rag.dump (테이블+데이터+임베딩 전부) 로 medical_rag DB를 새로 만들어요
echo 예전 medical_rag DB가 있으면 지우고 다시 만들어요
"%PGBIN%\dropdb.exe" -U postgres --if-exists medical_rag
if errorlevel 1 goto fail
"%PGBIN%\createdb.exe" -U postgres -E UTF8 -T template0 medical_rag
if errorlevel 1 goto fail
"%PGBIN%\pg_restore.exe" -U postgres -d medical_rag --no-owner medical_rag.dump
if errorlevel 1 goto fail
echo 완료!
pause
exit /b 0
:fail
echo 중간에 실패했어요. 위 메시지를 확인하세요.
pause
exit /b 1
