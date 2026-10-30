@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PGCLIENTENCODING=UTF8
if "%PGBIN%"=="" set PGBIN=C:\Program Files\PostgreSQL\18\bin
rem 지금 PC의 medical_rag DB를 medical_rag.dump 파일로 저장해요
"%PGBIN%\pg_dump.exe" -U postgres -Fc -Z 9 --no-owner -f medical_rag.dump medical_rag
if errorlevel 1 goto fail
for %%A in (medical_rag.dump) do echo 파일 크기: %%~zA 바이트
echo 완료!
pause
exit /b 0
:fail
echo 중간에 실패했어요. 위 메시지를 확인하세요.
pause
exit /b 1
