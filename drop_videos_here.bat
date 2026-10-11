@echo off
rem Drag video files or folders onto this file: it first shows a dry run of how they would enter 01_Inbox,
rem then applies it only after you press Y. Double-click it to pick videos in a dialog instead.
chcp 65001 >nul
set "PYTHONPATH=%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if "%~1"=="" (
  "%PY%" -m vsort ingest --pick
) else (
  "%PY%" -m vsort ingest %*
)
if errorlevel 1 goto end
echo.
set /p ok=Apply this plan? (Y/N)
if /i "%ok%"=="Y" "%PY%" -m vsort apply
:end
echo.
pause
