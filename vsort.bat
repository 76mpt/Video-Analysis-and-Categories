@echo off
rem Usage: vsort.bat <command> [options], e.g. vsort.bat ingest --pick
chcp 65001 >nul
set "PYTHONPATH=%~dp0"
"%~dp0.venv\Scripts\python.exe" -m vsort %*
