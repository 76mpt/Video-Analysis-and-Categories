@echo off
cd /d %~dp0
start "vsort web" /min .venv\Scripts\python.exe -m vsort web
