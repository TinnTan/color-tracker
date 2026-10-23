@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m idlelib
if errorlevel 1 pause
