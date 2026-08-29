@echo off
cd /d "%~dp0"
echo Installing required libraries...
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Please send a screenshot of this window.
  pause
  exit /b 1
)
echo.
echo Starting Color Shadow Box Studio...
py main.py
if errorlevel 1 pause
