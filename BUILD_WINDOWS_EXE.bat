@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Build Color Shadow Box Studio

echo ==================================================
echo   Building Color Shadow Box Studio for Windows
echo ==================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python, enable Add Python to PATH, then run this file again.
    pause
    exit /b 1
  )
  set "PY_CMD=python"
)

if not exist ".buildenv\Scripts\python.exe" (
  echo [1/6] Creating an isolated build environment...
  %PY_CMD% -m venv .buildenv
  if errorlevel 1 goto :fail
) else (
  echo [1/6] Build environment already exists.
)

call ".buildenv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo [2/6] Updating build tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/6] Installing application requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/6] Installing PyInstaller...
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 goto :fail

echo [5/6] Cleaning old build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist ColorShadowBoxStudio.spec del /q ColorShadowBoxStudio.spec

echo [6/6] Creating the Windows application...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name ColorShadowBoxStudio ^
  --version-file windows_version_info.txt ^
  --collect-data reportlab ^
  --collect-submodules PIL ^
  --hidden-import worker_runtime ^
  main.py
if errorlevel 1 goto :fail

echo.
echo ==================================================
echo BUILD SUCCEEDED
echo Application folder:
echo %CD%\dist\ColorShadowBoxStudio
echo.
echo Main program:
echo %CD%\dist\ColorShadowBoxStudio\ColorShadowBoxStudio.exe
echo ==================================================
start "" "%CD%\dist\ColorShadowBoxStudio"
pause
exit /b 0

:fail
echo.
echo ==================================================
echo BUILD FAILED
echo Read the last error shown above.
echo ==================================================
pause
exit /b 1
