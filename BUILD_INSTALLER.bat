@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Build Color Shadow Box Studio Installer

if not exist "dist\ColorShadowBoxStudio\ColorShadowBoxStudio.exe" (
  echo The Windows application has not been built yet.
  echo Running BUILD_WINDOWS_EXE.bat first...
  call BUILD_WINDOWS_EXE.bat
  if errorlevel 1 exit /b 1
)

set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"

if not exist "%ISCC%" (
  echo.
  echo Inno Setup 6 was not found.
  echo Install Inno Setup 6, then run this file again.
  echo The application EXE is already available in dist\ColorShadowBoxStudio.
  pause
  exit /b 1
)

if not exist installer_output mkdir installer_output
"%ISCC%" "installer\ColorShadowBoxStudio.iss"
if errorlevel 1 (
  echo Installer build failed.
  pause
  exit /b 1
)

echo.
echo Installer created successfully:
echo %CD%\installer_output\ColorShadowBoxStudio_Setup_v1.6.2.exe
start "" "%CD%\installer_output"
pause
