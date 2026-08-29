@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Build Color Shadow Box Studio v1.6.2 Setup

echo ==================================================
echo  Color Shadow Box Studio v1.6.2 - Full Setup Build
echo ==================================================
echo.
echo This will first build the Windows application, then the Setup installer.
echo.
call BUILD_WINDOWS_EXE.bat
if errorlevel 1 exit /b 1
call BUILD_INSTALLER.bat
if errorlevel 1 exit /b 1
exit /b 0
