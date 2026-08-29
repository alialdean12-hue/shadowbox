@echo off
cd /d "%~dp0"
echo Installing optional HEIC/HEIF support...
py -m pip install pillow-heif
pause
