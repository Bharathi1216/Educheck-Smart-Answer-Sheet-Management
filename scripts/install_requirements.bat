@echo off
REM Minimal installer for Windows (cmd)
REM Usage: open PowerShell/cmd in project root and run: scripts\install_requirements.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements from requirements.txt...
python -m pip install -r "%~dp0\..\requirements.txt"

echo Done. If you want an isolated environment, create and activate a venv first:
echo python -m venv venv
echo venv\Scripts\activate
