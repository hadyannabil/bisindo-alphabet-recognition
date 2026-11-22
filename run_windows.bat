@echo off
cd /d "%~dp0"
set "PYTHONPATH=src"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m bisindo_app.ui %*
) else (
  python -m bisindo_app.ui %*
)
pause
