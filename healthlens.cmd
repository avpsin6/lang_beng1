@echo off
rem Runs HealthLens from this folder, e.g.  healthlens demo
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo HealthLens is not installed yet. Double-click install.bat first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m healthlens %*
