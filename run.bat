@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
) else (
    set "PY_CMD=python"
)

echo Running Clot Database Generator (M = 9, 6, 3 | 300 s | 30 s frames)...
%PY_CMD% generate_clots.py %*
echo.
pause
