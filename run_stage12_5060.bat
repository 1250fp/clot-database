@echo off
rem Stage 1 m=16 (f0=2.233, reps 1-3, 600 s, extendable to 1800 s) on the 5060:
rem 3 minimized copies, one per replica, each in its own folder with a log.
rem Safe to run again after a stop/reboot: --resume skips finished work.
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py") || (set "PY=python")
set "OUTROOT=%~1"
if "%OUTROOT%"=="" set "OUTROOT=D:\clot-data"
if not exist "%OUTROOT%" mkdir "%OUTROOT%"
for %%r in (1 2 3) do start "m16 rep%%r" /min cmd /c "%PY% -u generate_library.py --out "%OUTROOT%\stage12_5060_r%%r" --m 16 --f0 2.233 --reps %%r --duration 1800 --stop-at 600 --resume >> "%OUTROOT%\stage12_5060_r%%r.log" 2>&1"
echo Started m=16 reps 1-3. Logs: %OUTROOT%\stage12_5060_r1.log (r2, r3).
