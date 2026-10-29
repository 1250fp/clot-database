@echo off
rem ======================================================================
rem  Stage 1 (m=4, m=8 at f0=2.233) + stage 2 (m=8 at f0=0 and 4.466)
rem  3 replicas each, 600 s now (saved so they can be extended to 1800 s).
rem  Starts 3 copies side by side, each in its own window and folder:
rem    A: m=4 f0=2.233 reps 1-3   + m=8 f0=2.233 reps 1-2   (28 slabs)
rem    B: m=8 f0=0     reps 1-3   + m=8 f0=2.233 rep 3      (32 slabs)
rem    C: m=8 f0=4.466 reps 1-3                             (24 slabs, slower)
rem
rem  Usage:  run_lab_5080.bat              (output in D:\clot-data\stage12_5080,
rem                                         or next to this .bat if there is no D:)
rem          run_lab_5080.bat E:\somewhere (output in that folder instead)
rem
rem  STOP:   close the 3 windows (or Ctrl+C in each). Each slab keeps a
rem          checkpoint every 300 s of simulated time; at most that is redone.
rem  RESUME (same or another PC, e.g. your 5060): copy the whole output folder
rem          (copyA, copyB, copyC with their .ckpt.npz and catalog.json) plus this
rem          clot-database folder, then run this .bat again with the folder path.
rem          It always uses --resume, so finished slabs are skipped.
rem          Never run the same folder on two PCs at the same time.
rem  Needs:  NVIDIA driver + Python with numpy (pip install -r requirements.txt).
rem ======================================================================
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

set "OUTROOT=%~1"
if "%OUTROOT%"=="" if exist D:\ set "OUTROOT=D:\clot-data\stage12_5080"
if "%OUTROOT%"=="" set "OUTROOT=%~dp0stage12_5080"
if not exist "%OUTROOT%" mkdir "%OUTROOT%"
echo Output folder: %OUTROOT%

set "RUN=%PY% -u generate_library.py --duration 1800 --stop-at 600 --resume"

start "clot copy A" cmd /c "%RUN% --out "%OUTROOT%\copyA" --m 4 --f0 2.233 --reps 1 2 3 & %RUN% --out "%OUTROOT%\copyA" --m 8 --f0 2.233 --reps 1 2 & pause"
start "clot copy B" cmd /c "%RUN% --out "%OUTROOT%\copyB" --m 8 --f0 0 --reps 1 2 3 & %RUN% --out "%OUTROOT%\copyB" --m 8 --f0 2.233 --reps 3 & pause"
start "clot copy C" cmd /c "%RUN% --out "%OUTROOT%\copyC" --m 8 --f0 4.466 --reps 1 2 3 & pause"

echo Started 3 copies. Progress: %OUTROOT%\copyA\progress.txt (and copyB, copyC).
echo Each window says "Done." and waits for a key when its copy has finished.
