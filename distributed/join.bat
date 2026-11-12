@echo off
rem Folding Lab volunteer training. Sets up what is missing, then trains until this window is closed.
rem The coordinator fills in {SERVER} and {TOKEN} when it serves this file from the join link.
title Folding Lab training
set "DIR=%USERPROFILE%\folding-lab-train"
if not exist "%DIR%" mkdir "%DIR%"
cd /d "%DIR%" || goto :failed

where py >nul 2>nul || goto :nopython
py -c "import numpy" >nul 2>nul || py -m pip install numpy || goto :failed
rem the CUDA 12.8 build: RTX 50-series GPUs need it, and it also runs on older NVIDIA GPUs and on the CPU
py -c "import torch" >nul 2>nul || py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :failed
curl -fsS -o contribute.py "{SERVER}/join/contribute.py?token={TOKEN}" || goto :unreachable
curl -fsS -o evonet.py "{SERVER}/join/evonet.py?token={TOKEN}" || goto :unreachable
py contribute.py --server {SERVER} --token {TOKEN} --chains 32
pause
exit /b

:nopython
echo Python is missing, so Windows will install Python 3.13 now.
winget install -e --id Python.Python.3.13
echo.
echo When that has finished, close this window and double-click this file again.
echo If nothing was installed, get Python 3.13 from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in its installer.
pause
exit /b 1

:unreachable
echo.
echo Could not reach {SERVER}
echo Check that the Mac is awake, that training is running on it, and that both computers are on the same network.
pause
exit /b 1

:failed
echo.
echo Setup did not finish. The lines above say why.
pause
exit /b 1
