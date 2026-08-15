@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call :main
set "ERR=%ERRORLEVEL%"
echo.
pause
endlocal & exit /b %ERR%

:main
set "PY=%~dp0runtime\python.exe"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"

echo ========================================
echo   dddd_trainer GUI launcher
echo ========================================
echo.

if not exist "%~dp0gui_app.py" (
  echo [ERROR] gui_app.py not found
  exit /b 1
)

if not exist "%PY%" (
  echo [ERROR] runtime\python.exe not found
  echo         Run setup_env.bat first
  exit /b 1
)

echo [INFO] Using embed Python: runtime\python.exe
"%PY%" -c "import sys; print(sys.version)"
if errorlevel 1 (
  echo [ERROR] runtime Python cannot run
  exit /b 1
)

"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  if exist "%~dp0runtime\get-pip.py" (
    echo [INFO] Installing pip via get-pip.py ...
    "%PY%" "%~dp0runtime\get-pip.py" --no-warn-script-location
  )
)
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] pip not available in runtime
  exit /b 1
)

"%PY%" -c "import fire,loguru,yaml,tqdm,numpy,PIL,PyQt6" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing base deps ...
  if not exist "%~dp0requirements.txt" (
    echo [ERROR] requirements.txt not found
    exit /b 1
  )
  "%PY%" -m pip install --upgrade pip -i "%PIP_MIRROR%"
  if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    exit /b 1
  )
  set "NUMPY_WHL=%~dp0dist\numpy-1.24.4-cp310-cp310-win_amd64.whl"
  if exist "%NUMPY_WHL%" (
    echo [INFO] Offline numpy wheel
    "%PY%" -m pip install "%NUMPY_WHL%" --no-cache-dir
  ) else (
    echo [WARN] Offline wheel missing, fallback online
    "%PY%" -m pip install "numpy==1.24.4" -i "%PIP_MIRROR%"
  )
  if errorlevel 1 (
    echo [ERROR] Failed to install numpy
    exit /b 1
  )
  "%PY%" -m pip install -r "%~dp0requirements.txt" -i "%PIP_MIRROR%"
  if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    exit /b 1
  )
  echo [INFO] Requirements installed
) else (
  echo [INFO] Base requirements already installed
)

"%PY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [TIP] PyTorch not installed. Use Env Check in GUI later.
) else (
  echo [INFO] PyTorch detected
)

echo.
echo [INFO] Starting GUI ...
"%PY%" "%~dp0gui_app.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [ERROR] GUI exited with code %RC%
  exit /b 1
)
exit /b 0