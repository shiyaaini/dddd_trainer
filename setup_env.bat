@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
call :main
set "ERR=!ERRORLEVEL!"
echo.
pause
endlocal & exit /b %ERR%

:main
set "ROOT=%cd%"
set "EMBED_ZIP=%ROOT%\dist\python-3.10.11-embed-amd64.zip"
set "RUNTIME=%ROOT%\runtime"
set "PY=%RUNTIME%\python.exe"
set "PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

echo ========================================
echo   Setup embed Python + base deps
echo ========================================
echo ZIP=%EMBED_ZIP%
echo RUNTIME=%RUNTIME%
echo NOTE: installs requirements.txt (PyQt6 etc.)
echo       does NOT install torch (use GUI Env Check later)
echo.

if not exist "%EMBED_ZIP%" (
  echo [ERROR] Missing embed zip under dist\
  exit /b 1
)
if not exist "%ROOT%\requirements.txt" (
  echo [ERROR] requirements.txt not found
  exit /b 1
)

where tar.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] tar.exe required
  exit /b 1
)
where robocopy.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] robocopy.exe required
  exit /b 1
)

echo [1/4] Extract embed Python to runtime\
if exist "%RUNTIME%" (
  echo       Removing old runtime\
  rmdir /s /q "%RUNTIME%"
  if exist "%RUNTIME%" (
    echo [ERROR] Cannot remove runtime. Close programs using it.
    exit /b 1
  )
)

set "EXTRACT=%ROOT%\dist\_embed_extract"
if exist "%EXTRACT%" rmdir /s /q "%EXTRACT%"
mkdir "%EXTRACT%"
tar.exe -xf "%EMBED_ZIP%" -C "%EXTRACT%"
if errorlevel 1 (
  echo [ERROR] Failed to extract zip
  exit /b 1
)

set "SRC_DIR="
if exist "%EXTRACT%\python-3.10.11-embed-amd64\python.exe" set "SRC_DIR=%EXTRACT%\python-3.10.11-embed-amd64"
if not defined SRC_DIR if exist "%EXTRACT%\python.exe" set "SRC_DIR=%EXTRACT%"
if not defined SRC_DIR (
  echo [ERROR] python.exe not found inside zip
  exit /b 1
)

mkdir "%RUNTIME%"
robocopy "%SRC_DIR%" "%RUNTIME%" /E /NFL /NDL /NJH /NJS /nc /ns /np >nul
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 (
  echo [ERROR] robocopy to runtime failed code=!RC!
  exit /b 1
)
if exist "%EXTRACT%" rmdir /s /q "%EXTRACT%"

if not exist "%PY%" (
  echo [ERROR] runtime\python.exe missing after extract
  exit /b 1
)

echo [2/4] Ensure site-packages enabled
set "PTH=%RUNTIME%\python310._pth"
if exist "%PTH%" (
  >"%PTH%" (
    echo python310.zip
    echo .
    echo Lib\site-packages
    echo import site
  )
)

echo [3/4] Verify pip
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  if exist "%RUNTIME%\get-pip.py" (
    echo       Running get-pip.py ...
    "%PY%" "%RUNTIME%\get-pip.py" --no-warn-script-location
  )
)
"%PY%" -m pip --version
if errorlevel 1 (
  echo [ERROR] pip is not available in runtime
  exit /b 1
)
"%PY%" -c "import sys; print(sys.version)"

echo [4/4] Install base deps (numpy first, then requirements; no torch)
"%PY%" -m pip install --upgrade pip -i "%PIP_MIRROR%"
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip
  exit /b 1
)
set "NUMPY_WHL=%ROOT%\dist\numpy-1.24.4-cp310-cp310-win_amd64.whl"
echo       Installing numpy first ...
if exist "%NUMPY_WHL%" (
  echo       Offline wheel: %NUMPY_WHL%
  "%PY%" -m pip install "%NUMPY_WHL%" --no-cache-dir
) else (
  echo       [WARN] Offline wheel missing, fallback online numpy==1.24.4
  "%PY%" -m pip install "numpy==1.24.4" -i "%PIP_MIRROR%"
)
if errorlevel 1 (
  echo [ERROR] Failed to install numpy
  exit /b 1
)
"%PY%" -m pip install -r "%ROOT%\requirements.txt" -i "%PIP_MIRROR%"
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.txt
  exit /b 1
)

echo.
echo Smoke check ...
"%PY%" -c "import fire,loguru,yaml,tqdm,numpy,PIL,PyQt6; print('base deps OK')"
if errorlevel 1 (
  echo [ERROR] Smoke import failed
  exit /b 1
)

echo.
echo ========================================
echo   DONE
echo ========================================
echo runtime is ready. Torch is NOT installed.
echo Next: double-click run_gui.bat
exit /b 0