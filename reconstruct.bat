@echo off
setlocal enabledelayedexpansion
title Photo to 3D  -  Gaussian Splatting pipeline

set "ROOT=%~dp0"

echo ==================================================================
echo    PHOTO TO 3D  -  from a handful of photos to a 3D model
echo ==================================================================
echo.

rem --- Input: dropped onto the .bat, or asked for ---
set "INPUT=%~1"
if "%INPUT%"=="" (
    echo  Drag a .zip ^(or a folder of photos^) onto this file,
    echo  or paste the path below and press Enter.
    echo.
    set /p "INPUT=  Path to zip/folder: "
)
set "INPUT=!INPUT:"=!"

if "!INPUT!"=="" (
    echo.
    echo  No input given. Exiting.
    pause
    exit /b 1
)

rem --- Iterations (Enter = 15000) ---
set "ITER=15000"
set /p "ITER=  Iterations [Enter = 15000]: "

echo.
echo  ------------------------------------------------------------------
echo   Input       : !INPUT!
echo   Iterations  : !ITER!
echo   Max Gaussians: 500000
echo  ------------------------------------------------------------------
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  ERROR: "python" not found in PATH. Install Python 3.9+ first.
    pause
    exit /b 1
)

python "%ROOT%reconstruct.py" "!INPUT!" --iter !ITER!
set "CODE=%errorlevel%"

echo.
if "!CODE!"=="0" (
    echo ==================================================================
    echo   DONE.  Your model is in:  output\^<name^>\model.ply
    echo ==================================================================
) else (
    echo ==================================================================
    echo   FINISHED WITH ERRORS ^(code !CODE!^). Check the messages above.
    echo ==================================================================
)
echo.
pause
endlocal
