@echo off
setlocal enabledelayedexpansion
title Reconstruccion 3D - Gaussian Splatting

set "ROOT=%~dp0"
set "ENTRADA=%ROOT%entrada"
set "SALIDA=%ROOT%salida"

rem --- Rutas de COLMAP / LichtFeld (archivo local, no se sube al repo) ---
if exist "%ROOT%herramientas.local.bat" call "%ROOT%herramientas.local.bat"

echo ==================================================================
echo    RECONSTRUCCION 3D  -  de un lote de fotos a un modelo 3D
echo ==================================================================
echo.
echo   Pon tus fotos en:  %ENTRADA%
echo.

rem --- Comprobar que hay imagenes en la carpeta de entrada ---
set "HAY=0"
for %%E in (jpg jpeg png bmp tif tiff) do (
    for %%F in ("%ENTRADA%\*.%%E") do set "HAY=1"
)
if "!HAY!"=="0" (
    echo   [!] No hay fotos en la carpeta "entrada".
    echo       Copia tu lote de fotos dentro de esa carpeta y vuelve a ejecutar.
    echo.
    pause
    exit /b 1
)

rem --- Iteraciones (Enter = 15000) ---
set "ITER=15000"
set /p "ITER=  Iteraciones [Enter = 15000, prueba rapida = 7000]: "

rem --- Carpeta de salida unica por fecha y hora ---
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "OUT=%SALIDA%\%STAMP%"

echo.
echo  ------------------------------------------------------------------
echo   Entrada : %ENTRADA%
echo   Salida  : %OUT%
echo   Iter    : %ITER%    Quitar fondo: SI (mascaras + recorte)
echo  ------------------------------------------------------------------
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  ERROR: "python" no esta en el PATH. Instala Python 3.9 o superior.
    pause
    exit /b 1
)

python "%ROOT%reconstruct.py" "%ENTRADA%" --output "%OUT%" --iter %ITER%
set "CODE=%errorlevel%"

echo.
if "%CODE%"=="0" (
    rem --- Limpiar la carpeta de entrada: las fotos ya estan copiadas en la salida ---
    for %%E in (jpg jpeg png bmp tif tiff) do del /q "%ENTRADA%\*.%%E" >nul 2>&1
    echo ==================================================================
    echo   LISTO.  Todo esta en:
    echo   %OUT%
    echo       images\      fotos iniciales ^(copia de seguridad^)
    echo       dense\       reconstruccion COLMAP ^(poses + nube^)
    echo       dense\masks\ mascaras del sujeto
    echo       model.ply    modelo 3D final ^(sin fondo^)
    echo.
    echo   La carpeta "entrada" se ha vaciado (los originales estan en images\).
    echo ==================================================================
    start "" explorer "%OUT%"
) else (
    echo ==================================================================
    echo   TERMINO CON ERRORES ^(codigo %CODE%^). Revisa los mensajes de arriba.
    echo ==================================================================
)
echo.
pause
endlocal
