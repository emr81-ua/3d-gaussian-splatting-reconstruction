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

rem --- Tipo de reconstruccion: persona / objeto (con mascaras) o escena completa ---
echo.
echo   Tipo de reconstruccion:
echo       [1] Persona          (quita el fondo con mascaras)
echo       [2] Objeto           (quita el fondo con mascaras)
echo       [3] Escena completa  (SIN mascaras, lo saca todo)
set "MASKARGS=--mask-model u2net_human_seg"
set "SUJETO=Persona"
set "QF=SI (mascaras + recorte)"
set "TIPO=1"
set /p "TIPO=  Elige 1, 2 o 3 [Enter = 1 Persona]: "
if "%TIPO%"=="2" set "MASKARGS=--mask-model u2net"
if "%TIPO%"=="2" set "SUJETO=Objeto"
if "%TIPO%"=="3" set "MASKARGS=--no-mask --no-crop"
if "%TIPO%"=="3" set "SUJETO=Escena completa"
if "%TIPO%"=="3" set "QF=NO (escena completa)"

rem --- Carpeta de salida unica por fecha y hora ---
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "OUT=%SALIDA%\%STAMP%"

echo.
echo  ------------------------------------------------------------------
echo   Entrada : %ENTRADA%
echo   Salida  : %OUT%
echo   Iter    : %ITER%    Modo: %SUJETO%    Quitar fondo: %QF%
echo  ------------------------------------------------------------------
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo  ERROR: "python" no esta en el PATH. Instala Python 3.9 o superior.
    pause
    exit /b 1
)

python "%ROOT%reconstruct.py" "%ENTRADA%" --output "%OUT%" --iter %ITER% %MASKARGS%
set "CODE=%errorlevel%"

echo.
rem --- Vaciar la carpeta de entrada al terminar (solo si las fotos ya se copiaron a la salida) ---
if exist "%OUT%\images\*.*" (
    for %%E in (jpg jpeg png bmp tif tiff) do del /q "%ENTRADA%\*.%%E" >nul 2>&1
    echo   Carpeta "entrada" vaciada. Copia de seguridad de las fotos en: %OUT%\images\
    echo.
)

if "%CODE%"=="0" (
    echo ==================================================================
    echo   LISTO.  Todo esta en:
    echo   %OUT%
    echo       images\      fotos iniciales ^(copia de seguridad^)
    echo       dense\       reconstruccion COLMAP ^(poses + nube^)
    echo       dense\masks\ mascaras del sujeto
    echo       model.ply    modelo 3D final ^(sin fondo^)
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
