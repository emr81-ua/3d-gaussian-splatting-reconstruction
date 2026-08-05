@echo off
rem Abre la interfaz grafica (ventana) del reconstructor 3D.
cd /d "%~dp0"

where pythonw >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw "%~dp0gui.py"
    exit /b
)
where python >nul 2>&1
if %errorlevel%==0 (
    python "%~dp0gui.py"
    exit /b
)
echo No se encontro Python. Instala Python 3.9 o superior.
pause
