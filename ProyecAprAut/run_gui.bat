@echo off
REM =============================================================================
REM EJECUTAR INTERFAZ GRÁFICA - WINE QUALITY PREDICTOR
REM =============================================================================
echo.
echo ========================================
echo   WINE QUALITY PREDICTOR - GUI
echo ========================================
echo.

cd src\gui
python wine_predictor_gui.py

pause
