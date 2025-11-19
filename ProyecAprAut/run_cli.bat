@echo off
REM =============================================================================
REM EJECUTAR CLI INTERACTIVO - WINE QUALITY PREDICTOR
REM =============================================================================
echo.
echo ========================================
echo   WINE QUALITY PREDICTOR - CLI
echo ========================================
echo.

cd src\models
python train_mlp_interactive.py

pause
