@echo off
REM =============================================================================
REM EJECUTAR ENTRENAMIENTO CON MÉTRICAS - WINE QUALITY PREDICTOR
REM =============================================================================
echo.
echo ========================================
echo   WINE QUALITY PREDICTOR
echo   ENTRENAMIENTO Y VISUALIZACIONES
echo ========================================
echo.

cd src\models
python train_with_metrics.py

pause
