@echo off
REM === Evaluate an already-trained main model (no training) ===
setlocal
cd /d "%~dp0.."
call conda activate rl_med_diag_final

set CONFIG=configs\sepsis_config.yaml
set MODEL=models\rl_agent_sepsis_interim_main.zip
set OUTDIR=reports\quick_check

if not exist "%MODEL%" (
  echo [ERROR] Missing model: %MODEL%
  exit /b 1
)

echo.
echo ===== EVALUATE MAIN AGENT (metrics + basic plots) =====
python -m src.eval.evaluate_interim_fast ^
  --config %CONFIG% ^
  --model %MODEL% ^
  --n-episodes 200 ^
  --outdir %OUTDIR%
IF ERRORLEVEL 1 ( echo [ERROR] Evaluation failed & exit /b 1 )

echo.
echo ===== EXTRA VISUALS (ROC/PR/Calibration/Cost) =====
python -m src.eval.evaluate_interim_fast_viz ^
  --config %CONFIG% ^
  --model %MODEL% ^
  --n-episodes 200 ^
  --outdir %OUTDIR%

echo.
echo ===== SUMMARY IMAGE =====
python -m src.eval.make_interim_summary_plot ^
  --metrics %OUTDIR%\metrics.json ^
  --out %OUTDIR%\summary_table.png ^
  --title "Sepsis RL — Quick Check"

echo.
echo ===== METRICS =====
type %OUTDIR%\metrics.json

echo.
echo See plots in %OUTDIR%
endlocal
