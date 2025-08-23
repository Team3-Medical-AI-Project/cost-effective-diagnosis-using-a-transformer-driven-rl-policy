@echo off
REM === Train + Evaluate main Sepsis agent for interim report (Windows/cmd) ===
setlocal
cd /d "%~dp0.."
call conda activate rl_med_diag_final

REM -- Patch the trainer once (Gymnasium removed env.seed). Safe to re-run.
powershell -Command "$p='src\\training\\train_rl_agent_sepsis_fast.py'; if (Test-Path $p) { $s=Get-Content $p -Raw; $r=$s -replace 'env\.seed\(seed \+ rank\)', 'try:`n            env.reset(seed=seed + rank)`n        except TypeError:`n            pass'; if($s -ne $r){ Set-Content $p $r -Encoding UTF8; Write-Host 'Patched train_rl_agent_sepsis_fast.py (seed->reset)'; } else { Write-Host 'Seed patch already present.' } }"

REM -- Auto-pick device (cuda if available)
for /f %%i in ('python -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')"') do set DEVICE=%%i

set CONFIG=configs\sepsis_config.yaml
set MODEL=models\rl_agent_sepsis_interim_main.zip
set OUTDIR=reports\interim_main

echo.
echo ===== TRAIN MAIN AGENT =====
python -m src.training.train_rl_agent_sepsis_fast ^
  --config %CONFIG% ^
  --device %DEVICE% ^
  --n_envs 1 ^
  --total_timesteps 300000 ^
  --use_two_diagnosis_actions ^
  --model_name %MODEL%
IF ERRORLEVEL 1 ( echo [ERROR] Training failed & exit /b 1 )

echo.
echo ===== EVALUATE MAIN AGENT (metrics + basic plots) =====
python -m src.eval.evaluate_interim_fast ^
  --config %CONFIG% ^
  --model %MODEL% ^
  --n-episodes 400 ^
  --outdir %OUTDIR%
IF ERRORLEVEL 1 ( echo [ERROR] Evaluation failed & exit /b 1 )

echo.
echo ===== EXTRA VISUALS (ROC/PR/Calibration/Cost) =====
python -m src.eval.evaluate_interim_fast_viz ^
  --config %CONFIG% ^
  --model %MODEL% ^
  --n-episodes 400 ^
  --outdir %OUTDIR%

echo.
echo ===== SUMMARY IMAGE =====
python -m src.eval.make_interim_summary_plot ^
  --metrics %OUTDIR%\metrics.json ^
  --out %OUTDIR%\summary_table.png ^
  --title "Sepsis RL — Interim Main"

echo.
echo ===== METRICS =====
type %OUTDIR%\metrics.json

echo.
echo Done. Artifacts saved to:
echo   %OUTDIR%\summary_table.png
echo   %OUTDIR%\confusion_matrix.png
echo   %OUTDIR%\panel_usage.png
echo   %OUTDIR%\roc_curve.png
echo   %OUTDIR%\pr_curve.png
echo   %OUTDIR%\calibration_curve.png
echo   %OUTDIR%\cost_hist.png
echo   %OUTDIR%\case_studies.txt
endlocal
