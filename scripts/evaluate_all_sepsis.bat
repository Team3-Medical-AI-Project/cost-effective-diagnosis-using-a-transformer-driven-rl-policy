@echo OFF
echo --- Starting Evaluation for All Sepsis Models ---

call conda activate rl_med_diag_final

echo.
echo ==========================================================
echo      EVALUATING: Main Reproducible Model
echo ==========================================================
python src/training/evaluate_agent_sepsis.py --model_path models/rl_agent_sepsis_final_reproducible.zip

echo.
echo ==========================================================
echo      EVALUATING: Ablation - No Transformer
echo ==========================================================
python src/training/evaluate_agent_sepsis.py --model_path models/rl_agent_sepsis_ablation_no_transformer.zip

echo.
echo ==========================================================
echo      EVALUATING: Ablation - No Masking
echo ==========================================================
python src/training/evaluate_agent_sepsis.py --model_path models/rl_agent_sepsis_ablation_no_masking.zip

echo.
echo ==========================================================
echo      EVALUATING: Ablation - No GAIN
echo ==========================================================
python src/training/evaluate_agent_sepsis.py --model_path models/rl_agent_sepsis_ablation_no_gain.zip


echo.
echo --- Evaluation Complete ---
pause