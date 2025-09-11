@echo OFF
echo --- Starting Hyperparameter Sweep for Sepsis Agent ---

REM Activate your conda environment
call conda activate rl_med_diag_final

REM Define the values for the sweep
set uncertainty_factors=0 0.05 0.1 0.3
set cost_scales=1.0 1.5 2.0

REM Loop through each combination and run the training
FOR %%U IN (%uncertainty_factors%) DO (
    FOR %%C IN (%cost_scales%) DO (
        echo.
        echo ==========================================================
        echo      RUNNING SWEEP: Uncertainty=%%U, Cost Scale=%%C
        echo ==========================================================
        
        python src/training/train_rl_agent_sepsis.py ^
            --uncertainty_factor %%U ^
            --cost_scale %%C ^
            --model_suffix _unc-%%U_cost-%%C
    )
)

echo.
echo --- Sweep Complete ---
pause