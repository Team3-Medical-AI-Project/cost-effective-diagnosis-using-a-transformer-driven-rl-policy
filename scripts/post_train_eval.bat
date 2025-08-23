@echo off
setlocal
set CFG=configs\sepsis_config.yaml
set MODEL=models\rl_agent_sepsis_interim_main.zip
set OUT=reports\interim_main
set PYTHONUTF8=1

echo Evaluating %MODEL% ...
python -m src.eval.evaluate_interim_fast --config "%CFG%" --model "%MODEL%" --n-episodes 400 --outdir "%OUT%"
python -m src.eval.evaluate_interim_fast_viz --config "%CFG%" --model "%MODEL%" --n-episodes 400 --outdir "%OUT%"

echo Creating compact summary image...
python - <<PY
import json, os, matplotlib.pyplot as plt
out="reports/interim_main"
m=json.load(open(os.path.join(out,"results.json")))
fig,ax=plt.subplots(figsize=(6,4))
txt=(f"Acc: {m['acc']:.3f}\nPrec: {m['precision']:.3f}\nRec: {m['recall']:.3f}\nF1: {m['f1']:.3f}\n"
     f"TN:{m['tn']}  FP:{m['fp']}  FN:{m['fn']}  TP:{m['tp']}\n"
     f"Avg cost: ${m['avg_cost']:.2f}   Steps: {m['avg_steps']:.1f}\nEpisodes: {m['n_episodes']}")
ax.axis('off'); ax.text(0.02,0.98,txt,va='top',ha='left',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(out,"summary_table.png"), dpi=200)
print("summary_table.png saved in", out)
PY

echo Done. Figures in %OUT%.
endlocal
