# Weather-Aware Decision Support for PV Operation and Maintenance

Code, data splits and results for the article *AI-Driven PV Monitoring and Decision Support via Vision Transformers and LLM-Based Reasoning*.

The framework combines two image classifiers, a rule matrix and a large language model (LLM):

- **MiT-B0** classifies 11 weather conditions.
- **Swin-Tiny** classifies six PV module states.
- The **Small Reasoning Module (SRM)** is a 66-rule matrix. It maps each predicted weather–module pair to a risk level, an alert status, a human action and an automated action.
- An **LLM consultation layer** (Gemma 4 31B) receives the SRM decision, explains it and reports safety concerns. It does not override the decision.

## Contents

| Folder | Content |
|---|---|
| `training/` | `pv_finetune.py`: duplicate removal (MD5), fixed stratified 30 % test split (seed 2026), 85/15 train/validation split per seed, end-to-end fine-tuning of six models (SimpleCNN, ResNet-18, EfficientNet-B0, ViT-Tiny, MiT-B0, Swin-Tiny) on both tasks with five seeds, early stopping on validation macro-F1 (patience 4, min-delta 0.002, at most 30 epochs), metrics, significance tests, FLOPs/latency and label audit. `setup_env.sh` creates the environment. |
| `results/` | `ALL_RESULTS.json`, `summary.json` (mean ± SD over seeds, paired t-tests, McNemar tests), `efficiency.json`, `label_audit.json`, `data_summary.json`, `srm_analysis.json`; `runs/` holds one JSON per model, task and seed (metrics, confusion matrix, training history, bootstrap CIs); `splits/` holds the exact train/test file lists; `figures/`. |
| `decision_analysis/` | `srm_versions.json` holds rule bases v1, v2 and v3, their audit against safety principles P1–P5, and the list of changed rules. `propagate_errors.py` propagates the per-seed confusion matrices exactly through the rules. It also computes the module-state-only baseline and the weather-dependence counts. |
| `llm_evaluation/` | `PV_LLM_Consultation_Evaluation.ipynb` (Google Colab, CPU) runs the rule audit and queries Gemma 4 31B for the 66 scenarios. The LLM is queried alone and given the SRM decision, with three repetitions at temperature 0.7, and the answers are scored against the SRM risk levels and principles P1–P5. `responses/` holds the raw responses and `llm_eval_v3.json`. |

## Data

The two public datasets are downloaded automatically with `kagglehub`:

- Weather: [jehanbhathena/weather-dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset), 6,862 images. 6,673 remain after duplicate removal.
- PV modules: [pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images), 885 images. 792 remain after duplicate removal.

The weather and module images come from independent sources. The 66 scenarios are therefore constructed pairings, not synchronised field observations.

## Reproduction

```bash
bash training/setup_env.sh                       # Python 3.11, PyTorch, timm, transformers, kagglehub
python training/pv_finetune.py                   # outputs in training/out (resumable); 60 runs ≈ 1 h on one H100 MIG 1g.10gb
python decision_analysis/propagate_errors.py     # decision-level propagation for rule bases v1-v3
```

The LLM evaluation runs in Google Colab and needs a Gemini API key stored in Colab Secrets as `GOOGLE_API_KEY`. Responses are appended to a log as they arrive, so an interrupted run resumes where it stopped. When a scenario has more than one saved response, the evaluation keeps the first. This applies to one duplicate in `llm_responses_gemma.jsonl`.

## Main results

| | MiT-B0 (weather) | Swin-Tiny (PV module) |
|---|---|---|
| Test macro-F1 (mean ± SD, 5 seeds) | 0.912 ± 0.009 | 0.906 ± 0.015 |
| Test accuracy | 0.907 ± 0.009 | 0.899 ± 0.014 |

Decision level, rule base v3:

- The decision equals the error-free decision in 0.828 of image pairs.
- The risk level is underestimated in 0.035 of pairs.
- A baseline that uses the module state alone gives the correct decision in 0.203 of pairs.

LLM comparison:

| Gemma 4 31B | Risk level equal to the SRM | Answers violating a safety principle |
|---|---|---|
| Alone | 0.561 | 14.6 % |
| Given the SRM v3 decision | 1.000 | 0 % |

## License

MIT (code). The datasets remain under their providers' terms.
