UC2 Full 3-Year Training Script
================================

File:
- train_uc2_full_3y.py

Purpose:
- Standalone training pipeline aligned to the notebook Sections 2-10 (no EDA/dashboard export).
- Uses full 3-year data by default (unless you set year/week filters).
- Builds features, runs train/validation/test split, baseline + tree models, threshold tuning, model selection, and final test evaluation.

Main Inputs (local CSV files):
- activeitems.csv
- train_P2_inbound_allitems_3years.csv
- promitto_weekly_sales.csv
- promitto_purchase_orders.csv
- promitto_receptions.csv
- network lead times.csv
- promitto active dcs.csv

External Data (auto-generated when missing or when refresh flags are used):
- ATC_COMPOSITE_TRENDS.csv (Google Trends via pytrends)
- CPI_DRUG_WEEKLY.csv (Statistics Canada URL)
- FLUNET_CANADA_WEEKLY.csv (WHO FluNet URL)
- DPD txt files from Health Canada (drug/ingred/ther/status/comp and _ia variants)

Install:
1) Create/activate your Python environment (Python 3.10+ recommended).
2) Install dependencies from requirenment.txt:
   pip install -r requirenment.txt

Quick Start:
python train_uc2_full_3y.py --data-dir /Users/Desktop/UC2 --output-dir /Users/Desktop/UC2/Output_py

Run with external refresh:
python train_uc2_full_3y.py \
  --data-dir /Users/Desktop/UC2 \
  --output-dir /Users/Desktop/UC2/Output_py \
  --refresh-dpd --refresh-google-trends --refresh-cpi --refresh-flu

Run with GPU attempt (fallback to CPU if unavailable):
python train_uc2_full_3y.py --data-dir /Users/Desktop/UC2 --output-dir /Users/Desktop/UC2/Output_py --try-gpu

Important CLI options:
- --year-filter none|2023|2021,2022,2023
- --start-yearweek 202101
- --end-yearweek 202352
- --skip-tuning
- --trials-xgb / --trials-lgb / --trials-rf / --trials-cb
- --refresh-dpd --refresh-google-trends --refresh-cpi --refresh-flu

Outputs (in --output-dir):
- section5_baseline_metrics.csv
- section6_metrics_validation.csv
- section6_metrics_test.csv
- tuning_summary.csv (if tuning enabled)
- threshold_optimization_summary.csv
- test_summary.csv
- run_summary.json
- model-specific tuned params / threshold json files

Notes:
- If logs show xgboost=False or catboost=False, those libraries are not installed in the active environment.
- If Optuna is not installed, script uses built-in random-search fallback tuner for Section 7.
- Script can fall back to CPU automatically.
