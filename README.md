# A Leakage-Aware Framework for Evaluating Customer-Activity Features in Multi-Horizon Retail Sales Forecasting

**Predictive contribution, feature-construction effects, and cross-retailer generalizability**

Syeda Fatima Naqvi · Laiba Arshad · Shajia Tahir — Department of Computer Science, FAST-NUCES Islamabad

---

## Overview

Customer counts are the strongest same-day correlate of store sales, but they are recorded *together with* the sales they are meant to predict. A careless customer feature therefore leaks the future, and even a leak-free feature may help only because of how it summarises history, not because customer data adds anything.

This project builds a **leakage-aware evaluation framework** and applies it to a five-component **Customer Value Index (CVI)**: activity level, spending proxy, stability, momentum and promotion responsiveness. It evaluates Ridge, Random Forest, LightGBM, LSTM and TCN models at horizons of 1, 7, 14 and 28 days on two retailers:

| Dataset | Stores | Activity measure | Test period |
|---|---|---|---|
| Rossmann Store Sales | 1,115 | customer counts | Feb–Jul 2015 |
| Corporación Favorita (Store Sales) | 54 | transactions (a proxy, not customers) | Jan–Aug 2017 |

### Research questions

- **RQ1 – Predictive contribution and attribution:** do leakage-free customer-activity features reduce forecasting error versus a sales-only baseline, and is the gain due to customer information, the feature-construction recipe, or the composite index?
- **RQ2 – Generalizability and profiling:** do these effects, and the behavioural store profiles built from the same features, hold across retailers with different activity measures?

### Key results

| Finding | Evidence |
|---|---|
| CVI features improve forecasts on Rossmann | LightGBM MAE −3.1 % to −7.1 % at h = 7, 14, 28 (Holm-adjusted p ≤ 0.011); 77–95 % of stores improve |
| …but mainly through **feature construction** | A sales-only analogue built with the same recipe performs as well |
| Components beat the composite index | Components better by 1.1–4.8 % of baseline MAE |
| Gain is smallest one day ahead and season-dependent | Opposite of H3; no gain in the Dec–Jan CV fold |
| Does **not** transfer to Favorita | No significant gain; raw transaction history helps (up to 5.3 %); the component ranking reverses (ρ = −0.90) |
| The pipeline is leakage-free | 17.2 million feature values recomputed after corrupting the future, with 0 mismatches |

### Hypotheses (pre-specified) and final decisions

| Hypothesis | Final decision |
|---|---|
| H1: CVI reduces forecasting error | Partially supported: supported on Rossmann (h ≥ 7), not on Favorita |
| H2: individual components perform at least as well as the composite | Supported on Rossmann; inconclusive on Favorita |
| H3: CVI gives stronger improvements at shorter horizons | Rejected on Rossmann (opposite direction); inconclusive on Favorita |
| H4: CVI benefits differ across store segments | Differences on Rossmann, but not uniquely attributable to CVI |
| H5: results are consistent across datasets | Not supported |
| H6: CVI produces meaningful store profiles | Limited, partial evidence |

---

## Repository structure

```
DS-PROJECT/
├── README.md                                   ← you are here
│
├── Notebook/
│   └── CVI_Retail_Forecasting.ipynb   ← MAIN NOTEBOOK: full study with all outputs, figures and interpretations
│
├── dashboard/
│   ├── CVI_Dashboard_Colab.ipynb               ← runs the Streamlit dashboard on Google Colab and gives a public link
│   ├── README.md                               ← dashboard pages and a 5-minute demo guide
│   └── CVI_dashboard_bundle.zip                ← dashboard code + data + saved models (163 MB, hosted on Google Drive, see below)
│
├── demo-video/
│   └── streamlit-app-2026-11-18-15-42-02.webm  ← screen recording of the dashboard demo
│
├── cvi-outputs/                                ← all figures produced by the notebook (300 dpi PNG)
│   ├── rossmann/figures/                        ← 48 Rossmann figures (rossmann_fig01 … fig43)
│   └── favorita/figures/                       ← 47 Favorita figures (favorita_fig01 … fig42)
│
└── data/                                       ← raw datasets (Kaggle)
    ├── rossmann/
    │   ├── train.csv                           ← daily sales and customers, 1,115 stores, 2013–2015
    │   └── store.csv                           ← store attributes
    └── favorita/
        ├── train.csv.zip                       ← daily unit sales by store and product family, 2013–2017
        ├── stores.csv                          ← store attributes
        ├── transactions.csv.zip                ← daily transactions per store (customer-activity proxy)
        └── holidays_events.csv                 ← national, regional and local holidays
```

### Where to find what

| I want to see… | Open |
|---|---|
| The full analysis with every output | `Notebook/CVI_Retail_Forecasting_executed.ipynb` |
| The demo | `demo-video/` |
| The interactive dashboard | `dashboard/` (see **How to run → Dashboard**) |
| All figures | `cvi-outputs/rossman/figures/` and `cvi-outputs/favorita/figures/` |
| The datasets | `data/` |

### Figure guide (same numbering for both datasets)

| Figures | Content |
|---|---|
| fig01–fig06 | Data quality (missing values, outliers), sales and activity distributions, trends, store comparison |
| fig07–fig12 | Temporal patterns: weekday, monthly, yearly and seasonal profiles, autocorrelation, store time series |
| fig13–fig18 | Sales vs customer activity: scatter, correlation heatmap, promotions, sales per customer, rolling correlation |
| fig19–fig22 | Advanced EDA: high vs low activity stores, store quadrants, stability, activity-change lead/lag |
| fig23–fig26 | CVI components: distributions, correlations, composite index and weights, example store |
| fig27–fig28 | LSTM and TCN training and validation loss / MAE curves (early stopping) |
| fig29–fig37 | Results: actual vs predicted, performance by horizon, feature-arm comparison, RMSE, residuals, feature importance, SHAP, single-component ablation, rolling-origin CV |
| fig38–fig41 | Store segmentation: choosing k, PCA view, cluster CVI profiles, performance by segment |
| fig42 | Bootstrap confidence intervals for every comparison |
| fig43 (Rossmann folder) | Cross-retailer comparison (Rossmann vs Favorita) |

---

## How to run

### Notebook

- **Just read it:** open `Notebook/CVI_Retail_Forecasting_executed.ipynb`; every output is saved in it. If GitHub cannot render it (10 MB), paste its URL into <https://nbviewer.org>.
- **Re-run it:** install the requirements, put the `data/` folder next to the notebook, and run all cells. Paths are configured in Section 2 of the notebook.

```bash
pip install pandas numpy scikit-learn lightgbm torch shap statsmodels seaborn pyarrow
```

A full run takes about 5 hours on an 8-core CPU. Setting `CVI_RUN_MODE=fast` runs a 15-minute smoke test.

### Dashboard

The bundle (163 MB) is above GitHub's 100 MB file limit, so it is hosted on Google Drive:
**Download `CVI_dashboard_bundle.zip`: https://drive.google.com/file/d/1k7DJVzzUe4h5vV1NLm_YUE_xOfttlKRr/view?usp=sharing**

- **On Google Colab:** put the zip in *My Drive*, open `dashboard/CVI_Dashboard_Colab.ipynb` in Colab, choose **Runtime → Run all**, then open the `https://…trycloudflare.com` link it prints.
- **On your computer:** unzip the bundle, open a terminal inside the `cvi_dashboard` folder, and run:

```bash
pip install streamlit plotly lightgbm pandas pyarrow scipy
```
```bash
python -m streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>.

| Page | What it shows |
|---|---|
| Overview | Research questions, headline results, hypothesis decisions |
| Data Explorer | Sales vs customer activity for any store, data quality, EDA figures |
| CVI Explorer | The five CVI components over time and the store's profile |
| **Live Forecast** | Change the inputs (promotion, holidays, sales level, CVI) and both saved models predict live, with a SHAP explanation |
| Model Results | Baseline vs CVI for every model, feature ablation, per-store robustness, training curves |
| Statistical Tests | Bootstrap confidence intervals, Diebold–Mariano, attribution contrasts |
| Segmentation · Leakage Audit · Cross-Retailer · Conclusions | Store profiles, the leakage audit, Rossmann vs Favorita, answers to the RQs |

---

## Methodology in brief

- **No leakage by construction:** every feature comes from an *as-of state* that closes at the forecast origin *o = t − h*. An executable audit verifies this: perturbation invariance, a power check, provenance assertions and a shuffled-target test.
- **Feature arms:**
  - A: baseline (sales history, calendar, store)
  - B: A + CVI components
  - C: A + composite CVI
  - D: A + components and composite (CVI-enhanced)
  - E: A + raw activity history
  - P: A + the same recipe applied to sales only (control)
  - ORACLE: A + same-day activity (leakage upper bound only)
- **Models:** Ridge, Random Forest, LightGBM (primary, tuned once and frozen), LSTM and TCN (early stopping, best-epoch checkpoints). Each model is compared with its own baseline.
- **Evaluation:** chronological train / validation / test splits, scored once on test. MAE is the primary metric. Inference uses a paired moving-block bootstrap, the Diebold–Mariano (HLN) test, Holm correction, per-store Wilcoxon tests and rolling-origin cross-validation.
- **Reproducibility:** fixed seeds (42); one executed notebook produces every number in the paper and the dashboard.

## Data sources

- Rossmann Store Sales: <https://www.kaggle.com/competitions/rossmann-store-sales>
- Store Sales – Time Series Forecasting (Corporación Favorita): <https://www.kaggle.com/competitions/store-sales-time-series-forecasting>

The data is included for academic review only and remains subject to the Kaggle competition rules.

## Citation

```
S. F. Naqvi, L. Arshad, S. Tahir, "A Leakage-Aware Framework for Evaluating Customer-Activity Features in
Multi-Horizon Retail Sales Forecasting: Predictive Contribution, Feature-Construction Effects, and
Cross-Retailer Generalizability," FAST-NUCES Islamabad, 2026.
```
