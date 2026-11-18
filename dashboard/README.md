# CVI Forecasting Dashboard (Streamlit)

## Run it on Google Colab

1. Upload `dashboard/CVI_dashboard_bundle.zip` (about 170 MB) to *My Drive* in Google Drive.
2. Open `dashboard/CVI_Dashboard_Colab.ipynb` in Colab (File → Upload notebook) and choose **Runtime → Run all**.
3. Open the `https://….trycloudflare.com` link that step 4 prints. It stays live while the Colab runtime runs.




## Pages

| Page | What you can do |
|---|---|
| 🏠 Overview | Headline numbers, the two research questions, hypothesis decisions, key findings |
| 🔎 Data Explorer | Dataset summary cards and the train/validation/test timeline; plot sales vs customer activity for any store and date range; data-quality checks; EDA figures |
| 🧮 CVI Explorer | The five CVI components over time for any store, a radar profile against the chain average, the store's segment, composite weights |
| 🔮 **Live Forecast** | Pick a store, date and horizon, then edit the inputs (promotion, holidays, recent sales level, CVI sliders). Both saved models predict live, with a SHAP explanation and the store's full test-period forecast |
| 📊 Model Results | Baseline vs CVI for all learners, the feature-arm ablation, **per-store robustness** (scatter, histogram, win rate, Wilcoxon), a filterable table of all metrics, LSTM/TCN training curves |
| 🧪 Statistical Tests | Bootstrap confidence intervals, Diebold–Mariano, attribution contrasts, rolling-origin CV; detailed tables are collapsed |
| 🧩 Store Segmentation | Cluster profiles and CVI benefit by segment |
| 🛡️ Leakage Audit | All checks (PASS), perturbation test, the oracle "what leakage would look like" bound |
| 🌍 Cross-Retailer | Rossmann vs Favorita for any contrast, and the component-ranking reversal |
| 🎓 Conclusions | Short answers to RQ1 and RQ2, the takeaway, limitations (collapsed) |



The live forecast uses the saved models on the exact test-period features (verified to reproduce the notebook's predictions exactly). 
