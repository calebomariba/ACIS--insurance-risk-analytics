# ACIS Insurance Risk Analytics & Predictive Pricing (South Africa)  
**AlphaCare Insurance Solutions – Motor Insurance Analytics Challenge | Dec 2025**

End-to-end risk analytics project to identify low-risk customer segments, statistically validate risk drivers, and build a predictive pricing engine that enables targeted premium reductions for the South African car insurance market.

### Business Goal
Help AlphaCare Insurance Solutions (ACIS) attract profitable new customers by:
- Discovering low-risk provinces, postal codes, driver profiles, and vehicle types
- Proving where statistically significant risk differences exist (and where they don’t)
- Building interpretable ML models for claim prediction and optimal risk-based premiums

### Key Deliverables
- Comprehensive EDA + beautiful, insight-driven visualizations  
- Rigorous A/B hypothesis testing (provinces, postal codes, gender, margins)  
- Reproducible data pipeline with **DVC** (Data Version Control)  
- Claim probability + severity models → risk-based premium framework  
- XGBoost + SHAP explanations for top risk drivers  
- Concrete marketing & pricing recommendations backed by p-values and feature impacts  

### Project Structure
```bash
ACIS-insurance-risk-analytics/
├── data/                  # Raw + versioned datasets (tracked with DVC)
│   ├── raw/
│   │   └── MachineLearningRating_v3.txt.dvc    # tracked by DVC
│   └── processed/                              # future cleaned versions
├── notebooks/             # Exploratory analysis & final visualizations
├── scripts/               # Clean, modular production pipeline
├── reports/               # Interim + final report (PDF/Medium-style)
├── sql/                   # Optional queries & aggregations
├── .dvc/                  # DVC configuration & cache
├── .gitignore
├── requirements.txt
└── README.md
```


### Tech Stack
Python • Pandas • Scikit-learn • XGBoost • SHAP • Matplotlib/Seaborn • Plotly • DVC • GitHub Actions • Jupyter

### Quick Start
```bash
git clone https://github.com/nathanaeldereje/ACIS-insurance-risk-analytics.git
cd ACIS-insurance-risk-analytics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull                   # pulls versioned data
jupyter notebook           # explore notebooks/
```

#### 🔧 Before Running Hypothesis Tests 
Run the following script to generate the analysis_table and summary reports:
```bash
python scripts/task3_prepare_analysis_table.py
dvc add data/processed/analysis_table.parquet
git add reports/task3/summary_counts.csv scripts/task3_prepare_analysis_table.py
git commit -m "task-3: prepared analysis table and group summaries"
dvc push
```
This script:
- validates financial data
- generates data/processed/analysis_table.parquet (tracked by DVC)
- creates summary reports in reports/task3/summary_counts.csv

### Current Progress (as of 7 December 2025)

| Task                                  | Status        | Notes                                                                                   |
|---------------------------------------|---------------|-----------------------------------------------------------------------------------------|
| **Task 1 — EDA & Visualisation**      | ✅ Completed  | 5 elite plots + **National Risk Heatmap (South Africa)** + advanced feature metrics     |
| **Task 2 — DVC Pipeline**             | ✅ Completed  | Fully reproducible → `dvc pull` ready • Local remote configured • Version-tracked data  |
| **Task 3 — Hypothesis Testing**       | ✅ Completed  | Province vs ZIP vs Gender: **P-values, Effect Sizes, Post-hoc tests, Margin analysis**  |
| **Task 4 — Predictive Modeling + SHAP** | ✅ Completed | Random Forest (Severity) + SHAP explainability + model comparison + pricing foundation |


### Pipeline Status
![DVC](https://img.shields.io/badge/DVC-tracked-brightgreen?style=flat&logo=data-version-control) 
![Data Versioned](https://img.shields.io/badge/Data_Versioned-100%25-success) 
![Reproducible](https://img.shields.io/badge/Reproducible-yes-28a745)
