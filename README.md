# Retail Sales Forecasting & Business Performance Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-DAX-yellow)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end retail analytics solution: from raw transactional data to demand forecasts and an executive Power BI dashboard.

---

## 🎯 Problem Statement

Retail chains need accurate sales forecasts to optimize inventory, plan promotions, and allocate staff. This project transforms historical transaction data into:
- **Demand forecasts** (next-period sales predictions)
- **Performance insights** (revenue, profit, discount, seasonality)
- **Actionable recommendations** for inventory and promotional planning

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw CSV    │────▶│  Data Pipeline   │────▶│  Clean Dataset  │
│  (Superstore)│    │  (standardize,   │     │  (9,994 rows,   │
│             │    │   merge, clean,   │     │   28 features)  │
└─────────────┘     │   feature eng.)  │     └────────┬────────┘
                    └────────┬─────────┘              │
                             │                        │
              ┌──────────────┼──────────────┐        │
              ▼              ▼              ▼        ▼
        ┌──────────┐  ┌────────────┐  ┌────────────┐ ┌────────┐
        │   EDA    │  │ Statistics │  │  Modeling  │ │ PowerBI│
        │ (charts) │  │ (IQR, etc) │  │ (4 models) │ │ (DAX)  │
        └──────────┘  └────────────┘  └─────┬────────┘ └────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
            ┌───────────────┐        ┌────────────────┐       ┌───────────────┐
            │ Forecasts     │        │ Error Analysis │       │ Business      │
            │ (test set)    │        │ by store/cat   │       │ Action Report │
            └───────────────┘        └────────────────┘       └───────────────┘
```

---

## 📊 Key Results

| Model | RMSE | WMAPE | R² | Rank |
|-------|------|-------|-----|------|
| **Neural Network (MLP)** | **451.95** | **83.1%** | **0.288** | 🥇 |
| Linear Regression | 456.19 | 89.6% | 0.274 | 🥈 |
| Random Forest | 460.03 | 85.8% | 0.262 | 🥉 |
| Decision Tree | 494.35 | 87.7% | 0.148 | 4th |

> **Best Model**: Neural Network (64→32 hidden layers, early stopping)
> **Split**: Chronological 80/20 (last 20% of unique dates held out — **no future leakage**)

---

## 📁 Project Structure

```
Retail_Sales_Forecasting_Project/
├── data/
│   ├── raw/
│   │   └── superstore_sales.csv          # Source data (9,994 rows)
│   ├── processed/
│   │   ├── clean_sales.csv               # Cleaned dataset
│   │   ├── model_comparison.csv          # Model metrics
│   │   ├── test_forecasts.csv            # Held-out predictions
│   │   ├── forecast_error_by_store.csv   # Error by state (48)
│   │   └── forecast_error_by_category.csv # Error by category (3)
│   ├── figures/
│   │   ├── monthly_revenue.svg           # Monthly trend (chronological)
│   │   ├── category_revenue.svg          # Revenue by category
│   │   └── store_revenue.svg             # Revenue by state proxy
│   └── business_action_report.md         # Executive summary
├── notebooks/
│   └── retail_analytics_eda.ipynb        # Interactive EDA
├── powerbi/
│   ├── measures.dax                      # DAX measures (Revenue, Profit Margin, AOV, Growth)
│   └── README.md                         # Dashboard design guide
├── src/
│   └── retail_analytics.py               # Main pipeline (CLI)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/Shannu320/RETAIL-SALES-FORECASTING.git
cd Retail_Sales_Forecasting_Project
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python src/retail_analytics.py --input data/raw --output data/processed
```

**Outputs generated:**
- `data/processed/clean_sales.csv` — 9,994 cleaned transactions with 28 engineered features
- `data/processed/model_comparison.csv` — Model performance metrics
- `data/processed/forecast_error_by_store.csv` — MAPE/MAE by state (48)
- `data/processed/forecast_error_by_category.csv` — MAPE/MAE by category (3)
- `data/figures/*.svg` — EDA visualizations
- `data/business_action_report.md` — Executive action report

### Explore Interactively
```bash
pip install jupyter
jupyter notebook notebooks/retail_analytics_eda.ipynb
```

---

## 📈 Power BI Dashboard

1. Open **Power BI Desktop**
2. **Get Data → Text/CSV** → Import `data/processed/clean_sales.csv` as **FactSales**
3. Create Calendar table using DAX from `powerbi/measures.dax`
4. Build 3 pages per `powerbi/README.md`:
   - **Executive Overview**: Revenue, Profit, Profit Margin, Orders, AOV, YoY Growth
   - **Product & Location**: Category contribution, top products, state comparison, discount vs profit
   - **Forecast Quality**: Actual vs Predicted, MAPE by category, MAPE by state

---

## 🔬 Methodology

### Data Pipeline
- **Column standardization**: Fuzzy alias matching (e.g., "Order Date" → `order_date`)
- **Multi-file merge**: Joins on all shared `_id` columns
- **Cleaning**: Type coercion, missing value imputation (median/mode), outlier clipping
- **Feature engineering**: Year, month, quarter, day-of-week, weekend flag, order_year_month
- **Store proxy**: Derived from `state` (Superstore lacks true store dimension)

### Modeling
- **Time-aware split**: Last 20% of unique dates → test set (prevents leakage)
- **Features**: `quantity`, `discount`, date parts, `category`, `sub_category`, `segment`, `region`, `store`, `ship_mode`
- **Preprocessing**: Median imputation (numeric), OneHotEncoder (categorical), StandardScaler (NN only)
- **Models**: Linear Regression, Decision Tree, Random Forest, MLPRegressor
- **Metrics**: MAE, RMSE, MAPE, WMAPE, R²

---

## ⚠️ Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **High WMAPE (83%)** | Forecasts have high uncertainty; directional use only | Add external features (holidays, weather, promo calendar) |
| **No true store dimension** | State used as proxy; masks intra-state variation | Integrate store master data when available |
| **No external features** | Misses holiday spikes, weather effects, marketing events | Enrich with calendar, weather, promotion datasets |
| **Single dataset** | Results specific to Superstore (2015–2018) | Validate on client data before deployment |
| **Batch-only** | No streaming/incremental retraining | Add scheduled retraining + model monitoring |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, custom SVG |
| **ML** | scikit-learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, MLPRegressor) |
| **BI** | Power BI, DAX |
| **Notebook** | Jupyter |
| **CLI** | argparse, pathlib |

---

## 📋 Requirements

```txt
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
```

Install: `pip install -r requirements.txt`

---

## 🧪 Testing (Optional)

```bash
pip install pytest
pytest tests/ -v
```

---

## 📄 License

MIT License — feel free to use for learning, portfolio, or commercial projects.

---

## 📚 References

- Dataset: [Sample Superstore](https://github.com/leonism/sample-superstore) (public retail transactions)
- Time-series CV: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- DAX Patterns: [SQLBI](https://www.sqlbi.com/articles/dax-patterns/)