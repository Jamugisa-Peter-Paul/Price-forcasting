# Market Price Forecasting for Ugandan Staple Crops

A Multi-Model Time-Series Approach Using WFP Market Data and Climate Covariates.

**Course:** Artificial Intelligence and Machine Learning  
**Programme:** Master of Science in Computer Science (Year 1, Semester 2)  
**Institution:** Uganda Christian University (UCU)

---

## Problem Statement

Price volatility for staple crops — maize, beans, and cassava — causes food insecurity and income shocks for smallholder farmers and traders across Uganda. This project builds and compares three machine learning forecasting models (**ARIMA**, **Facebook Prophet**, and **LSTM**) trained on historical WFP market price data, augmented with CHIRPS rainfall data and World Bank commodity prices as external covariates.

**Research Question:** *How accurately can machine learning models forecast staple crop market prices 4 to 8 weeks ahead in Uganda, and which model achieves the best performance when rainfall is incorporated as a predictor?*

## Key Results

| Model | Avg MAE (UGX) | Avg RMSE | Avg MAPE |
|-------|---------------|----------|----------|
| ARIMA | 757.45 | 882.33 | 25.11% |
| Prophet | 841.27 | 969.86 | 29.66% |
| **LSTM** | **503.13** | **606.40** | **17.65%** |

The LSTM neural network achieved the best performance across all metrics, with MAPE as low as 11% in some districts.

## Project Structure

```
├── Market_Price_Forecasting.ipynb   # Full analysis notebook (EDA → Models → Evaluation)
├── app.py                           # Streamlit interactive dashboard
├── wfp_food_prices_uga.csv          # WFP Uganda food price data
├── uga-rainfall-subnat-full.csv     # CHIRPS satellite rainfall data
├── wb_commodity_prices.xlsx         # World Bank commodity price indices
├── wfp_markets_uga.csv              # WFP markets lookup table
├── Project_Cheatsheet_Presentation.tex  # Presentation script (LaTeX)
├── Project_Cheatsheet_Presentation.pdf  # Compiled presentation cheatsheet
├── AI_and_ML_Proposal_3.txt         # Original project proposal
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Datasets

| Source | Description | File |
|--------|-------------|------|
| [WFP Food Prices (HDX)](https://data.humdata.org/dataset/wfp-food-prices-for-ug) | Weekly/monthly crop prices in UGX across Ugandan districts | `wfp_food_prices_uga.csv` |
| [CHIRPS Rainfall (UC Santa Barbara)](https://www.chc.ucsb.edu/data/chirps) | Monthly rainfall in mm per sub-national region | `uga-rainfall-subnat-full.csv` |
| [World Bank Commodity Markets](https://www.worldbank.org/en/research/commodity-markets) | Global crude oil and maize price indices | `wb_commodity_prices.xlsx` |

## Models Implemented

1. **ARIMA (SARIMAX)** — Classical statistical baseline with AIC/BIC grid search for optimal (p,d,q) order.
2. **Facebook Prophet** — Additive decomposition model with rainfall as an external regressor.
3. **LSTM (Deep Learning)** — Two-layer recurrent neural network (64→32 units, dropout=0.2, Adam optimizer) trained on sliding-window sequences using TensorFlow/Keras.

All models use an **80/20 chronological train/test split** and are evaluated across **3 crops × 3 districts**.

## Streamlit Dashboard

The interactive dashboard allows users to:
- Select a **crop** (Maize, Beans, Cassava flour)
- Select a **district** (Kampala, Gulu, Mbale)
- Choose a **forecast horizon** (4 or 8 weeks ahead)
- View predicted price trends alongside historical data
- Compare ARIMA, Prophet, and LSTM performance side-by-side

### Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Feature Engineering

- **Lag features:** price at t-1, t-2, t-4, t-8
- **Rolling statistics:** 4-week and 8-week mean/std
- **Seasonal indicators:** month sin/cos encoding, harvest season dummies
- **Climate covariates:** rainfall lags (t-1, t-4, t-8)
- **Global covariates:** World Bank crude oil and maize price indices

## Scope

- **Crops:** Maize, Beans, Cassava flour
- **Districts:** Kampala, Gulu, Mbale
- **Time period:** 2010–2024

## Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | pandas, NumPy |
| Statistical ML | statsmodels (ARIMA/SARIMAX) |
| Additive ML | Facebook Prophet |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Visualisation | matplotlib, seaborn, Plotly |
| Dashboard | Streamlit |
| Version Control | GitHub |

## References

- Taylor, S.J. & Letham, B. (2018). *Forecasting at scale.* The American Statistician, 72(1), 37-45.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735-1780.
