# ⚡ Power Load Forecasting Dashboard

An AI/ML project that forecasts short-term electricity demand from historical
consumption data — built as part of an NTPC summer internship (AI/ML specialization).

🔗 **Live demo:** https://ntpc-energy-forecast.streamlit.app/

## Overview
Electricity load forecasting is critical for power utilities like NTPC to plan
generation scheduling, manage reserve capacity, and reduce wastage from
over-generation. This project builds a Random Forest regression model on
lagged historical load features to predict daily electricity consumption and
forecast the next N days.

## Features
- Interactive dashboard built with Streamlit
- Historical daily load visualization
- Model evaluation (RMSE, MAE) on a held-out test period
- Actual vs Predicted comparison chart
- Configurable multi-day-ahead forecast
- Adjustable model parameters (lag window, forecast horizon, tree count)

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (Random Forest Regressor)
- Plotly (interactive charts)
- Streamlit (web app + deployment)

## Dataset
Historical hourly electricity consumption data (resampled to daily
frequency). For an NTPC-specific version, replace `events.csv` with actual
plant generation/demand data (same two-column format: timestamp + load value).

## Run Locally
```bash
git clone https://github.com/<your-username>/ntpc-energy-forecast.git
cd ntpc-energy-forecast
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── app.py                 # Streamlit dashboard
├── events.csv              # Historical load dataset
├── requirements.txt        # Python dependencies
└── Machine_Learning_to_predict_Energy_Consumption.ipynb  # Original research notebook (LSTM approach)
```

## Future Improvements
- Swap in real NTPC plant-level generation/demand data
- Add weather features (temperature, humidity) as external regressors
- Compare Random Forest against LSTM/ARIMA baselines
- Add per-plant / per-region forecasting

## Author
Prachi Verma — B.Tech CSE (AI/ML), Axis Institute of Technology and Management
