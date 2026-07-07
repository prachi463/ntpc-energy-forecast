import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Power Load Forecasting Dashboard", page_icon="⚡", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("⚡ Load Forecasting")
st.sidebar.markdown(
    "Predicts daily electricity load using historical consumption data. "
    "Built as an AI/ML internship project for power-sector demand forecasting."
)
n_lags = st.sidebar.slider("Number of past days used as input (lag features)", 3, 14, 7)
forecast_horizon = st.sidebar.slider("Days to forecast ahead", 1, 14, 7)
n_estimators = st.sidebar.slider("Random Forest trees (n_estimators)", 50, 300, 150, step=50)

st.title("⚡ Power Load Forecasting Dashboard")
st.caption(
    "AI/ML project — forecasts short-term electricity demand from historical hourly load data, "
    "resampled to daily consumption. Useful for grid planning and generation scheduling."
)

# ---------- Load & prepare data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("events.csv")
    df["Start time UTC"] = pd.to_datetime(df["Start time UTC"])
    df = df.set_index("Start time UTC")
    daily = df["Electricity consumption in Finland"].resample("D").mean().dropna()
    return daily

daily = load_data()

st.subheader("Historical Daily Load")
st.line_chart(daily)

# ---------- Feature engineering: lag features ----------
def make_lag_dataset(series, n_lags):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df["y"]
    return X, y

X, y = make_lag_dataset(daily, n_lags)

split = int(len(X) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ---------- Train model ----------
@st.cache_resource
def train_model(X_train, y_train, n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train, n_estimators)

# ---------- Evaluate ----------
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

col1, col2, col3 = st.columns(3)
col1.metric("Test RMSE (MWh)", f"{rmse:,.1f}")
col2.metric("Test MAE (MWh)", f"{mae:,.1f}")
col3.metric("Training samples", f"{len(X_train):,}")

st.subheader("Actual vs Predicted (Test Period)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name="Actual", mode="lines"))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_test, name="Predicted", mode="lines"))
fig.update_layout(xaxis_title="Date", yaxis_title="Load (MWh)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast future days ----------
st.subheader(f"Forecast: Next {forecast_horizon} Days")

last_known = list(daily.values[-n_lags:])
future_preds = []
current_input = last_known.copy()

for _ in range(forecast_horizon):
    x_input = np.array(current_input[-n_lags:]).reshape(1, -1)
    next_pred = model.predict(x_input)[0]
    future_preds.append(next_pred)
    current_input.append(next_pred)

future_dates = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
forecast_df = pd.DataFrame({"Forecasted Load (MWh)": future_preds}, index=future_dates)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=daily.index[-60:], y=daily.values[-60:], name="Recent Actual", mode="lines"))
fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecasted Load (MWh)"], name="Forecast", mode="lines+markers"))
fig2.update_layout(xaxis_title="Date", yaxis_title="Load (MWh)", legend=dict(orientation="h"))
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(forecast_df.style.format("{:.1f}"))

st.markdown("---")
st.caption(
    "Model: Random Forest Regressor on lagged daily load features. "
    "Dataset: historical hourly electricity consumption, resampled to daily frequency. "
    "This project can be extended to plant-level generation output, feeding it real NTPC "
    "generation/demand data to make it site-specific."
)
