# ⚡ Power Load Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.9-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Live-success)

🚀 End-to-end machine learning system developed during my NTPC internship to forecast short-term electricity demand using historical consumption data.

🔗 **Live Demo:** https://ntpc-energy-forecast.streamlit.app/

---

## 📌 Problem Statement

Accurate electricity demand forecasting is critical for large-scale power utilities like NTPC to:

- ⚡ Optimize generation scheduling  
- 📉 Reduce overproduction and energy wastage  
- 🔌 Maintain grid stability and reliability  

This project models short-term electricity demand as a supervised learning problem using historical time-series data.

---

## 🧠 Key Highlights

- Built a **Random Forest regression model** for time-series forecasting  
- Engineered **lag-based features** to capture temporal dependencies  
- Evaluated model using **RMSE and MAE on hold-out test data**  
- Developed an **interactive Streamlit dashboard** for real-time predictions  
- Enabled **multi-day forecasting with dynamic user controls**  

---

## 🏗️ System Architecture
                    ┌──────────────────────────────┐
                    │        User Interface        │
                    │      (Streamlit Dashboard)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        User Inputs           │
                    │  - Forecast Horizon (Days)   │
                    │  - Lag Window Size           │
                    │  - Model Parameters          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Data Preprocessing      │
                    │  - Load CSV Data             │
                    │  - Handle Missing Values     │
                    │  - Resample (Hourly → Daily) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Feature Engineering Layer  │
                    │  - Lag Feature Creation      │
                    │  - Time-based Features       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Machine Learning Model     │
                    │  Random Forest Regressor     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Model Evaluation         │
                    │   - RMSE                     │
                    │   - MAE                      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Forecast Generation        │
                    │  - Multi-step Prediction     │
                    │  - Future Load Estimation    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Visualization Layer      │
                    │  - Actual vs Predicted Graph │
                    │  - Forecast Plot             │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Final Output          │
                    │   Interactive Insights       │
                    │   + Future Predictions       │
                    └──────────────────────────────┘  

---

## 🤖 Model Details

- **Algorithm:** Random Forest Regressor  
- **Approach:** Converted time-series data into supervised format using lag features  

### Why Random Forest?

- Captures **non-linear relationships** effectively  
- Robust to noise in real-world energy data  
- Requires minimal hyperparameter tuning  
- Performs well on structured/tabular datasets  

---

## 📈 Features

- 📊 Historical load visualization  
- 🔍 Actual vs Predicted comparison  
- ⚙️ Adjustable parameters:
  - Lag window  
  - Forecast horizon  
  - Number of trees  
- 📅 Multi-day forecasting capability  

---

## 🛠️ Tech Stack

- **Programming:** Python (Pandas, NumPy)  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Visualization:** Plotly  
- **Deployment:** Streamlit  

---

## 📂 Dataset

- Historical electricity consumption data  
- Resampled from hourly → daily frequency  
- Format: `timestamp | load value`  

---

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/ntpc-energy-forecast.git
cd ntpc-energy-forecast
pip install -r requirements.txt
streamlit run app.py

---

## 📁 Project Structure
├── app.py
├── events.csv
├── requirements.txt
├── Machine_Learning_to_predict_Energy_Consumption.ipynb

---

🔮 Future Improvements
Compare with LSTM / ARIMA models
Integrate weather-based features
Deploy as a scalable REST API
Add region-wise / plant-wise forecasting

---
👩‍💻 Author

Prachi Verma
B.Tech CSE (AI/ML)

