import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as GO
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config for maximum aesthetic appeal
st.set_page_config(
    page_title="Uganda Crop Predictor API", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CUSTOM CSS FOR PREMIUM AESTHETICS
# ==========================================
st.markdown("""
<style>
    /* Main background and typography */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Beautiful headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    
    /* Styling metric cards cleanly */
    div[data-testid="metric-container"] {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(25, 28, 36, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Ensure the button looks great */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4ECDC4 0%, #FF6B6B 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
    }
    div.stButton > button:hover {
        opacity: 0.8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS AND UTILS
# ==========================================
TARGET_CROPS = ['Maize', 'Beans', 'Cassava flour']
CROP_ALIASES = {'Maize (white)': 'Maize'}
TARGET_DISTRICTS = ['Kampala', 'Gulu', 'Mbale']
DISTRICT_REGION = {'Kampala': 'UG3', 'Gulu': 'UG1', 'Mbale': 'UG2'}

LSTM_WINDOW = 6
LSTM_EPOCHS = 100
LSTM_BATCH = 16
LSTM_FEATURES = [
    'price', 'rainfall', 'wb_crude_oil', 'wb_maize',
    'price_lag_1', 'price_lag_2', 'month_sin', 'month_cos', 'rain_lag_1'
]

@st.cache_resource
def get_tf_models():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    return tf, Sequential, LSTM, Dense, Dropout, EarlyStopping

# ==========================================
# 3. DATA PROCESSING & MODELING
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(district, crop):
    wfp = pd.read_csv('wfp_food_prices_uga.csv', parse_dates=['date'])
    rainfall_raw = pd.read_csv('uga-rainfall-subnat-full.csv', parse_dates=['date'])
    
    wb_raw = pd.read_excel('wb_commodity_prices.xlsx', sheet_name='Monthly Prices', header=None)
    wb_headers = wb_raw.iloc[4].tolist()
    wb_headers[0] = 'date_str'
    wb_data = wb_raw.iloc[6:].copy().reset_index(drop=True)
    wb_data.columns = wb_headers
    wb_data['date'] = pd.to_datetime(wb_data['date_str'].astype(str).str.replace('M', '-'), format='%Y-%m', errors='coerce')
    wb = wb_data[['date', 'Crude oil, average', 'Maize']].copy()
    wb.columns = ['date', 'wb_crude_oil', 'wb_maize']
    wb['wb_crude_oil'] = pd.to_numeric(wb['wb_crude_oil'], errors='coerce')
    wb['wb_maize'] = pd.to_numeric(wb['wb_maize'], errors='coerce')
    wb = wb.dropna(subset=['date']).set_index('date').sort_index()

    mask = wfp['commodity'].isin(TARGET_CROPS + list(CROP_ALIASES.keys())) & wfp['admin1'].isin(TARGET_DISTRICTS)
    target_wfp = wfp[mask].copy()
    target_wfp['commodity'] = target_wfp['commodity'].replace(CROP_ALIASES)
    target_wfp['date'] = target_wfp['date'].dt.to_period('M').dt.to_timestamp()
    price_monthly = target_wfp.groupby(['date', 'admin1', 'commodity'])['price'].mean().reset_index()

    rain_reg = rainfall_raw[rainfall_raw['adm_level'] == 1].copy()
    pcode = DISTRICT_REGION[district]
    r = rain_reg[rain_reg['PCODE'] == pcode].set_index('date')['rfh'].resample('MS').mean()

    mask_c = (price_monthly['admin1'] == district) & (price_monthly['commodity'] == crop)
    p = price_monthly[mask_c].set_index('date')[['price']].sort_index()
    if len(p) < 24:
        return None
        
    full_idx = pd.date_range(p.index.min(), p.index.max(), freq='MS')
    p = p.reindex(full_idx)
    p['price'] = p['price'].interpolate(method='linear', limit=3)
    p = p.dropna(subset=['price'])
    p = p.join(r.rename('rainfall'), how='left')
    p = p.join(wb[['wb_crude_oil', 'wb_maize']], how='left')
    p[['rainfall', 'wb_crude_oil', 'wb_maize']] = p[['rainfall', 'wb_crude_oil', 'wb_maize']].ffill().bfill()
    
    p['price_lag_1'] = p['price'].shift(1)
    p['price_lag_2'] = p['price'].shift(2)
    p['month'] = p.index.month
    p['month_sin'] = np.sin(2 * np.pi * p['month'] / 12)
    p['month_cos'] = np.cos(2 * np.pi * p['month'] / 12)
    p['rain_lag_1'] = p['rainfall'].shift(1)
    
    return p.dropna()

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

@st.cache_data(show_spinner=False)
def train_eval_model(_df):
    """Trains 80/20 split to generate historical performance metrics."""
    tf, Sequential, LSTM, Dense, Dropout, EarlyStopping = get_tf_models()
    n = len(_df)
    s = int(n * 0.8)
    train, test = _df.iloc[:s], _df.iloc[s:]
    
    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train[LSTM_FEATURES].values)
    test_sc = scaler.transform(test[LSTM_FEATURES].values)
    
    X_train, y_train = create_sequences(train_sc, LSTM_WINDOW)
    combined = np.vstack([train_sc[-LSTM_WINDOW:], test_sc])
    X_test, y_test = create_sequences(combined, LSTM_WINDOW)
    
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_WINDOW, len(LSTM_FEATURES))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0, callbacks=[es])
    
    y_pred_sc = model.predict(X_test, verbose=0).flatten()
    
    def inv_price(vals):
        dummy = np.zeros((len(vals), len(LSTM_FEATURES)))
        dummy[:, 0] = vals
        return scaler.inverse_transform(dummy)[:, 0]
        
    y_pred_inv = inv_price(y_pred_sc)
    y_test_inv = inv_price(y_test)
    
    return train, test, y_test_inv, y_pred_inv

@st.cache_data(show_spinner=False)
def generate_true_future(_df, horizon):
    """Trains on 100% data and recursively predicts N months into literal future."""
    tf, Sequential, LSTM, Dense, Dropout, EarlyStopping = get_tf_models()
    scaler = MinMaxScaler()
    df_sc = scaler.fit_transform(_df[LSTM_FEATURES].values)
    
    X, y = create_sequences(df_sc, LSTM_WINDOW)
    
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_WINDOW, len(LSTM_FEATURES))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Train heavily on all data
    model.fit(X, y, epochs=70, batch_size=LSTM_BATCH, verbose=0)
    
    future_df = _df.copy()
    
    # Recursive loop into the future
    for i in range(horizon):
        window_raw = future_df.iloc[-LSTM_WINDOW:][LSTM_FEATURES].values
        window_sc = scaler.transform(window_raw)
        
        pred_sc = model.predict(window_sc[np.newaxis, :, :], verbose=0)[0, 0]
        
        dummy = np.zeros((1, len(LSTM_FEATURES)))
        dummy[0, 0] = pred_sc
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        
        last_date = future_df.index[-1]
        next_date = last_date + pd.DateOffset(months=1)
        
        new_row = future_df.iloc[-1].copy()
        new_row.name = next_date
        
        new_row['price'] = pred_price
        new_row['price_lag_1'] = future_df.iloc[-1]['price']
        new_row['price_lag_2'] = future_df.iloc[-2]['price']
        new_row['rain_lag_1']  = future_df.iloc[-1]['rainfall']
        
        month = next_date.month
        new_row['month'] = month
        new_row['month_sin'] = np.sin(2 * np.pi * month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        future_df = pd.concat([future_df, pd.DataFrame([new_row])])
        
    return future_df.iloc[-horizon:]

# ==========================================
# 3b. ARIMA & PROPHET (for Comparison Panel)
# ==========================================
@st.cache_data(show_spinner=False)
def train_eval_arima(_df):
    """Train ARIMA with AIC grid search and evaluate on 80/20 test split."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import itertools
    n = len(_df)
    s = int(n * 0.8)
    train, test = _df.iloc[:s], _df.iloc[s:]
    best_aic, best_order = float('inf'), (1, 1, 1)
    for p in range(4):
        for d in range(2):
            for q in range(4):
                try:
                    m = SARIMAX(train['price'], order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                    r = m.fit(disp=False, maxiter=200)
                    if r.aic < best_aic:
                        best_aic, best_order = r.aic, (p, d, q)
                except Exception:
                    continue
    model = SARIMAX(train['price'], order=best_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)
    preds = result.forecast(steps=len(test))
    return test['price'].values, preds.values, best_order

@st.cache_data(show_spinner=False)
def train_eval_prophet(_df):
    """Train Prophet with rainfall regressor and evaluate on 80/20 test split."""
    from prophet import Prophet
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    n = len(_df)
    s = int(n * 0.8)
    train, test = _df.iloc[:s], _df.iloc[s:]
    train_p = pd.DataFrame({'ds': train.index, 'y': train['price'].values,
                            'rainfall': train['rainfall'].values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False)
    m.add_regressor('rainfall')
    m.fit(train_p)
    future = pd.DataFrame({'ds': test.index, 'rainfall': test['rainfall'].values})
    fc = m.predict(future)
    return test['price'].values, fc['yhat'].values

# ==========================================
# 4. DASHBOARD UI FRAMEWORK
# ==========================================
st.title("🌾 Deep Market Forecaster")
st.markdown("##### Real-time LSTM Price Predictions for Ugandan Staple Crops")
st.markdown("---")

# Wrap exact inputs inside a form so it waits for 'Predict' click
with st.sidebar:
    st.header("🎛️ Parameters")
    with st.form(key="prediction_form"):
        selected_district = st.selectbox("Select District:", TARGET_DISTRICTS)
        selected_crop = st.selectbox("Select Crop:", TARGET_CROPS)
        
        horizon_label = st.radio("Select Forecast Horizon:", options=["4 Weeks Ahead", "8 Weeks Ahead"])
        horizon = 1 if horizon_label == "4 Weeks Ahead" else 2
        
        submit_button = st.form_submit_button(label="Generate Forecast")
        
    st.markdown("---")
    st.markdown("""
    **Model Information:**
    - **Architecture:** Recurrent Neural Network (LSTM)
    - **Methodology:** Recursive Time-Series Generation
    """)

# Wait for submit
if not submit_button:
    st.info("👈 Please select your parameters from the sidebar and click **Generate Forecast** to run the prediction model.")
    st.stop()

# Main execution begins
with st.spinner('Aggregating Historical WFP Market Data & CHIRPS Rainfall...'):
    df_main = load_and_preprocess_data(selected_district, selected_crop)

if df_main is None or len(df_main) < 30:
    st.error(f"Insufficient historical data available for {selected_crop} in {selected_district}. Please select another combination.")
    st.stop()

# 1. Train Evaluation Model
with st.spinner('Validating Neural Network precision on unseen historical data...'):
    train_df, test_df, actual_test, eval_preds = train_eval_model(df_main)

# Calculate Metrics from Eval Model
mae = mean_absolute_error(actual_test, eval_preds)
rmse = math.sqrt(mean_squared_error(actual_test, eval_preds))
mape = np.mean(np.abs((actual_test - eval_preds) / actual_test)) * 100

st.subheader("Model Performance Validation (80/20 Test Split)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Mean Absolute Error (UGX)", value=f"Shs {mae:.1f}")
with col2:
    st.metric(label="Root Mean Squared Error", value=f"{rmse:.1f}")
with col3:
    st.metric(label="Mean Absolute Pct Error (MAPE)", value=f"{mape:.2f}%", delta="- Error Rate", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Train True Future Model
with st.spinner('Training model on 100% of data to generate Unseen Future Trajectory...'):
    future_out = generate_true_future(df_main, horizon)

# Plotting Interactive Chart
st.subheader(f"True Forward Forecasting: {selected_crop} in {selected_district}")

fig = GO.Figure()

# 1. Historical Data (Train) trace
fig.add_trace(GO.Scatter(
    x=train_df.index, 
    y=train_df['price'], 
    mode='lines', 
    name='Historical Data (Train Split)', 
    line=dict(color='#4ECDC4', width=2)
))

# 2. Test Data (Actuals withheld during evaluation)
fig.add_trace(GO.Scatter(
    x=test_df.index[:len(actual_test)], 
    y=actual_test, 
    mode='lines', 
    name='Real Ex-post Prices (Test Actuals)', 
    line=dict(color='#A3A3A3', width=2)
))

# 3. LSTM Model Predictions (Evaluation over Test Split)
fig.add_trace(GO.Scatter(
    x=test_df.index[:len(eval_preds)], 
    y=eval_preds, 
    mode='lines+markers', 
    name='LSTM Test-Set Evaluation', 
    line=dict(color='#FF6B6B', width=2, dash='dash')
))

# 4. True Future Predict Trace (Starting from last historical point)
connector_date = df_main.index[-1]
connector_val = df_main.iloc[-1]['price']
future_x = [connector_date] + list(future_out.index)
future_y = [connector_val] + list(future_out['price'].values)

fig.add_trace(GO.Scatter(
    x=future_x, 
    y=future_y, 
    mode='lines+markers', 
    name='literal Future Forecast (Unseen)', 
    line=dict(color='#f39c12', width=3, dash='longdash'),
    marker=dict(size=8, color='#f1c40f')
))

fig.update_layout(
    xaxis_title='Date Timeline',
    yaxis_title='Price (UGX per kg)',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')

st.plotly_chart(fig, use_container_width=True)

st.success(f"Forecasting complete. The model successfully projected the next {horizon} month(s) into the future using recursive extrapolation.")

# ==========================================
# 5. MODEL COMPARISON PANEL
# ==========================================
st.markdown("---")
st.subheader("📊 Model Comparison Panel: ARIMA vs Prophet vs LSTM")
st.caption("All three models are trained on the same 80/20 chronological split for fair comparison.")

with st.spinner('Training ARIMA baseline (AIC grid search)...'):
    arima_actual, arima_preds, arima_order = train_eval_arima(df_main)

with st.spinner('Training Facebook Prophet with rainfall regressor...'):
    prophet_actual, prophet_preds = train_eval_prophet(df_main)

# Compute metrics for all models
def calc_metrics(actual, predicted):
    m_mae = mean_absolute_error(actual, predicted)
    m_rmse = math.sqrt(mean_squared_error(actual, predicted))
    m_mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
    return round(m_mae, 2), round(m_rmse, 2), round(m_mape, 2)

arima_mae, arima_rmse, arima_mape = calc_metrics(arima_actual, arima_preds)
prophet_mae, prophet_rmse, prophet_mape = calc_metrics(prophet_actual, prophet_preds)

comp_df = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet', 'LSTM'],
    'MAE (UGX)': [arima_mae, prophet_mae, round(mae, 2)],
    'RMSE': [arima_rmse, prophet_rmse, round(rmse, 2)],
    'MAPE (%)': [arima_mape, prophet_mape, round(mape, 2)]
})

best_model = comp_df.loc[comp_df['MAE (UGX)'].idxmin(), 'Model']

col_t, col_b = st.columns([3, 1])
with col_t:
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
with col_b:
    st.metric("🏆 Best Model", best_model)
    st.caption(f"ARIMA order: {arima_order}")

# MAE comparison bar chart
models_list = ['ARIMA', 'Prophet', 'LSTM']
mae_vals = [arima_mae, prophet_mae, round(mae, 2)]
mape_vals = [arima_mape, prophet_mape, round(mape, 2)]
bar_colors = ['#3498db', '#e74c3c', '#2ecc71']

fig_mae = GO.Figure()
fig_mae.add_trace(GO.Bar(
    x=models_list, y=mae_vals,
    marker_color=bar_colors,
    text=[f'{v:,.0f}' for v in mae_vals],
    textposition='auto'
))
fig_mae.update_layout(
    title='Mean Absolute Error (MAE) — Lower is Better',
    yaxis_title='MAE (UGX)',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)
fig_mae.update_xaxes(showgrid=False)
fig_mae.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')

# MAPE comparison bar chart
fig_mape = GO.Figure()
fig_mape.add_trace(GO.Bar(
    x=models_list, y=mape_vals,
    marker_color=bar_colors,
    text=[f'{v:.1f}%' for v in mape_vals],
    textposition='auto'
))
fig_mape.update_layout(
    title='Mean Absolute Percentage Error (MAPE) — Lower is Better',
    yaxis_title='MAPE (%)',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)
fig_mape.update_xaxes(showgrid=False)
fig_mape.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(fig_mae, use_container_width=True)
with chart_col2:
    st.plotly_chart(fig_mape, use_container_width=True)

st.info(f"**Comparison complete.** {best_model} achieves the lowest MAE ({comp_df.loc[comp_df['Model']==best_model, 'MAE (UGX)'].values[0]:,.0f} UGX) "
        f"and MAPE ({comp_df.loc[comp_df['Model']==best_model, 'MAPE (%)'].values[0]:.1f}%) for {selected_crop} in {selected_district}.")

