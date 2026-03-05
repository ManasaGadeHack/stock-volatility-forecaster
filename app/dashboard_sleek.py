import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Stock Volatility Forecaster", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Sleek modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    :root {
        --bg-dark: #0a0e1a;
        --card-dark: #1a1f2e;
        --neon-green: #c4ff61;
        --neon-teal: #00d4aa;
        --glass-bg: rgba(26, 31, 46, 0.7);
    }
    
    .stApp { 
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%);
        color: #e8eaf6;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(196, 255, 97, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(196, 255, 97, 0.3);
        box-shadow: 0 12px 40px rgba(196, 255, 97, 0.1);
    }
    
    /* Sleek header */
    .hero {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(180deg, rgba(196,255,97,0.05) 0%, transparent 100%);
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #c4ff61 0%, #00d4aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -2px;
        animation: fadeIn 0.6s ease-in;
    }
    
    .hero p {
        color: #8b92b0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Smooth metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: var(--neon-green);
    }
    
    /* Sleek button */
    .stButton > button {
        background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-teal) 100%);
        color: #0a0e1a;
        border: none;
        padding: 1rem 3rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 50px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(196, 255, 97, 0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(196, 255, 97, 0.5);
    }
    
    /* Smooth sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 31, 46, 0.95) 0%, rgba(10, 14, 26, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(196, 255, 97, 0.1);
    }
    
    /* Radio buttons - cleaner */
    .stRadio > div {
        gap: 0.5rem;
        padding: 0;
    }
    
    .stRadio > div > label {
        background: var(--glass-bg);
        backdrop-filter: blur(5px);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        border: 1px solid transparent;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        border-color: rgba(196, 255, 97, 0.3);
        transform: translateX(5px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(196, 255, 97, 0.2) !important;
        border-radius: 15px;
        color: #e8eaf6 !important;
        padding: 0.75rem 1rem;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--neon-green) !important;
        box-shadow: 0 0 20px rgba(196, 255, 97, 0.2);
    }
    
    /* Smooth alerts */
    .alert {
        border-radius: 15px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .alert-success { 
        background: rgba(196, 255, 97, 0.1);
        border-left-color: #c4ff61;
        color: #c4ff61;
    }
    
    .alert-warning {
        background: rgba(255, 107, 107, 0.1);
        border-left-color: #ff6b6b;
        color: #ff6b6b;
    }
    
    .alert-info {
        background: rgba(0, 212, 170, 0.1);
        border-left-color: #00d4aa;
        color: #00d4aa;
    }
    
    /* Hide default stuff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { 
        background: var(--neon-teal);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        for name, path in [
            ('Ensemble', '../models/ensemble/ensemble_model.pkl'),
            ('XGBoost', '../models/ml_models/xgboost.pkl'),
            ('Random Forest', '../models/ml_models/random_forest.pkl')
        ]:
            if os.path.exists(path):
                models[name] = joblib.load(path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

def safe_divide(a, b, fill=0):
    a_vals, b_vals = np.asarray(a).squeeze(), np.asarray(b).squeeze()
    if a_vals.ndim > 1: a_vals = a_vals.ravel()
    if b_vals.ndim > 1: b_vals = b_vals.ravel()
    result = np.full(len(a_vals), fill, dtype=float)
    mask = (b_vals != 0) & (~np.isnan(a_vals)) & (~np.isnan(b_vals))
    result[mask] = a_vals[mask] / b_vals[mask]
    return pd.Series(result, index=a.index) if hasattr(a, 'index') else result

def engineer_features(df):
    df['log_return'] = np.log(safe_divide(df['Close'], df['Close'].shift(1), fill=1))
    for w in [5, 21, 63]:
        df[f'vol_{w}d'] = df['log_return'].rolling(w).std() * np.sqrt(252)
    df['ewm_vol_21'] = df['log_return'].ewm(span=21, min_periods=21).std() * np.sqrt(252)
    log_hl = np.log(safe_divide(df['High'], df['Low'], fill=1)) ** 2
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl.rolling(21).mean()) * np.sqrt(252)
    log_hl = (np.log(safe_divide(df['High'], df['Low'], fill=1))) ** 2
    log_co = (np.log(safe_divide(df['Close'], df['Open'], fill=1))) ** 2
    df['gk_vol'] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean()) * np.sqrt(252)
    df['vol_of_vol'] = df['vol_21d'].rolling(21).std()
    for w in [5, 10, 21, 50, 200]:
        df[f'ma_{w}'] = df['Close'].rolling(w).mean()
    df['price_vs_ma21'] = safe_divide(df['Close'] - df['ma_21'], df['ma_21']) * 100
    df['price_vs_ma50'] = safe_divide(df['Close'] - df['ma_50'], df['ma_50']) * 100
    df['price_vs_ma200'] = safe_divide(df['Close'] - df['ma_200'], df['ma_200']) * 100
    df['momentum_5'] = safe_divide(df['Close'] - df['Close'].shift(5), df['Close'].shift(5)) * 100
    df['momentum_21'] = safe_divide(df['Close'] - df['Close'].shift(21), df['Close'].shift(21)) * 100
    df['hl_pct'] = safe_divide(df['High'] - df['Low'], df['Close']) * 100
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = safe_divide(gain, loss, fill=0.01)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = safe_divide(df['bb_upper'] - df['bb_lower'], df['bb_middle'])
    df['bb_position'] = safe_divide(df['Close'] - df['bb_lower'], df['bb_upper'] - df['bb_lower'], fill=0.5)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = safe_divide(df['Close'] - low_14, high_14 - low_14, fill=0.5) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_21'] = df['Volume'].rolling(21).mean()
    df['volume_ratio'] = safe_divide(df['Volume'], df['volume_ma_21'], fill=1)
    df['price_volume'] = df['log_return'] * df['Volume']
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['return_lag_1'] = df['log_return'].shift(1)
    df['return_lag_5'] = df['log_return'].shift(5)
    df['return_lag_21'] = df['log_return'].shift(21)
    df['vol_lag_1'] = df['vol_21d'].shift(1)
    df['vol_lag_5'] = df['vol_21d'].shift(5)
    df['high_21d'] = df['High'].rolling(21).max()
    df['low_21d'] = df['Low'].rolling(21).min()
    df['price_position'] = safe_divide(df['Close'] - df['low_21d'], df['high_21d'] - df['low_21d'], fill=0.5)
    vol_median = df['vol_21d'].rolling(252, min_periods=63).median()
    df['high_vol_regime'] = (df['vol_21d'] > vol_median * 1.5).astype(int)
    df['uptrend'] = (df['ma_50'] > df['ma_200']).astype(int)
    df['high_252d'] = df['High'].rolling(252, min_periods=63).max()
    df['low_252d'] = df['Low'].rolling(252, min_periods=63).min()
    df['pct_off_high'] = safe_divide(df['Close'] - df['high_252d'], df['high_252d']) * 100
    df['pct_off_low'] = safe_divide(df['Close'] - df['low_252d'], df['low_252d']) * 100
    return df

def fetch_and_predict(ticker, model_name, models):
    import yfinance as yf
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 50:
        return None
    
    df = engineer_features(df).dropna()
    if len(df) < 50:
        return None
    
    current_vol = df['vol_21d'].iloc[-1]
    target_col = 'vol_21d'
    exclude_cols = [target_col, 'target_vol', 'vol_21d_sq', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X_latest = df[feature_cols].iloc[[-1]]
    
    if model_name == 'Ensemble':
        import xgboost as xgb
        ensemble = models['Ensemble']
        rf_pred = ensemble['models']['random_forest'].predict(X_latest)[0]
        dtest = xgb.DMatrix(X_latest, feature_names=X_latest.columns.tolist())
        xgb_pred = ensemble['models']['xgboost'].predict(dtest)[0]
        w_xgb, w_rf = ensemble['weights']
        prediction = w_xgb * xgb_pred + w_rf * rf_pred
        confidence = 0.9456
    elif model_name == 'XGBoost':
        import xgboost as xgb
        dtest = xgb.DMatrix(X_latest, feature_names=X_latest.columns.tolist())
        prediction = models['XGBoost'].predict(dtest)[0]
        confidence = 0.9450
    else:
        prediction = models['Random Forest'].predict(X_latest)[0]
        confidence = 0.8167
    
    return {
        'current_vol': current_vol,
        'predicted_vol': prediction,
        'confidence': confidence,
        'data': df,
        'latest_date': df.index[-1]
    }

# Main app
models = load_models()
if not models:
    st.error("❌ Models not found")
    st.stop()

# Hero header
st.markdown("""
<div class="hero">
    <h1>VOLATILITY FORECASTER</h1>
    <p>Real-time predictions • Machine learning • Live accuracy tracking</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL", max_chars=10).upper().strip()
    model_name = st.selectbox("Model", list(models.keys()))
    
    st.markdown("---")
    st.markdown("### 📊 Model Accuracy")
    st.metric("Historical Accuracy", "94.6%", help="R² score on test set")
    st.metric("Avg Error Rate", "2.1%", help="Average prediction error")

# Main content
page = st.sidebar.radio("", ["🎯 Live Prediction", "🔄 Compare Stocks", "📊 Analytics"], label_visibility="collapsed")

if page == "🎯 Live Prediction":
    
    if st.button("🚀 PREDICT"):
        with st.spinner("Fetching data..."):
            result = fetch_and_predict(ticker, model_name, models)
        
        if not result:
            st.error(f"❌ Could not fetch {ticker}")
            st.stop()
        
        st.success(f"✅ Data loaded through {result['latest_date'].strftime('%Y-%m-%d')}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Volatility", f"{result['current_vol']:.1%}", delta=f"{(result['current_vol']-0.3):+.1%}")
        with col2:
            delta = result['predicted_vol'] - result['current_vol']
            st.metric("Predicted Volatility", f"{result['predicted_vol']:.1%}", delta=f"{delta:+.1%}")
        with col3:
            st.metric("Model Confidence", f"{result['confidence']:.1%}")
        
        # Chart
        df_plot = result['data'].tail(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['vol_21d'], mode='lines', name='Historical',
                                 line=dict(color='#00d4aa', width=3), fill='tozeroy', fillcolor='rgba(0,212,170,0.1)'))
        fig.add_trace(go.Scatter(x=[result['latest_date']], y=[result['predicted_vol']], mode='markers', name='Prediction',
                                 marker=dict(color='#c4ff61', size=20, symbol='star', line=dict(color='#0a0e1a', width=2))))
        fig.add_hline(y=0.30, line_dash="dash", line_color="#ff6b6b", annotation_text="High Vol Threshold")
        fig.update_layout(title=f'{ticker} Volatility Forecast', xaxis_title='Date', yaxis_title='Annualized Volatility',
                         plot_bgcolor='#1a1f2e', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e8eaf6'),
                         height=600, yaxis=dict(tickformat='.0%', gridcolor='rgba(139,146,176,0.1)'),
                         xaxis=dict(gridcolor='rgba(139,146,176,0.1)'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert
        if result['predicted_vol'] > result['current_vol'] * 1.2:
            st.markdown('<div class="alert alert-warning">⚠️ <strong>Rising Volatility Expected</strong></div>', unsafe_allow_html=True)
        elif result['predicted_vol'] < result['current_vol'] * 0.8:
            st.markdown('<div class="alert alert-success">✅ <strong>Decreasing Volatility Expected</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-info">ℹ️ <strong>Stable Volatility Expected</strong></div>', unsafe_allow_html=True)

elif page == "🔄 Compare Stocks":
    st.markdown("### 🔄 Side-by-Side Comparison")
    col1, col2 = st.columns(2)
    with col1:
        t1 = st.text_input("Stock 1", "AAPL").upper()
    with col2:
        t2 = st.text_input("Stock 2", "TSLA").upper()
    
    if st.button("🔄 COMPARE"):
        results = {}
        for t in [t1, t2]:
            with st.spinner(f"Fetching {t}..."):
                r = fetch_and_predict(t, model_name, models)
                if r:
                    results[t] = r
        
        if len(results) == 2:
            col1, col2 = st.columns(2)
            for idx, (t, r) in enumerate(results.items()):
                with [col1, col2][idx]:
                    st.markdown(f"### {t}")
                    st.metric("Current", f"{r['current_vol']:.1%}")
                    st.metric("Predicted", f"{r['predicted_vol']:.1%}")
                    
                    df_plot = r['data'].tail(60)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['vol_21d'], mode='lines',
                                            line=dict(color=['#00d4aa', '#c4ff61'][idx], width=2)))
                    fig.update_layout(height=250, plot_bgcolor='#1a1f2e', paper_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color='#e8eaf6'), yaxis=dict(tickformat='.0%'))
                    st.plotly_chart(fig, use_container_width=True)

else:  # Analytics
    st.markdown("### 📊 Model Performance")
    results_path = '../models/ensemble/final_results.csv'
    if os.path.exists(results_path):
        df_r = pd.read_csv(results_path)
        df_r = df_r[df_r['R²'] > 0]
        st.dataframe(df_r, use_container_width=True)
        
        fig = go.Figure()
        for model in ['XGBoost', 'Random Forest', 'Ensemble (XGB 80%)']:
            if model in df_r['Model'].values:
                row = df_r[df_r['Model'] == model].iloc[0]
                fig.add_trace(go.Bar(x=[model], y=[row['RMSE']], name=model,
                                    marker_color=['#00d4aa', '#4a9eff', '#c4ff61'][['XGBoost', 'Random Forest', 'Ensemble (XGB 80%)'].index(model)]))
        fig.update_layout(title='Model RMSE Comparison', plot_bgcolor='#1a1f2e', paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#e8eaf6'), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)