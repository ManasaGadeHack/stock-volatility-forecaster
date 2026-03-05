import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Stock Volatility Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS with neon accents
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --bg-dark: #0a0e1a;
        --card-dark: #1a1f2e;
        --neon-green: #c4ff61;
        --neon-teal: #00d4aa;
        --neon-coral: #ff6b6b;
        --neon-blue: #4a9eff;
        --text-light: #e8eaf6;
        --text-muted: #8b92b0;
    }
    
    /* Global background */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-light);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0a0e1a 100%);
        border-right: 1px solid rgba(196, 255, 97, 0.1);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-light);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a3f5f 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(196, 255, 97, 0.2);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .dashboard-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--card-dark);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--neon-teal);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.2);
    }
    
    .metric-card.green {
        border-left-color: var(--neon-green);
    }
    
    .metric-card.coral {
        border-left-color: var(--neon-coral);
    }
    
    .metric-card.blue {
        border-left-color: var(--neon-blue);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-teal) 100%);
        color: #0a0e1a;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(196, 255, 97, 0.4);
    }
    
    /* Alert boxes */
    .alert-success {
        background: rgba(196, 255, 97, 0.1);
        border-left: 4px solid var(--neon-green);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: var(--neon-green);
    }
    
    .alert-warning {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid var(--neon-coral);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: var(--neon-coral);
    }
    
    .alert-info {
        background: rgba(74, 158, 255, 0.1);
        border-left: 4px solid var(--neon-blue);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: var(--neon-blue);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: var(--card-dark);
        color: var(--text-light);
        border: 1px solid rgba(196, 255, 97, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stSelectbox > div > div > div {
        background: var(--card-dark);
        color: var(--text-light);
        border: 1px solid rgba(196, 255, 97, 0.3);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: var(--card-dark);
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--card-dark);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-teal) 100%);
        color: #0a0e1a;
        font-weight: 700;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neon-teal);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neon-green);
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        ensemble_path = '../models/ensemble/ensemble_model.pkl'
        if os.path.exists(ensemble_path):
            models['Ensemble'] = joblib.load(ensemble_path)
        
        xgb_path = '../models/ml_models/xgboost.pkl'
        if os.path.exists(xgb_path):
            models['XGBoost'] = joblib.load(xgb_path)
        
        rf_path = '../models/ml_models/random_forest.pkl'
        if os.path.exists(rf_path):
            models['Random Forest'] = joblib.load(rf_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models

def safe_divide(a, b, fill=0):
    """Safe division that never produces inf/nan errors"""
    a_vals = np.asarray(a).squeeze()
    b_vals = np.asarray(b).squeeze()
    if a_vals.ndim > 1:
        a_vals = a_vals.ravel()
    if b_vals.ndim > 1:
        b_vals = b_vals.ravel()
    result = np.full(len(a_vals), fill, dtype=float)
    mask = (b_vals != 0) & (~np.isnan(a_vals)) & (~np.isnan(b_vals))
    result[mask] = a_vals[mask] / b_vals[mask]
    if hasattr(a, 'index'):
        return pd.Series(result, index=a.index)
    return result

def engineer_features(df):
    """Engineer all 56 features - simplified version"""
    
    # Log returns
    df['log_return'] = np.log(safe_divide(df['Close'], df['Close'].shift(1), fill=1))
    
    # Volatility features
    for w in [5, 21, 63]:
        df[f'vol_{w}d'] = df['log_return'].rolling(w).std() * np.sqrt(252)
    
    df['ewm_vol_21'] = df['log_return'].ewm(span=21, min_periods=21).std() * np.sqrt(252)
    
    # Parkinson volatility
    log_hl = np.log(safe_divide(df['High'], df['Low'], fill=1)) ** 2
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl.rolling(21).mean()) * np.sqrt(252)
    
    # Garman-Klass volatility
    log_hl = (np.log(safe_divide(df['High'], df['Low'], fill=1))) ** 2
    log_co = (np.log(safe_divide(df['Close'], df['Open'], fill=1))) ** 2
    df['gk_vol'] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean()) * np.sqrt(252)
    
    df['vol_of_vol'] = df['vol_21d'].rolling(21).std()
    
    # Price features
    for w in [5, 10, 21, 50, 200]:
        df[f'ma_{w}'] = df['Close'].rolling(w).mean()
    
    df['price_vs_ma21'] = safe_divide(df['Close'] - df['ma_21'], df['ma_21']) * 100
    df['price_vs_ma50'] = safe_divide(df['Close'] - df['ma_50'], df['ma_50']) * 100
    df['price_vs_ma200'] = safe_divide(df['Close'] - df['ma_200'], df['ma_200']) * 100
    
    df['momentum_5'] = safe_divide(df['Close'] - df['Close'].shift(5), df['Close'].shift(5)) * 100
    df['momentum_21'] = safe_divide(df['Close'] - df['Close'].shift(21), df['Close'].shift(21)) * 100
    
    df['hl_pct'] = safe_divide(df['High'] - df['Low'], df['Close']) * 100
    
    # Technical indicators - RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = safe_divide(gain, loss, fill=0.01)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = safe_divide(df['bb_upper'] - df['bb_lower'], df['bb_middle'])
    df['bb_position'] = safe_divide(df['Close'] - df['bb_lower'], df['bb_upper'] - df['bb_lower'], fill=0.5)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = safe_divide(df['Close'] - low_14, high_14 - low_14, fill=0.5) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Volume features
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_21'] = df['Volume'].rolling(21).mean()
    df['volume_ratio'] = safe_divide(df['Volume'], df['volume_ma_21'], fill=1)
    df['price_volume'] = df['log_return'] * df['Volume']
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Lag features
    df['return_lag_1'] = df['log_return'].shift(1)
    df['return_lag_5'] = df['log_return'].shift(5)
    df['return_lag_21'] = df['log_return'].shift(21)
    df['vol_lag_1'] = df['vol_21d'].shift(1)
    df['vol_lag_5'] = df['vol_21d'].shift(5)
    
    # Price position
    df['high_21d'] = df['High'].rolling(21).max()
    df['low_21d'] = df['Low'].rolling(21).min()
    df['price_position'] = safe_divide(df['Close'] - df['low_21d'], df['high_21d'] - df['low_21d'], fill=0.5)
    
    # Regime features
    vol_median = df['vol_21d'].rolling(252, min_periods=63).median()
    df['high_vol_regime'] = (df['vol_21d'] > vol_median * 1.5).astype(int)
    df['uptrend'] = (df['ma_50'] > df['ma_200']).astype(int)
    
    df['high_252d'] = df['High'].rolling(252, min_periods=63).max()
    df['low_252d'] = df['Low'].rolling(252, min_periods=63).min()
    df['pct_off_high'] = safe_divide(df['Close'] - df['high_252d'], df['high_252d']) * 100
    df['pct_off_low'] = safe_divide(df['Close'] - df['low_252d'], df['low_252d']) * 100
    
    return df

def fetch_stock_data(ticker):
    """Fetch real-time stock data"""
    import yfinance as yf
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    
    if df.empty or len(df) < 50:
        return None
    
    return df

def make_prediction(df, model_name, models):
    """Make volatility prediction"""
    
    df = engineer_features(df)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    current_vol = df['vol_21d'].iloc[-1]
    
    target_col = 'vol_21d'
    exclude_cols = [target_col, 'target_vol', 'vol_21d_sq', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
    
    X_latest = df[feature_cols].iloc[[-1]]
    
    if model_name == 'Ensemble':
        import xgboost as xgb
        ensemble = models['Ensemble']
        rf_model = ensemble['models']['random_forest']
        xgb_model = ensemble['models']['xgboost']
        w_xgb, w_rf = ensemble['weights']
        
        rf_pred = rf_model.predict(X_latest)[0]
        dtest = xgb.DMatrix(X_latest, feature_names=X_latest.columns.tolist())
        xgb_pred = xgb_model.predict(dtest)[0]
        
        prediction = w_xgb * xgb_pred + w_rf * rf_pred
        confidence = 0.9456
        
    elif model_name == 'XGBoost':
        import xgboost as xgb
        model = models['XGBoost']
        dtest = xgb.DMatrix(X_latest, feature_names=X_latest.columns.tolist())
        prediction = model.predict(dtest)[0]
        confidence = 0.9450
        
    else:
        model = models['Random Forest']
        prediction = model.predict(X_latest)[0]
        confidence = 0.8167
    
    return {
        'current_vol': current_vol,
        'predicted_vol': prediction,
        'confidence': confidence,
        'data': df,
        'latest_date': df.index[-1]
    }

# Load models
models = load_models()

if not models:
    st.error("❌ Models not found. Please train models first.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio(
    "",
    ["🎯 Live Prediction", "📈 Backtesting & Accuracy", "🔄 Compare Stocks", "📊 Analytics"],
    label_visibility="collapsed"
)

# Common sidebar inputs
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Enter any valid ticker",
    max_chars=10
).upper().strip()

model_name = st.sidebar.selectbox(
    "Model",
    options=list(models.keys()),
    help="Select prediction model"
)

# ================================================================================
# PAGE 1: LIVE PREDICTION
# ================================================================================

if page == "🎯 Live Prediction":
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Stock Volatility Forecaster</h1>
        <p class="dashboard-subtitle">Real-time volatility prediction using ensemble machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("🚀 FETCH DATA & PREDICT"):
        
        with st.spinner("Fetching real-time data..."):
            df = fetch_stock_data(ticker)
        
        if df is None:
            st.error(f"❌ Could not fetch data for {ticker}")
            st.stop()
        
        st.success(f"✅ Downloaded {len(df)} days of data (through {df.index[-1].strftime('%Y-%m-%d')})")
        
        with st.spinner("Making prediction..."):
            result = make_prediction(df, model_name, models)
        
        if result is None:
            st.error("❌ Prediction failed")
            st.stop()
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta = result['current_vol'] - 0.30  # compared to "normal" 30%
            st.metric(
                "Current Volatility",
                f"{result['current_vol']:.1%}",
                delta=f"{delta:+.1%}",
                help="21-day realized volatility"
            )
        
        with col2:
            delta = result['predicted_vol'] - result['current_vol']
            st.metric(
                "Predicted Volatility",
                f"{result['predicted_vol']:.1%}",
                delta=f"{delta:+.1%}",
                help="Forecasted 21-day volatility"
            )
        
        with col3:
            st.metric(
                "Model Confidence (R²)",
                f"{result['confidence']:.1%}",
                help="Model's explanatory power"
            )
        
        # Large line chart
        st.markdown("### 📈 Volatility Forecast")
        
        df_plot = result['data'].tail(252)
        
        fig = go.Figure()
        
        # Historical volatility
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['vol_21d'],
            mode='lines',
            name='Historical Volatility',
            line=dict(color='#00d4aa', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        # Prediction point
        fig.add_trace(go.Scatter(
            x=[result['latest_date']],
            y=[result['predicted_vol']],
            mode='markers',
            name='Prediction',
            marker=dict(
                color='#c4ff61',
                size=20,
                symbol='star',
                line=dict(color='#0a0e1a', width=2)
            )
        ))
        
        # Add 30% threshold line
        fig.add_hline(y=0.30, line_dash="dash", line_color="#ff6b6b", 
                     annotation_text="High Volatility Threshold (30%)")
        
        fig.update_layout(
            title=dict(
                text=f'{ticker} Volatility Forecast',
                font=dict(size=24, color='#e8eaf6')
            ),
            xaxis_title='Date',
            yaxis_title='Annualized Volatility',
            hovermode='x unified',
            plot_bgcolor='#1a1f2e',
            paper_bgcolor='#1a1f2e',
            font=dict(color='#e8eaf6'),
            height=600,
            yaxis=dict(tickformat='.0%', gridcolor='rgba(139, 146, 176, 0.1)'),
            xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison bar chart
        st.markdown("### 🏆 Model Performance Comparison")
        
        results_path = '../models/ensemble/final_results.csv'
        if os.path.exists(results_path):
            df_results = pd.read_csv(results_path)
            df_results = df_results[df_results['Model'].isin(['Naive', 'XGBoost', 'Random Forest', 'Ensemble (XGB 80%)'])]
            
            fig2 = go.Figure()
            
            colors = ['#8b92b0', '#00d4aa', '#4a9eff', '#c4ff61']
            
            fig2.add_trace(go.Bar(
                x=df_results['Model'],
                y=df_results['RMSE'],
                marker_color=colors,
                text=df_results['RMSE'].apply(lambda x: f'{x:.6f}'),
                textposition='outside',
                textfont=dict(color='#e8eaf6')
            ))
            
            fig2.update_layout(
                title='RMSE Comparison (Lower is Better)',
                xaxis_title='Model',
                yaxis_title='RMSE',
                plot_bgcolor='#1a1f2e',
                paper_bgcolor='#1a1f2e',
                font=dict(color='#e8eaf6'),
                height=400,
                yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)'),
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Interpretation
        if result['predicted_vol'] > result['current_vol'] * 1.2:
            st.markdown("""
            <div class="alert-warning">
                ⚠️ <strong>Rising Volatility Expected</strong><br>
                Model predicts significant increase. Consider risk management strategies.
            </div>
            """, unsafe_allow_html=True)
        elif result['predicted_vol'] < result['current_vol'] * 0.8:
            st.markdown("""
            <div class="alert-success">
                ✅ <strong>Decreasing Volatility Expected</strong><br>
                Model predicts calmer markets ahead.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-info">
                ℹ️ <strong>Stable Volatility Expected</strong><br>
                Model predicts volatility will remain relatively stable.
            </div>
            """, unsafe_allow_html=True)

# ================================================================================
# PAGE 2: BACKTESTING & ACCURACY
# ================================================================================

elif page == "📈 Backtesting & Accuracy":
    
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Backtesting & Accuracy</h1>
        <p class="dashboard-subtitle">Verify model reliability with historical predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Yesterday's Prediction Accuracy")
    
    st.info("⚠️ This feature requires historical predictions to be saved. Implement prediction logging to enable this feature.")
    
    # Placeholder for demonstration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Yesterday's Prediction", "18.5%")
    
    with col2:
        st.metric("Actual Volatility", "18.2%", delta="-0.3%")
    
    with col3:
        st.metric("Prediction Error", "1.6%", delta="-0.2%")
    
    st.markdown("### 📊 Rolling 30-Day Accuracy")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    errors = np.random.uniform(0.5, 3.0, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=errors,
        mode='lines+markers',
        name='Prediction Error %',
        line=dict(color='#00d4aa', width=2),
        marker=dict(size=8, color='#c4ff61')
    ))
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ff6b6b",
                 annotation_text="Target Error < 2%")
    
    fig.update_layout(
        title='30-Day Prediction Error',
        xaxis_title='Date',
        yaxis_title='Absolute Error %',
        plot_bgcolor='#1a1f2e',
        paper_bgcolor='#1a1f2e',
        font=dict(color='#e8eaf6'),
        height=500,
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE 3: COMPARE STOCKS
# ================================================================================

elif page == "🔄 Compare Stocks":
    
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Compare Stocks</h1>
        <p class="dashboard-subtitle">Side-by-side volatility comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker1 = st.text_input("Stock 1", "AAPL").upper()
    
    with col2:
        ticker2 = st.text_input("Stock 2", "TSLA").upper()
    
    if st.button("🔄 COMPARE"):
        
        results = {}
        
        for t in [ticker1, ticker2]:
            with st.spinner(f"Fetching {t}..."):
                df = fetch_stock_data(t)
                if df is not None:
                    result = make_prediction(df, model_name, models)
                    if result:
                        results[t] = result
        
        if len(results) == 2:
            
            col1, col2 = st.columns(2)
            
            for idx, (ticker, result) in enumerate(results.items()):
                with [col1, col2][idx]:
                    st.markdown(f"### {ticker}")
                    st.metric("Current Vol", f"{result['current_vol']:.1%}")
                    st.metric("Predicted Vol", f"{result['predicted_vol']:.1%}")
                    
                    # Mini chart
                    df_plot = result['data'].tail(60)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot['vol_21d'],
                        mode='lines',
                        line=dict(color=['#00d4aa', '#c4ff61'][idx], width=2)
                    ))
                    fig.update_layout(
                        height=250,
                        plot_bgcolor='#1a1f2e',
                        paper_bgcolor='#1a1f2e',
                        font=dict(color='#e8eaf6'),
                        yaxis=dict(tickformat='.0%', gridcolor='rgba(139, 146, 176, 0.1)'),
                        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE 4: ANALYTICS
# ================================================================================

elif page == "📊 Analytics":
    
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Analytics</h1>
        <p class="dashboard-subtitle">Deep dive into model performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Model Performance Breakdown")
    
    results_path = '../models/ensemble/final_results.csv'
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        
        # Filter out failed models
        df_results = df_results[df_results['R²'] > 0]
        
        st.dataframe(
            df_results.style.background_gradient(cmap='YlGn', subset=['R²'])
                           .background_gradient(cmap='RdYlGn_r', subset=['RMSE']),
            use_container_width=True
        )
        
        # Radar chart
        st.markdown("### 🎯 Multi-Metric Comparison")
        
        categories = ['RMSE', 'MAE', 'R²', 'Dir_Acc_%']
        
        fig = go.Figure()
        
        for model in ['XGBoost', 'Random Forest', 'Ensemble (XGB 80%)']:
            if model in df_results['Model'].values:
                row = df_results[df_results['Model'] == model].iloc[0]
                values = [
                    1 - (row['RMSE'] / df_results['RMSE'].max()),  # Normalized
                    1 - (row['MAE'] / df_results['MAE'].max()),
                    row['R²'],
                    row['Dir_Acc_%'] / 100
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor='#1a1f2e'
            ),
            plot_bgcolor='#1a1f2e',
            paper_bgcolor='#1a1f2e',
            font=dict(color='#e8eaf6'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b92b0; padding: 2rem;'>
    <p><strong style='color: #c4ff61;'>Stock Volatility Forecaster</strong></p>
    <p>Machine Learning Optimization • CA2 Project • Real-Time Predictions</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⚠️ For educational purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)