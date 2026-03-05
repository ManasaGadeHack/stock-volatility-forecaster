import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Volatility Forecaster", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.4);
    }
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Stock Volatility Forecaster</h1>
    <p>Real-time stock price volatility prediction using advanced machine learning</p>
</div>
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
    result = pd.Series(index=a.index, dtype=float)
    mask = (b != 0) & (b.notna()) & (a.notna())
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill
    return result

def engineer_features(df):
    """Engineer all 56 features using SAFE operations"""
    
    # Volatility features
    df['log_return'] = np.log(safe_divide(df['Close'], df['Close'].shift(1), fill=1))
    
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

def fetch_and_predict(ticker, model_name, models):
    """Fetch real-time data and make prediction"""
    
    try:
        # Import yfinance here
        import yfinance as yf
        
        # Fetch data
        with st.spinner(f'📡 Fetching real-time data for {ticker}...'):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty or len(df) < 50:
                st.error(f"❌ No data found for ticker: {ticker}")
                return None
        
        st.success(f"✅ Downloaded {len(df)} days of data (through {df.index[-1].strftime('%Y-%m-%d')})")
        
        # Engineer features
        with st.spinner('⚙️ Engineering 56 features...'):
            df = engineer_features(df)
            df = df.dropna()
        
        if len(df) < 50:
            st.error("❌ Insufficient data after feature engineering")
            return None
        
        # Get current volatility
        current_vol = df['vol_21d'].iloc[-1]
        
        # Prepare features for prediction
        target_col = 'vol_21d'
        exclude_cols = [target_col, 'target_vol', 'vol_21d_sq', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
        
        X_latest = df[feature_cols].iloc[[-1]]
        
        # Make prediction
        with st.spinner(f'🤖 Generating prediction with {model_name}...'):
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
                
            else:  # Random Forest
                model = models['Random Forest']
                prediction = model.predict(X_latest)[0]
                confidence = 0.8167
        
        return {
            'ticker': ticker,
            'current_vol': current_vol,
            'predicted_vol': prediction,
            'confidence': confidence,
            'data': df,
            'latest_date': df.index[-1]
        }
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return None

# Main app
models = load_models()

if not models:
    st.error("❌ No models found. Please train models first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### 📈 Stock Selection")
    ticker = st.text_input(
        "Enter ANY Stock Ticker",
        value="AAPL",
        help="Type any valid ticker: AAPL, TSLA, GOOGL, NVDA, SPY, etc.",
        max_chars=10
    ).upper().strip()
    
    st.markdown("### 🤖 Model Selection")
    model_name = st.selectbox(
        "Choose Model",
        options=list(models.keys()),
        help="Select prediction model"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(f"""
    **Real-time data:** ✅ Yes
    
    **Any stock:** ✅ Yes
    
    **Models loaded:** {len(models)}
    
    Enter any ticker and click predict!
    """)

# Main content
if st.button("🔮 Fetch Data & Predict", use_container_width=True):
    
    if not ticker or len(ticker) == 0:
        st.warning("⚠️ Please enter a stock ticker")
    else:
        result = fetch_and_predict(ticker, model_name, models)
        
        if result:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Volatility",
                    f"{result['current_vol']:.1%}",
                    help="21-day realized volatility (annualized)"
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
            
            # Chart
            df_plot = result['data'].tail(252)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['vol_21d'],
                mode='lines',
                name='Historical Volatility',
                line=dict(color='#1f4e79', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[result['latest_date']],
                y=[result['predicted_vol']],
                mode='markers',
                name='Prediction',
                marker=dict(color='#ef4444', size=15, symbol='star')
            ))
            
            fig.update_layout(
                title=f'{ticker} Volatility Forecast',
                xaxis_title='Date',
                yaxis_title='Annualized Volatility',
                hovermode='x unified',
                template='plotly_white',
                height=500,
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if result['predicted_vol'] > result['current_vol'] * 1.2:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>Rising Volatility Expected</strong><br>
                    Model predicts significant increase. Consider risk management.
                </div>
                """, unsafe_allow_html=True)
            elif result['predicted_vol'] < result['current_vol'] * 0.8:
                st.markdown("""
                <div class="success-box">
                    ✅ <strong>Decreasing Volatility Expected</strong><br>
                    Model predicts calmer markets ahead.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    ℹ️ <strong>Stable Volatility Expected</strong><br>
                    Model predicts volatility will remain relatively stable.
                </div>
                """, unsafe_allow_html=True)

# Model comparison
st.markdown("---")
st.markdown("## 🏆 Model Performance Comparison")

results_path = '../models/ensemble/final_results.csv'
if os.path.exists(results_path):
    df_results = pd.read_csv(results_path)
    df_results = df_results[df_results['Model'].isin(['Naive', 'XGBoost', 'Random Forest', 'Ensemble (XGB 80%)'])]
    
    fig = go.Figure()
    colors = {'Naive': '#94a3b8', 'XGBoost': '#10b981', 'Random Forest': '#3b82f6', 'Ensemble (XGB 80%)': '#8b5cf6'}
    
    fig.add_trace(go.Bar(
        x=df_results['Model'],
        y=df_results['RMSE'],
        marker_color=[colors.get(m, '#64748b') for m in df_results['Model']],
        text=df_results['RMSE'].apply(lambda x: f'{x:.6f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='RMSE Comparison (Lower is Better)',
        xaxis_title='Model',
        yaxis_title='RMSE',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p><strong>Stock Volatility Forecaster</strong></p>
    <p>Real-Time Predictions • Machine Learning Optimization • CA2 Project</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⚠️ For educational purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)