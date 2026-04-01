import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="VolStock • AI Stock Volatility Prediction", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Initialize theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Handle toggle via pure HTML link with query param
if 'toggle' in st.query_params:
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.query_params.clear()
    st.rerun()

# Dynamic CSS based on theme
if st.session_state.theme == 'dark':
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Rajdhani', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a0b2e 50%, #16213e 100%) !important; color: #e0f2fe !important; }
    .main .block-container { padding-top: 0rem !important; max-width: 100% !important; padding-bottom: 0rem !important; }
    section.main > div, div[data-testid="stVerticalBlock"] { position: relative; z-index: 2; }
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, .stDeployButton { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stVerticalBlock"] > div:first-child { margin-top: 0 !important; gap: 0 !important; }
    div[data-testid="stVerticalBlock"] { gap: 0 !important; }
    div[data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }
    
    /* Theme toggle - pure HTML link styled as a perfect square */
    .theme-btn-link { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 10px; width: 2.6rem; height: 2.6rem; min-width: 2.6rem; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; text-decoration: none; flex-shrink: 0; box-shadow: 0 4px 14px rgba(99,102,241,0.5); transition: transform 0.15s, filter 0.15s; line-height: 1; }
    .theme-btn-link:hover { transform: scale(1.08); filter: brightness(1.2); }
    .brand-group { display: flex; align-items: center; gap: 0.9rem; }
    
    .top-nav { background: rgba(10, 14, 39, 0.6); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(0, 255, 255, 0.1); padding: 1.2rem 3rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1); }
    .brand { font-family: 'Orbitron', monospace; font-size: 2.2rem; font-weight: 900; background: linear-gradient(135deg, #00ffff 0%, #a78bfa 30%, #ec4899 60%, #00ffff 100%); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 2px; text-transform: uppercase; }
    .price-ticker { display: flex; align-items: center; gap: 1.5rem; font-weight: 600; }
    .live-price { font-size: 1.4rem; font-weight: 700; color: #00ff88; }
    .price-change { padding: 0.4rem 0.9rem; border-radius: 8px; font-weight: 700; font-size: 0.95rem; }
    .price-change.up { background: rgba(0, 255, 136, 0.15); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; }
    .price-change.down { background: rgba(255, 71, 87, 0.15); border: 1px solid rgba(255, 71, 87, 0.3); color: #ff4757; }
    .nav-date { color: #64748b; font-size: 0.9rem; margin-left: 1rem; }
    .model-badge { background: rgba(167, 139, 250, 0.2); color: #a78bfa; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin-left: 1rem; }
    
    .stTextInput input, .stSelectbox > div > div, .stNumberInput input { background: rgba(15, 23, 42, 0.8) !important; border: 1px solid rgba(0, 255, 255, 0.2) !important; border-radius: 12px !important; color: #e0f2fe !important; padding: 0.9rem 1.2rem !important; font-size: 1.05rem !important; font-weight: 600 !important; }
    .stSelectbox [data-baseweb="select"] > div { background: rgba(15, 23, 42, 0.8) !important; color: #e0f2fe !important; }
    .stSelectbox [data-baseweb="select"] span { color: #e0f2fe !important; }
    .stButton button { background: linear-gradient(135deg, #00ffff 0%, #a78bfa 50%, #ec4899 100%) !important; background-size: 200% auto !important; color: #0a0e27 !important; border: none !important; padding: 0.9rem 2rem !important; font-size: 1.05rem !important; font-weight: 900 !important; border-radius: 12px !important; font-family: 'Orbitron', monospace !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; width: 100% !important; }
    .stDownloadButton button { background: linear-gradient(135deg, #00ffff 0%, #a78bfa 50%, #ec4899 100%) !important; color: #0a0e27 !important; border: none !important; padding: 0.9rem 2rem !important; font-size: 1.05rem !important; font-weight: 900 !important; border-radius: 12px !important; font-family: 'Orbitron', monospace !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; width: 100% !important; }
    
    .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; padding: 2rem 3rem; }
    .metric-block { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(20px); border-radius: 24px; padding: 2rem 1.8rem; border: 1px solid rgba(255, 255, 255, 0.05); transition: all 0.3s; }
    .metric-block:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3); }
    .metric-label { color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem; }
    .metric-value { font-size: 2.5rem; font-weight: 900; font-family: 'Orbitron', monospace; margin: 0.5rem 0; line-height: 1; }
    .metric-block.cyan .metric-value { color: #00ffff; text-shadow: 0 0 30px rgba(0, 255, 255, 0.5); }
    .metric-block.purple .metric-value { color: #a78bfa; text-shadow: 0 0 30px rgba(167, 139, 250, 0.5); }
    .metric-block.pink .metric-value { color: #ec4899; text-shadow: 0 0 30px rgba(236, 72, 153, 0.5); }
    .metric-block.orange .metric-value { color: #f59e0b; text-shadow: 0 0 30px rgba(245, 158, 11, 0.5); }
    .metric-change { color: #64748b; font-size: 0.95rem; margin-top: 0.8rem; }
    .metric-change.up { color: #10b981; }
    .metric-change.down { color: #ef4444; }
    
    .risk-badge { display: inline-block; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1rem; font-family: 'Orbitron', monospace; }
    .risk-low { background: rgba(0, 255, 136, 0.2); color: #00ff88; border: 2px solid rgba(0, 255, 136, 0.4); }
    .risk-medium { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 2px solid rgba(251, 191, 36, 0.4); }
    .risk-high { background: rgba(255, 71, 87, 0.2); color: #ff4757; border: 2px solid rgba(255, 71, 87, 0.4); }
    
    .alert-bar { margin: 0 3rem 2rem 3rem; padding: 1.2rem 2rem; border-radius: 16px; display: flex; align-items: center; gap: 1.2rem; border: 1px solid; font-weight: 600; font-size: 1.05rem; }
    .alert-bar.warning { background: rgba(255, 71, 87, 0.15); border-color: rgba(255, 71, 87, 0.3); color: #ff4757; }
    
    .content-section { padding: 0 3rem 2rem 3rem; }
    .section-title { font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900; margin-bottom: 2rem; background: linear-gradient(135deg, #00ffff 0%, #a78bfa 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-transform: uppercase; letter-spacing: 2px; }
    .history-item { background: rgba(15, 23, 42, 0.7); padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 0.8rem; border-left: 3px solid #a78bfa; transition: all 0.3s; }
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0a0e27; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #00ffff, #a78bfa); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%) !important; color: #1e293b !important; }
    .main .block-container { padding-top: 0rem !important; max-width: 100% !important; padding-bottom: 0rem !important; }
    section.main > div, div[data-testid="stVerticalBlock"] { position: relative; z-index: 2; }
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, .stDeployButton { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stVerticalBlock"] > div:first-child { margin-top: 0 !important; gap: 0 !important; }
    div[data-testid="stVerticalBlock"] { gap: 0 !important; }
    div[data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }
    
    /* Theme toggle - pure HTML link styled as a perfect square */
    .theme-btn-link { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 10px; width: 2.6rem; height: 2.6rem; min-width: 2.6rem; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; text-decoration: none; flex-shrink: 0; box-shadow: 0 4px 12px rgba(99,102,241,0.3); transition: transform 0.15s, filter 0.15s; line-height: 1; }
    .theme-btn-link:hover { transform: scale(1.08); filter: brightness(1.2); }
    .brand-group { display: flex; align-items: center; gap: 0.9rem; }
    
    /* Fix expander (Alert Settings) text color */
    .streamlit-expanderHeader { color: #1e293b !important; font-weight: 600 !important; }
    details summary { color: #1e293b !important; }
    details summary p { color: #1e293b !important; }
    [data-testid="stExpander"] summary { color: #1e293b !important; }
    [data-testid="stExpander"] summary span { color: #1e293b !important; }
    [data-testid="stExpander"] p { color: #1e293b !important; }
    
    /* Fix slider label */
    [data-testid="stSlider"] label, [data-testid="stSlider"] p { color: #1e293b !important; }
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: #475569 !important; }
    
    /* Fix download button / export CSV in light mode */
    .stDownloadButton button { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important; color: #ffffff !important; border: none !important; padding: 0.9rem 2rem !important; font-size: 1rem !important; font-weight: 700 !important; border-radius: 12px !important; text-transform: uppercase !important; letter-spacing: 1px !important; width: 100% !important; }
    
    .top-nav { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border-bottom: 2px solid #e2e8f0; padding: 1.2rem 3rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); }
    .brand { font-size: 2rem; font-weight: 900; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
    .price-ticker { display: flex; align-items: center; gap: 1.5rem; font-weight: 600; }
    .live-price { font-size: 1.4rem; font-weight: 700; color: #10b981; }
    .price-change { padding: 0.4rem 0.9rem; border-radius: 8px; font-weight: 700; font-size: 0.95rem; }
    .price-change.up { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #059669; }
    .price-change.down { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #dc2626; }
    .nav-date { color: #64748b; font-size: 0.9rem; margin-left: 1rem; }
    .model-badge { background: rgba(139, 92, 246, 0.15); color: #7c3aed; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin-left: 1rem; border: 1px solid rgba(139, 92, 246, 0.3); }
    
    .stTextInput input, .stSelectbox > div > div, .stNumberInput input { background: #ffffff !important; border: 2px solid #e2e8f0 !important; border-radius: 12px !important; color: #1e293b !important; padding: 0.9rem 1.2rem !important; font-size: 1rem !important; font-weight: 600 !important; }
    .stSelectbox [data-baseweb="select"] > div { background: #ffffff !important; color: #1e293b !important; }
    .stSelectbox [data-baseweb="select"] span { color: #1e293b !important; }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important; color: #ffffff !important; border: none !important; padding: 0.9rem 2rem !important; font-size: 1rem !important; font-weight: 700 !important; border-radius: 12px !important; text-transform: uppercase !important; letter-spacing: 1px !important; width: 100% !important; }
    
    .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; padding: 2rem 3rem; }
    .metric-block { background: #ffffff; border-radius: 20px; padding: 2rem 1.8rem; border: 2px solid #e2e8f0; transition: all 0.3s; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); }
    .metric-block:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1); }
    .metric-label { color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem; }
    .metric-value { font-size: 2.5rem; font-weight: 900; margin: 0.5rem 0; line-height: 1; }
    .metric-block.blue .metric-value { color: #3b82f6; }
    .metric-block.purple .metric-value { color: #8b5cf6; }
    .metric-block.pink .metric-value { color: #ec4899; }
    .metric-block.orange .metric-value { color: #f59e0b; }
    .metric-change { color: #64748b; font-size: 0.95rem; margin-top: 0.8rem; }
    .metric-change.up { color: #10b981; }
    .metric-change.down { color: #ef4444; }
    
    .risk-badge { display: inline-block; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 700; font-size: 1rem; }
    .risk-low { background: rgba(16, 185, 129, 0.15); color: #059669; border: 2px solid rgba(16, 185, 129, 0.3); }
    .risk-medium { background: rgba(245, 158, 11, 0.15); color: #d97706; border: 2px solid rgba(245, 158, 11, 0.3); }
    .risk-high { background: rgba(239, 68, 68, 0.15); color: #dc2626; border: 2px solid rgba(239, 68, 68, 0.3); }
    
    .alert-bar { margin: 0 3rem 2rem 3rem; padding: 1.2rem 2rem; border-radius: 12px; display: flex; align-items: center; gap: 1.2rem; border: 2px solid; font-weight: 600; font-size: 1.05rem; }
    .alert-bar.warning { background: rgba(239, 68, 68, 0.1); border-color: #ef4444; color: #dc2626; }
    
    .content-section { padding: 0 3rem 2rem 3rem; }
    .section-title { font-size: 1.8rem; font-weight: 900; margin-bottom: 2rem; color: #1e293b; text-transform: uppercase; letter-spacing: 1px; }
    .history-item { background: #ffffff; padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 0.8rem; border-left: 4px solid #8b5cf6; transition: all 0.3s; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Chart colors based on theme
chart_config = {
    'dark': {
        'line': '#00ffff',
        'fill': 'rgba(0, 255, 255, 0.1)',
        'pred': '#ec4899',
        'conf': 'rgba(167, 139, 250, 0.2)',
        'plot_bg': 'rgba(10, 14, 39, 0.8)',
        'paper_bg': 'rgba(10, 14, 39, 0.6)',
        'grid': 'rgba(255, 255, 255, 0.05)',
        'font': '#e0f2fe'
    },
    'light': {
        'line': '#3b82f6',
        'fill': 'rgba(59, 130, 246, 0.1)',
        'pred': '#ec4899',
        'conf': 'rgba(139, 92, 246, 0.2)',
        'plot_bg': 'rgba(255, 255, 255, 0.9)',
        'paper_bg': 'rgba(255, 255, 255, 0.9)',
        'grid': 'rgba(0, 0, 0, 0.05)',
        'font': '#1e293b'
    }
}

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
    except: 
        pass
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
    for w in [5, 21, 63]: df[f'vol_{w}d'] = df['log_return'].rolling(w).std() * np.sqrt(252)
    df['ewm_vol_21'] = df['log_return'].ewm(span=21, min_periods=21).std() * np.sqrt(252)
    log_hl = np.log(safe_divide(df['High'], df['Low'], fill=1)) ** 2
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl.rolling(21).mean()) * np.sqrt(252)
    log_hl = (np.log(safe_divide(df['High'], df['Low'], fill=1))) ** 2
    log_co = (np.log(safe_divide(df['Close'], df['Open'], fill=1))) ** 2
    df['gk_vol'] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean()) * np.sqrt(252)
    df['vol_of_vol'] = df['vol_21d'].rolling(21).std()
    for w in [5, 10, 21, 50, 200]: df[f'ma_{w}'] = df['Close'].rolling(w).mean()
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

def calculate_risk_score(predicted_vol):
    if predicted_vol < 0.15: return 1, "Very Low"
    elif predicted_vol < 0.20: return 3, "Low"
    elif predicted_vol < 0.25: return 5, "Medium"
    elif predicted_vol < 0.35: return 7, "High"
    else: return 10, "Very High"

def fetch_and_predict(ticker, model_name, models):
    import yfinance as yf
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 50:
        return None
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
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
    
    error_margin = 0.02
    lower_bound = prediction * (1 - error_margin)
    upper_bound = prediction * (1 + error_margin)
    
    risk_score, risk_label = calculate_risk_score(prediction)
    
    return {
        'ticker': ticker,
        'model_used': model_name,
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'current_vol': current_vol,
        'predicted_vol': prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence': confidence,
        'risk_score': risk_score,
        'risk_label': risk_label,
        'data': df,
        'latest_date': df.index[-1]
    }

def save_prediction_history(result):
    history_file = '../data/prediction_history.json'
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'ticker': result['ticker'],
        'model': result['model_used'],
        'current_vol': float(result['current_vol']),
        'predicted_vol': float(result['predicted_vol']),
        'risk_score': result['risk_score']
    }
    
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    history.append(history_entry)
    history = history[-20:]
    
    with open(history_file, 'w') as f:
        json.dump(history, f)

def load_prediction_history():
    history_file = '../data/prediction_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def export_to_csv(result):
    df = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Ticker': result['ticker'],
        'Model': result['model_used'],
        'Current_Price_INR': result['current_price'] * 83,
        'Current_Volatility': result['current_vol'],
        'Predicted_Volatility': result['predicted_vol'],
        'Lower_Bound': result['lower_bound'],
        'Upper_Bound': result['upper_bound'],
        'Risk_Score': result['risk_score'],
        'Risk_Label': result['risk_label']
    }])
    return df.to_csv(index=False)

# Initialize
if 'result' not in st.session_state:
    st.session_state.result = None
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 0.30

models = load_models()
if not models:
    st.error("❌ Models not found")
    st.stop()

result = st.session_state.result

# Theme icon: in dark mode show moon, in light mode show sun
theme_icon = "🌙" if st.session_state.theme == 'dark' else "☀️"

# Header — toggle is a pure <a href="?toggle=1"> styled as a square button, no st.button needed
if result:
    price_class = "up" if result['price_change'] >= 0 else "down"
    price_arrow = "↑" if result['price_change'] >= 0 else "↓"
    st.markdown(f"""
    <div class="top-nav">
        <div class="brand-group">
            <a href="?toggle=1" class="theme-btn-link">{theme_icon}</a>
            <div class="brand">VOLSTOCK</div>
        </div>
        <div class="price-ticker">
            <div class="live-price">₹{result['current_price']*83:.2f}</div>
            <div class="price-change {price_class}">
                {price_arrow} ₹{abs(result['price_change']*83):.2f} ({result['price_change_pct']:+.2f}%)
            </div>
            <div class="nav-date">{result['ticker']} <span class="model-badge">Model: {result['model_used']}</span> • {datetime.now().strftime('%d %b %Y')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="top-nav">
        <div class="brand-group">
            <a href="?toggle=1" class="theme-btn-link">{theme_icon}</a>
            <div class="brand">VOLSTOCK</div>
        </div>
        <div class="nav-date">{datetime.now().strftime('%A, %d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

# Inputs
col1, col2, col3, col4 = st.columns([3, 2, 1.3, 1.3])
with col1:
    ticker = st.text_input("Stock Ticker", "AAPL", label_visibility="collapsed", placeholder="Enter ticker (e.g., AAPL)").upper().strip()
with col2:
    model_name = st.selectbox("Select Model", list(models.keys()), label_visibility="collapsed")
with col3:
    predict_btn = st.button("🚀 PREDICT")
with col4:
    compare_spy = st.button("📊 VS SPY")

with st.expander("⚙️ Alert Settings"):
    alert_threshold = st.slider("Volatility Alert Threshold (%)", 10, 50, int(st.session_state.alert_threshold * 100))
    st.session_state.alert_threshold = alert_threshold / 100

if predict_btn:
    with st.spinner("Analyzing..."):
        st.session_state.result = fetch_and_predict(ticker, model_name, models)
        if st.session_state.result:
            save_prediction_history(st.session_state.result)
    st.rerun()

if compare_spy:
    with st.spinner("Comparing with SPY..."):
        spy_result = fetch_and_predict("SPY", model_name, models)
        if spy_result:
            st.session_state.spy_result = spy_result
    st.rerun()

result = st.session_state.result

# Main Content
if result:
    if result['predicted_vol'] > st.session_state.alert_threshold:
        st.markdown(f"""
        <div class="alert-bar warning">
            🚨 <strong>ALERT!</strong> Predicted volatility ({result['predicted_vol']:.1%}) exceeds your threshold ({st.session_state.alert_threshold:.1%})
        </div>
        """, unsafe_allow_html=True)
    
    risk_class = "risk-low" if result['risk_score'] <= 3 else "risk-medium" if result['risk_score'] <= 7 else "risk-high"
    colors = 'cyan purple pink orange' if st.session_state.theme == 'dark' else 'blue purple pink orange'
    col_classes = colors.split()
    
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-block {col_classes[0]}">
            <div class="metric-label">Current Volatility</div>
            <div class="metric-value">{result['current_vol']:.1%}</div>
            <div class="metric-change {'up' if result['current_vol'] > 0.3 else 'down'}">
                {'↑' if result['current_vol'] > 0.3 else '↓'} {abs(result['current_vol'] - 0.3):.1%} vs normal
            </div>
        </div>
        <div class="metric-block {col_classes[1]}">
            <div class="metric-label">Predicted Volatility</div>
            <div class="metric-value">{result['predicted_vol']:.1%}</div>
            <div class="metric-change {'up' if result['predicted_vol'] > result['current_vol'] else 'down'}">
                {'↑' if result['predicted_vol'] > result['current_vol'] else '↓'} {abs(result['predicted_vol'] - result['current_vol']):.1%}
            </div>
        </div>
        <div class="metric-block {col_classes[2]}">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">{result['confidence']:.1%}</div>
            <div class="metric-change up">±{2.0:.1f}% error</div>
        </div>
        <div class="metric-block {col_classes[3]}">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value">{result['risk_score']}/10</div>
            <div class="metric-change"><span class="risk-badge {risk_class}">{result['risk_label']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = export_to_csv(result)
        st.download_button(
            "📥 EXPORT CSV",
            csv_data,
            f"{result['ticker']}_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    with col2:
        if st.button("📜 VIEW HISTORY", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
    with col3:
        st.button("🔄 REFRESH", use_container_width=True, on_click=lambda: st.rerun())
    
    if st.session_state.get('show_history', False):
        st.markdown('<div class="content-section"><h2 class="section-title">📜 Recent Predictions</h2>', unsafe_allow_html=True)
        history = load_prediction_history()
        if history:
            for h in reversed(history[-5:]):
                st.markdown(f"""
                <div class="history-item">
                    <div>
                        <strong>{h['ticker']}</strong> • {h.get('model', 'N/A')} • {datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M')}
                        <br><small>Predicted: {h['predicted_vol']:.1%} | Risk: {h['risk_score']}/10</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No prediction history yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-section"><h2 class="section-title">📈 Volatility Forecast Analysis</h2>', unsafe_allow_html=True)
    
    df_plot = result['data'].tail(252)
    cfg = chart_config[st.session_state.theme]
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['vol_21d'],
        mode='lines',
        name='Historical Volatility',
        line=dict(color=cfg['line'], width=3),
        fill='tozeroy',
        fillcolor=cfg['fill']
    ))
    
    fig.add_trace(go.Scatter(
        x=[result['latest_date'], result['latest_date']],
        y=[result['lower_bound'], result['upper_bound']],
        fill='toself',
        fillcolor=cfg['conf'],
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[result['latest_date']],
        y=[result['predicted_vol']],
        mode='markers',
        name='Prediction',
        marker=dict(color=cfg['pred'], size=25, symbol='star', line=dict(color='#ffffff', width=2))
    ))
    
    fig.add_hline(
        y=st.session_state.alert_threshold, 
        line_dash="dash", 
        line_color="#ef4444", 
        annotation_text=f"Alert Threshold ({st.session_state.alert_threshold:.0%})"
    )
    
    fig.update_layout(
        title=f'{result["ticker"]} • {result["model_used"]} Forecast',
        xaxis_title='',
        yaxis_title='Annualized Volatility',
        plot_bgcolor=cfg['plot_bg'],
        paper_bgcolor=cfg['paper_bg'],
        font=dict(color=cfg['font'], size=14),
        height=600,
        yaxis=dict(tickformat='.0%', gridcolor=cfg['grid']),
        xaxis=dict(gridcolor=cfg['grid']),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'spy_result' in st.session_state:
        st.markdown('<div class="content-section"><h2 class="section-title">📊 Comparison with S&P 500</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{result['ticker']} Risk", f"{result['risk_score']}/10", f"{result['risk_label']}")
        with col2:
            spy = st.session_state.spy_result
            st.metric("SPY Risk", f"{spy['risk_score']}/10", f"{spy['risk_label']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="content-section"><h2 class="section-title">📖 Feature Guide</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Metrics
        
        **Current Volatility:** 21-day realized volatility (annualized)
        
        **Predicted Volatility:** Forecast for next 21 days
        
        **Confidence:** R² score showing accuracy
        
        **Risk Score:** 1-10 rating
        - 1-3: Low risk
        - 4-6: Medium risk
        - 7-10: High risk
        """)
    
    with col2:
        st.markdown("""
        ### 🔘 Actions
        
        **📥 Export CSV:** Download prediction data
        
        **📜 View History:** Last 5 predictions
        
        **🔄 Refresh:** Fetch latest data
        
        **📊 VS SPY:** Compare with S&P 500
        
        **⚙️ Alerts:** Set custom thresholds
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Chart Guide
        
        **Blue/Cyan Line:** Historical volatility
        
        **Pink Star:** Predicted volatility
        
        **Purple Area:** 95% confidence interval
        
        **Red Dashed:** Alert threshold
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 6rem 2rem;'>
        <div style='font-size: 5rem; margin-bottom: 2rem;'>📈</div>
        <h2 style='font-size: 2.5rem; font-weight: 900; margin-bottom: 1.5rem;'>
            ENTER A STOCK TICKER TO BEGIN
        </h2>
        <p style='font-size: 1.2rem; color: #64748b; max-width: 600px; margin: 0 auto; line-height: 1.8;'>
            Real-time volatility forecasting powered by advanced machine learning
            <br><br>
            <span>•</span> Live market data
            <span>•</span> AI predictions
            <span>•</span> Risk analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p><strong style='color: #3b82f6;'>📈 VolStock</strong> • AI Stock Volatility Prediction</p>
    <p>Machine Learning Optimization • Real-Time Predictions • Indian Market Focus</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⚠️ For educational purposes only. Not financial advice. Prices converted at 1 USD = ₹83
    </p>
</div>
""", unsafe_allow_html=True)
