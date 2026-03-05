# 02_feature_engineering.py
# This script takes your raw data and creates 50+ smart features from it

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────
DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all CSV files
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

print("=" * 60)
print(" STEP 2: FEATURE ENGINEERING")
print("=" * 60)
print(f"\nFound {len(csv_files)} files to process\n")


# ══════════════════════════════════════════════════
# HELPER: CLEAN AND CONVERT DATA
# ══════════════════════════════════════════════════

def clean_numeric_columns(df):
    """Convert all price/volume columns to proper numbers"""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def safe_divide(numerator, denominator, fill_value=0):
    """Safely divide two series, handling division by zero"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill_value)


# ══════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ══════════════════════════════════════════════════

def add_volatility_features(df):
    """Calculate different types of volatility measures"""
    # Exponentially weighted volatility (recent days matter more)
    df['ewm_vol_21'] = df['log_return'].ewm(span=21, min_periods=21).std() * np.sqrt(252)
    
    # Parkinson volatility (uses High and Low prices)
    hl_ratio = np.log(df['High'] / df['Low']) ** 2
    df['parkinson_vol'] = np.sqrt(hl_ratio.rolling(21, min_periods=21).mean() / (4 * np.log(2))) * np.sqrt(252)
    
    # Garman-Klass volatility
    log_hl = (np.log(df['High'] / df['Low'])) ** 2
    log_co = (np.log(df['Close'] / df['Open'])) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df['gk_vol'] = np.sqrt(gk.rolling(21, min_periods=21).mean()) * np.sqrt(252)
    
    # Volatility of volatility
    df['vol_of_vol'] = df['vol_21d'].rolling(21, min_periods=21).std()
    
    return df


def add_price_features(df):
    """Price patterns and moving averages"""
    # Moving averages
    df['ma_5']   = df['Close'].rolling(5, min_periods=5).mean()
    df['ma_10']  = df['Close'].rolling(10, min_periods=10).mean()
    df['ma_21']  = df['Close'].rolling(21, min_periods=21).mean()
    df['ma_50']  = df['Close'].rolling(50, min_periods=50).mean()
    df['ma_200'] = df['Close'].rolling(200, min_periods=200).mean()
    
    # Price vs moving average (in %)
    df['price_vs_ma21']  = safe_divide((df['Close'] - df['ma_21']), df['ma_21']) * 100
    df['price_vs_ma50']  = safe_divide((df['Close'] - df['ma_50']), df['ma_50']) * 100
    df['price_vs_ma200'] = safe_divide((df['Close'] - df['ma_200']), df['ma_200']) * 100
    
    # Price momentum
    df['momentum_5']  = safe_divide((df['Close'] - df['Close'].shift(5)), df['Close'].shift(5)) * 100
    df['momentum_21'] = safe_divide((df['Close'] - df['Close'].shift(21)), df['Close'].shift(21)) * 100
    
    # High-Low range as % of close
    df['hl_pct'] = safe_divide((df['High'] - df['Low']), df['Close']) * 100
    
    return df


def add_technical_indicators(df):
    """Professional trading indicators"""
    # RSI - Relative Strength Index
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    rs = safe_divide(gain, loss, fill_value=50)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, min_periods=12).mean()
    ema_26 = df['Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20, min_periods=20).mean()
    bb_std = df['Close'].rolling(20, min_periods=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = safe_divide((df['bb_upper'] - df['bb_lower']), df['bb_middle'])
    df['bb_position'] = safe_divide((df['Close'] - df['bb_lower']), (df['bb_upper'] - df['bb_lower']))
    
    # ATR - Average True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14, min_periods=14).mean()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14, min_periods=14).min()
    high_14 = df['High'].rolling(14, min_periods=14).max()
    df['stoch_k'] = safe_divide((df['Close'] - low_14), (high_14 - low_14)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=3).mean()
    
    return df


def add_volume_features(df):
    """Trading volume patterns"""
    # Volume moving averages
    df['volume_ma_5']  = df['Volume'].rolling(5, min_periods=5).mean()
    df['volume_ma_21'] = df['Volume'].rolling(21, min_periods=21).mean()
    
    # Volume ratio
    df['volume_ratio'] = safe_divide(df['Volume'], df['volume_ma_21'], fill_value=1)
    
    # Price-volume relationship
    df['price_volume'] = df['log_return'] * df['Volume']
    
    # On-Balance Volume (simplified)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df


def add_lag_features(df):
    """What happened in the recent past"""
    # Lagged returns
    df['return_lag_1'] = df['log_return'].shift(1)
    df['return_lag_5'] = df['log_return'].shift(5)
    df['return_lag_21'] = df['log_return'].shift(21)
    
    # Lagged volatility
    df['vol_lag_1'] = df['vol_21d'].shift(1)
    df['vol_lag_5'] = df['vol_21d'].shift(5)
    
    # Rolling max/min prices
    df['high_21d'] = df['High'].rolling(21, min_periods=21).max()
    df['low_21d'] = df['Low'].rolling(21, min_periods=21).min()
    df['price_position'] = safe_divide((df['Close'] - df['low_21d']), (df['high_21d'] - df['low_21d']))
    
    return df


def add_regime_features(df):
    """What type of market are we in"""
    # Volatility regime
    vol_median = df['vol_21d'].rolling(252, min_periods=252).median()
    df['high_vol_regime'] = (df['vol_21d'] > vol_median * 1.5).astype(int)
    
    # Trend regime
    df['uptrend'] = (df['ma_50'] > df['ma_200']).astype(int)
    
    # Distance from 52-week high/low
    df['high_252d'] = df['High'].rolling(252, min_periods=252).max()
    df['low_252d'] = df['Low'].rolling(252, min_periods=252).min()
    df['pct_off_high'] = safe_divide((df['Close'] - df['high_252d']), df['high_252d']) * 100
    df['pct_off_low'] = safe_divide((df['Close'] - df['low_252d']), df['low_252d']) * 100
    
    return df


# ══════════════════════════════════════════════════
# PROCESS ALL FILES
# ══════════════════════════════════════════════════

feature_counts = []

for csv_file in csv_files:
    ticker = csv_file.replace('.csv', '')
    print(f"Processing {ticker}...", end=" ")
    
    try:
        # Load data
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file), index_col=0, parse_dates=True)
        
        # Clean data types
        df = clean_numeric_columns(df)
        
        original_cols = len(df.columns)
        
        # Apply all feature engineering functions
        df = add_volatility_features(df)
        df = add_price_features(df)
        df = add_technical_indicators(df)
        df = add_volume_features(df)
        df = add_lag_features(df)
        df = add_regime_features(df)
        
        new_cols = len(df.columns)
        features_added = new_cols - original_cols
        
        # Drop rows with too many NaN (from rolling calculations)
        df_clean = df.dropna()
        
        # Save to processed folder
        output_path = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
        df_clean.to_csv(output_path)
        
        feature_counts.append({
            'Ticker': ticker,
            'Original': original_cols,
            'New': new_cols,
            'Added': features_added,
            'Rows': len(df_clean)
        })
        
        print(f"✓  {features_added} features added → {len(df_clean)} clean rows")
    
    except Exception as e:
        print(f"✗  Error: {str(e)}")
        feature_counts.append({
            'Ticker': ticker,
            'Original': 0,
            'New': 0,
            'Added': 0,
            'Rows': 0
        })


# ══════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" FEATURE ENGINEERING COMPLETE")
print("=" * 60)

summary_df = pd.DataFrame(feature_counts)
print("\n" + summary_df.to_string(index=False))

successful = summary_df[summary_df['Rows'] > 0]
if len(successful) > 0:
    print(f"\n📁 Processed files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📊 Total features per file: ~{successful['New'].iloc[0]}")
    print("\n✅ Ready for model training!")
    print("\nNEXT STEP: Run  python 03_baseline_models.py")
else:
    print("\n⚠️  No files were processed successfully. Check errors above.")