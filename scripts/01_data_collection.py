# 01_data_collection.py
# This script downloads stock data from the internet and saves it to your computer

import yfinance as yf       # downloads stock prices from Yahoo Finance
import pandas as pd         # handles tables of data
import numpy as np          # does maths
import os                   # creates folders on your computer
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────
# These are the 8 stocks we want to study
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", "SPY", "^VIX", "CL=F", "^NSEI"]
#Crude oil WTI/NIFTY 50
# Date range: 10 years of data
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# Where to save the files
DATA_DIR = "../data/raw"
os.makedirs(DATA_DIR, exist_ok=True)   # creates the folder if it doesn't exist

# ── DOWNLOAD DATA ─────────────────────────────────
print("Starting download...")
print("=" * 50)

for ticker in TICKERS:
    print(f"Downloading {ticker}...", end=" ")

    # Download the stock data
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True)

    # Add a column for daily return (how much price changed each day)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Add volatility columns (how much prices bounce around)
    df["vol_5d"]  = df["log_return"].rolling(5).std()  * np.sqrt(252)
    df["vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["log_return"].rolling(63).std() * np.sqrt(252)

    # This is what we want to PREDICT: volatility 21 days into the future
    df["target_vol"] = df["vol_21d"].shift(-21)

    # Save to CSV file
    safe_name = ticker.replace("^", "")   # VIX not ^VIX for filename
    filepath  = os.path.join(DATA_DIR, f"{safe_name}.csv")
    df.to_csv(filepath)

    print(f"✓  saved {len(df)} rows  →  {filepath}")

print("=" * 50)
print("✅ ALL DONE! Check your data/raw folder.")