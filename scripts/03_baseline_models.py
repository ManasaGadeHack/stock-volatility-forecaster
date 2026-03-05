# 03_baseline_models.py
# Builds 3 simple baseline models to compare against later

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────
DATA_DIR = "../data/processed"
MODELS_DIR = "../models/baseline"
FIG_DIR = "../results/figures"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# We'll test on SPY (S&P 500) as our main example
TEST_TICKER = "SPY_features.csv"

print("=" * 60)
print(" STEP 3: BASELINE MODELS")
print("=" * 60)


# ══════════════════════════════════════════════════
# LOAD AND SPLIT DATA
# ══════════════════════════════════════════════════

def load_and_split(filepath):
    """
    Load data and split into train/test
    Train: up to 2020-12-31
    Test:  2021-01-01 onwards
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Our target is the realized volatility we want to predict
    target_col = 'vol_21d'
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found!")
    
    # Split date
    split_date = '2021-01-01'
    
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()
    
    print(f"\n📊 Data loaded:")
    print(f"   Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows)")
    print(f"   Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)")
    
    return train, test, target_col


# ══════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculate 4 performance metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy (did we predict up/down correctly?)
    y_true_change = np.diff(y_true)
    y_pred_change = np.diff(y_pred)
    correct_direction = np.sum((y_true_change > 0) == (y_pred_change > 0))
    dir_accuracy = correct_direction / len(y_true_change) * 100
    
    return {
        'Model': model_name,
        'RMSE': round(rmse, 6),
        'MAE': round(rmse, 6),
        'R²': round(r2, 4),
        'Dir_Acc_%': round(dir_accuracy, 2)
    }


# ══════════════════════════════════════════════════
# MODEL 1: NAIVE FORECAST
# ══════════════════════════════════════════════════

def naive_forecast(train, test, target_col):
    """
    Simplest model: tomorrow = today
    Prediction: next volatility = current volatility
    """
    print("\n[1/3] Naive Forecast (tomorrow = today)...")
    
    # For test set, predict each day = previous day
    y_test = test[target_col].values
    y_pred = test[target_col].shift(1).values  # shift by 1 = use yesterday's value
    
    # Remove first NaN from shift
    y_test = y_test[1:]
    y_pred = y_pred[1:]
    
    metrics = calculate_metrics(y_test, y_pred, "Naive")
    
    print(f"   ✓  RMSE: {metrics['RMSE']:.6f}")
    
    return y_pred, metrics


# ══════════════════════════════════════════════════
# MODEL 2: MOVING AVERAGE
# ══════════════════════════════════════════════════

def moving_average_forecast(train, test, target_col, window=21):
    """
    Predict using rolling average of past volatility
    """
    print(f"\n[2/3] Moving Average ({window}-day window)...")
    
    # Combine train and test for rolling calculation
    full_data = pd.concat([train, test])
    
    # Calculate rolling mean
    rolling_mean = full_data[target_col].rolling(window=window).mean()
    
    # Extract predictions for test period
    test_indices = test.index
    y_pred = rolling_mean.loc[test_indices].values
    y_test = test[target_col].values
    
    # Remove NaN from start of rolling window
    mask = ~np.isnan(y_pred)
    y_pred = y_pred[mask]
    y_test = y_test[mask]
    
    metrics = calculate_metrics(y_test, y_pred, f"MA-{window}")
    
    print(f"   ✓  RMSE: {metrics['RMSE']:.6f}")
    
    return y_pred, metrics


# ══════════════════════════════════════════════════
# MODEL 3: GARCH(1,1)
# ══════════════════════════════════════════════════

def garch_forecast(train, test, target_col):
    """
    GARCH(1,1) - The financial industry standard
    Models volatility clustering and mean reversion
    """
    print("\n[3/3] GARCH(1,1) model...")
    
    # GARCH needs returns, not volatility directly
    # We'll use log returns
    train_returns = train['log_return'].dropna() * 100  # scale to percentages
    
    try:
        # Fit GARCH(1,1) model
        # p=1, q=1 are the standard parameters
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        fitted = model.fit(disp='off')
        
        # Forecast volatility for test period
        n_test = len(test)
        forecast = fitted.forecast(horizon=n_test, reindex=False)
        
        # Extract conditional volatility (annualized)
        y_pred = np.sqrt(forecast.variance.values[-1, :]) / 100 * np.sqrt(252)
        
        # Get actual test values
        y_test = test[target_col].values
        
        # Match lengths (GARCH might return slightly different length)
        min_len = min(len(y_pred), len(y_test))
        y_pred = y_pred[:min_len]
        y_test = y_test[:min_len]
        
        metrics = calculate_metrics(y_test, y_pred, "GARCH(1,1)")
        
        print(f"   ✓  RMSE: {metrics['RMSE']:.6f}")
        print(f"   ℹ️  GARCH parameters: ω={fitted.params['omega']:.6f}, "
              f"α={fitted.params['alpha[1]']:.4f}, β={fitted.params['beta[1]']:.4f}")
        
        return y_pred, metrics, fitted
    
    except Exception as e:
        print(f"   ✗  GARCH fitting failed: {str(e)}")
        print(f"   ℹ️  Using simple volatility as fallback")
        
        # Fallback to simple historical volatility
        y_pred = np.full(len(test), train[target_col].mean())
        y_test = test[target_col].values
        
        metrics = calculate_metrics(y_test, y_pred, "GARCH(1,1)*")
        return y_pred, metrics, None


# ══════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════

def plot_comparison(test, predictions_dict, target_col, fig_dir):
    """
    Create comparison chart of all models vs actual
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Get test dates and actual values
    test_dates = test.index
    y_actual = test[target_col].values
    
    # Plot 1: Full test period
    ax1.plot(test_dates, y_actual, label='Actual', color='black', 
             linewidth=2, alpha=0.8)
    
    colors = {'Naive': 'blue', 'MA-21': 'green', 'GARCH(1,1)': 'red'}
    
    for model_name, y_pred in predictions_dict.items():
        # Match lengths
        dates = test_dates[:len(y_pred)]
        ax1.plot(dates, y_pred, label=model_name, 
                linewidth=1.5, alpha=0.7, color=colors.get(model_name, 'gray'))
    
    ax1.set_title('Baseline Models: Volatility Predictions vs Actual', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Realized Volatility (annualized)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 2: Zoom into volatile period (first 6 months of test)
    zoom_end = min(120, len(test))
    zoom_dates = test_dates[:zoom_end]
    
    ax2.plot(zoom_dates, y_actual[:zoom_end], label='Actual', 
            color='black', linewidth=2, alpha=0.8)
    
    for model_name, y_pred in predictions_dict.items():
        pred_zoom = y_pred[:zoom_end]
        ax2.plot(zoom_dates[:len(pred_zoom)], pred_zoom, 
                label=model_name, linewidth=1.5, alpha=0.7,
                color=colors.get(model_name, 'gray'))
    
    ax2.set_title('Zoomed View: First 6 Months', fontsize=12)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Realized Volatility', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, '03_baseline_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Chart saved: {save_path}")


def plot_metrics_bar(results_df, fig_dir):
    """
    Bar chart comparing model performance
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    models = results_df['Model'].values
    x = np.arange(len(models))
    
    # RMSE (lower is better)
    axes[0].bar(x, results_df['RMSE'].values, color='steelblue', alpha=0.7)
    axes[0].set_title('RMSE (lower is better)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # R² (higher is better)
    axes[1].bar(x, results_df['R²'].values, color='seagreen', alpha=0.7)
    axes[1].set_title('R² (higher is better)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=0)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Directional Accuracy (higher is better)
    axes[2].bar(x, results_df['Dir_Acc_%'].values, color='coral', alpha=0.7)
    axes[2].set_title('Directional Accuracy % (higher is better)', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=0)
    axes[2].axhline(50, color='gray', linestyle='--', linewidth=1, label='Random guess')
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, '03_baseline_metrics.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Chart saved: {save_path}")


# ══════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Load data
    data_path = os.path.join(DATA_DIR, TEST_TICKER)
    train, test, target_col = load_and_split(data_path)
    
    # Train all 3 baseline models
    results = []
    predictions = {}
    
    # Model 1: Naive
    pred_naive, metrics_naive = naive_forecast(train, test, target_col)
    results.append(metrics_naive)
    predictions['Naive'] = pred_naive
    
    # Model 2: Moving Average
    pred_ma, metrics_ma = moving_average_forecast(train, test, target_col, window=21)
    results.append(metrics_ma)
    predictions['MA-21'] = pred_ma
    
    # Model 3: GARCH
    pred_garch, metrics_garch, garch_model = garch_forecast(train, test, target_col)
    results.append(metrics_garch)
    predictions['GARCH(1,1)'] = pred_garch
    
    # Create results table
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(MODELS_DIR, 'baseline_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Display results
    print("\n" + "=" * 60)
    print(" BASELINE MODELS RESULTS")
    print("=" * 60)
    print("\n" + results_df.to_string(index=False))
    
    # Create visualizations
    print("\n" + "=" * 60)
    print(" CREATING CHARTS")
    print("=" * 60)
    
    plot_comparison(test, predictions, target_col, FIG_DIR)
    plot_metrics_bar(results_df, FIG_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print(" ✅ BASELINE MODELS COMPLETE")
    print("=" * 60)
    print(f"\n📁 Results saved: {results_path}")
    print(f"📊 Charts saved: {FIG_DIR}")
    
    best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    best_rmse = results_df['RMSE'].min()
    
    print(f"\n🏆 Best baseline model: {best_model} (RMSE: {best_rmse:.6f})")
    print(f"\n💡 Next step: Build ML models that beat RMSE < {best_rmse:.6f}")
    print("\nNEXT STEP: Run  python 04_ml_models.py")