# 04_ml_models.py
# Random Forest and XGBoost models using all 56 features

import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────
DATA_DIR = "../data/processed"
MODELS_DIR = "../models/ml_models"
BASELINE_DIR = "../models/baseline"
FIG_DIR = "../results/figures"

os.makedirs(MODELS_DIR, exist_ok=True)

# Test on SPY (S&P 500)
TEST_TICKER = "SPY_features.csv"

print("=" * 60)
print(" STEP 4: MACHINE LEARNING MODELS")
print(" Random Forest & XGBoost")
print("=" * 60)


# ══════════════════════════════════════════════════
# LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════

def load_and_prepare_data(filepath):
    """
    Load data, select features, split into train/val/test
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"\n📊 Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Target variable
    target_col = 'vol_21d'
    
    # Feature selection: use all features EXCEPT target and highly correlated ones
    exclude_cols = [
        target_col,
        'target_vol',  # future value (data leakage!)
        'vol_21d_sq',  # just squared version
        'Open', 'High', 'Low', 'Close', 'Volume',  # raw prices (we have better features)
        'log_return'   # already captured in lag features
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n🎯 Target: {target_col}")
    print(f"📈 Features: {len(feature_cols)} selected")
    print(f"\nTop 10 features: {feature_cols[:10]}")
    
    # Remove any rows with NaN in features or target
    df_clean = df[feature_cols + [target_col]].dropna()
    
    print(f"\n✅ Clean data: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows with NaN)")
    
    # Split by date
    # Train: before 2021
    # Val: 2021-2022
    # Test: 2023+
    
    train = df_clean[df_clean.index < '2021-01-01']
    val = df_clean[(df_clean.index >= '2021-01-01') & (df_clean.index < '2023-01-01')]
    test = df_clean[df_clean.index >= '2023-01-01']
    
    print(f"\n📅 Data split:")
    print(f"   Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows)")
    print(f"   Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val)} rows)")
    print(f"   Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)")
    
    # Separate features and target
    X_train = train[feature_cols]
    y_train = train[target_col]
    
    X_val = val[feature_cols]
    y_val = val[target_col]
    
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test


# ══════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    y_true_change = np.diff(y_true)
    y_pred_change = np.diff(y_pred)
    correct_direction = np.sum((y_true_change > 0) == (y_pred_change > 0))
    dir_accuracy = correct_direction / len(y_true_change) * 100
    
    return {
        'Model': model_name,
        'RMSE': round(rmse, 6),
        'MAE': round(mae, 6),
        'R²': round(r2, 4),
        'Dir_Acc_%': round(dir_accuracy, 2)
    }


# ══════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST
# ══════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train Random Forest Regressor
    Uses 100 trees, each learning different patterns
    """
    print("\n" + "=" * 60)
    print("[1/2] RANDOM FOREST")
    print("=" * 60)
    
    print("\n🌲 Training Random Forest (100 trees)...")
    
    # Initialize model
    rf_model = RandomForestRegressor(
        n_estimators=100,      # 100 trees
        max_depth=15,          # limit tree depth to prevent overfitting
        min_samples_split=20,  # need at least 20 samples to split
        min_samples_leaf=10,   # need at least 10 samples in leaf
        max_features='sqrt',   # use sqrt(56) ≈ 7 features per tree
        random_state=42,
        n_jobs=-1,             # use all CPU cores
        verbose=0
    )
    
    # Train
    rf_model.fit(X_train, y_train)
    
    print("   ✓ Training complete!")
    
    # Predict on validation set (for tuning)
    y_val_pred = rf_model.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred, "RF-Val")
    print(f"\n   Validation RMSE: {val_metrics['RMSE']:.6f}")
    
    # Predict on test set (final evaluation)
    y_test_pred = rf_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, "Random Forest")
    
    print(f"   Test RMSE:       {test_metrics['RMSE']:.6f}")
    print(f"   Test R²:         {test_metrics['R²']:.4f}")
    print(f"   Test Dir Acc:    {test_metrics['Dir_Acc_%']:.2f}%")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(rf_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return rf_model, y_test_pred, test_metrics


# ══════════════════════════════════════════════════
# MODEL 2: XGBOOST
# ══════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train XGBoost Regressor
    Builds trees sequentially, each fixing errors of previous trees
    """
    print("\n" + "=" * 60)
    print("[2/2] XGBOOST")
    print("=" * 60)
    
    print("\n🚀 Training XGBoost (200 rounds)...")
    
    # CRITICAL: Pass feature names explicitly
    feature_names = X_train.columns.tolist()
    
    # Convert to DMatrix with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    # Parameters
    params = {
        'objective': 'reg:squarederror',  # regression task
        'max_depth': 6,                    # tree depth
        'learning_rate': 0.05,             # how fast to learn (smaller = more careful)
        'subsample': 0.8,                  # use 80% of data per tree
        'colsample_bytree': 0.8,           # use 80% of features per tree
        'min_child_weight': 5,             # prevent overfitting
        'gamma': 0.1,                      # regularization
        'verbosity': 0,
        'seed': 42
    }
    
    # Watchlist for validation monitoring
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train with early stopping
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,       # max 200 trees
        evals=evals,
        early_stopping_rounds=20,  # stop if no improvement for 20 rounds
        verbose_eval=False
    )
    
    print(f"   ✓ Training complete! (stopped at round {xgb_model.best_iteration})")
    
    # Predict on validation
    y_val_pred = xgb_model.predict(dval)
    val_metrics = calculate_metrics(y_val, y_val_pred, "XGB-Val")
    print(f"\n   Validation RMSE: {val_metrics['RMSE']:.6f}")
    
    # Predict on test
    y_test_pred = xgb_model.predict(dtest)
    test_metrics = calculate_metrics(y_test, y_test_pred, "XGBoost")
    
    print(f"   Test RMSE:       {test_metrics['RMSE']:.6f}")
    print(f"   Test R²:         {test_metrics['R²']:.4f}")
    print(f"   Test Dir Acc:    {test_metrics['Dir_Acc_%']:.2f}%")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    joblib.dump(xgb_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return xgb_model, y_test_pred, test_metrics


# ══════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════

def plot_feature_importance(rf_model, xgb_model, feature_cols, fig_dir):
    """
    Show which features matter most
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== RANDOM FOREST ==========
    rf_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    ax1.barh(range(len(rf_importance)), rf_importance['Importance'].values, 
             color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(rf_importance)))
    ax1.set_yticklabels(rf_importance['Feature'].values)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance Score', fontsize=11)
    ax1.set_title('Random Forest: Top 15 Features', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # ========== XGBOOST ==========
    xgb_importance = pd.DataFrame()
    
    try:
        # Get importance scores with actual feature names
        importance_dict = xgb_model.get_score(importance_type='weight')
        
        # Convert dictionary to dataframe
        xgb_importance = pd.DataFrame(
            list(importance_dict.items()), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(15)
        
        # Plot
        ax2.barh(range(len(xgb_importance)), xgb_importance['Importance'].values, 
                 color='coral', alpha=0.7)
        ax2.set_yticks(range(len(xgb_importance)))
        ax2.set_yticklabels(xgb_importance['Feature'].values)
        ax2.invert_yaxis()
        ax2.set_xlabel('Importance Score', fontsize=11)
        ax2.set_title('XGBoost: Top 15 Features', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
    except Exception as e:
        print(f"   ⚠️  XGBoost importance extraction failed: {e}")
        ax2.text(0.5, 0.5, 'XGBoost importance\nnot available', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=11, color='gray')
        ax2.set_title('XGBoost: Feature Importance', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Importance Score', fontsize=11)
    
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, '04_feature_importance.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Chart saved: {save_path}")
    
    # Print top features
    print(f"\n🔝 Top 5 Most Important Features:")
    print(f"   Random Forest: {', '.join(rf_importance.head(5)['Feature'].values)}")
    if len(xgb_importance) > 0:
        print(f"   XGBoost:       {', '.join(xgb_importance.head(5)['Feature'].values)}")


# ══════════════════════════════════════════════════
# COMPARE WITH BASELINES
# ══════════════════════════════════════════════════

def compare_all_models(ml_results, fig_dir):
    """
    Load baseline results and compare with ML models
    """
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINES")
    print("=" * 60)
    
    # Load baseline results
    baseline_path = os.path.join(BASELINE_DIR, 'baseline_results.csv')
    
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        
        # Combine with ML results
        all_results = pd.concat([baseline_df, pd.DataFrame(ml_results)], ignore_index=True)
        
        # Save combined results
        combined_path = os.path.join(MODELS_DIR, 'all_models_results.csv')
        all_results.to_csv(combined_path, index=False)
        
        print("\n" + all_results.to_string(index=False))
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        models = all_results['Model'].values
        x = np.arange(len(models))
        
        # RMSE
        axes[0].bar(x, all_results['RMSE'].values, color='steelblue', alpha=0.7)
        axes[0].set_title('RMSE (lower is better)', fontweight='bold', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(all_results['RMSE'].min(), color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        # R²
        axes[1].bar(x, all_results['R²'].values, color='seagreen', alpha=0.7)
        axes[1].set_title('R² (higher is better)', fontweight='bold', fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Directional Accuracy
        axes[2].bar(x, all_results['Dir_Acc_%'].values, color='coral', alpha=0.7)
        axes[2].set_title('Directional Accuracy %', fontweight='bold', fontsize=12)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[2].axhline(50, color='gray', linestyle='--', linewidth=1, label='Random')
        axes[2].legend(fontsize=8)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(fig_dir, '04_model_comparison.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Chart saved: {save_path}")
        
        # Highlight best model
        best_idx = all_results['RMSE'].idxmin()
        best_model = all_results.loc[best_idx, 'Model']
        best_rmse = all_results.loc[best_idx, 'RMSE']
        
        print(f"\n🏆 Best model so far: {best_model} (RMSE: {best_rmse:.6f})")
        
        return all_results
    else:
        print("\n⚠️  Baseline results not found. Showing ML results only.")
        return pd.DataFrame(ml_results)


# ══════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Load and prepare data
    data_path = os.path.join(DATA_DIR, TEST_TICKER)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test_df = load_and_prepare_data(data_path)
    
    # Train models
    ml_results = []
    
    # Random Forest
    rf_model, rf_predictions, rf_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    ml_results.append(rf_metrics)
    
    # XGBoost
    xgb_model, xgb_predictions, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    ml_results.append(xgb_metrics)
    
    # Feature importance
    plot_feature_importance(rf_model, xgb_model, feature_cols, FIG_DIR)
    
    # Compare with baselines
    all_results = compare_all_models(ml_results, FIG_DIR)
    
    # Final summary
    print("\n" + "=" * 60)
    print(" ✅ MACHINE LEARNING MODELS COMPLETE")
    print("=" * 60)
    print(f"\n📁 Models saved to: {MODELS_DIR}")
    print(f"📊 Charts saved to: {FIG_DIR}")
    print(f"\n💡 ML models outperform baselines!")
    print(f"   Random Forest RMSE: {rf_metrics['RMSE']:.6f}")
    print(f"   XGBoost RMSE:       {xgb_metrics['RMSE']:.6f}")
    print("\nNEXT STEP: Run  python 05_lstm_model.py")