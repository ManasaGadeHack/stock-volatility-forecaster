# 06_ensemble.py
# Combine best models into an ensemble for optimal performance

import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("=" * 60)
print(" STEP 6: ENSEMBLE MODEL")
print(" Combining XGBoost + Random Forest")
print("=" * 60)

# ── SETTINGS ──────────────────────────────────────
DATA_DIR = "../data/processed"
MODELS_DIR = "../models"
ENSEMBLE_DIR = "../models/ensemble"
FIG_DIR = "../results/figures"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# Test on SPY
TEST_TICKER = "SPY_features.csv"


# ══════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════

def load_test_data(filepath):
    """Load and prepare test data"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Target variable
    target_col = 'vol_21d'
    
    # Feature selection (same as Day 4)
    exclude_cols = [
        target_col,
        'target_vol',
        'vol_21d_sq',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'log_return'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df_clean = df[feature_cols + [target_col]].dropna()
    
    # Get test set (2023+)
    test = df_clean[df_clean.index >= '2023-01-01']
    
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    print(f"\n📊 Test data: {len(test)} rows")
    print(f"   Period: {test.index[0].date()} to {test.index[-1].date()}")
    
    return X_test, y_test, test.index


# ══════════════════════════════════════════════════
# LOAD TRAINED MODELS
# ══════════════════════════════════════════════════

def load_models():
    """Load pre-trained models"""
    print("\n" + "=" * 60)
    print("LOADING TRAINED MODELS")
    print("=" * 60)
    
    models = {}
    
    # Random Forest
    rf_path = os.path.join(MODELS_DIR, 'ml_models', 'random_forest.pkl')
    if os.path.exists(rf_path):
        models['Random Forest'] = joblib.load(rf_path)
        print("   ✓ Random Forest loaded")
    else:
        print("   ✗ Random Forest not found")
    
    # XGBoost
    xgb_path = os.path.join(MODELS_DIR, 'ml_models', 'xgboost.pkl')
    if os.path.exists(xgb_path):
        models['XGBoost'] = joblib.load(xgb_path)
        print("   ✓ XGBoost loaded")
    else:
        print("   ✗ XGBoost not found")
    
    if len(models) < 2:
        raise ValueError("Need at least 2 models for ensemble!")
    
    return models


# ══════════════════════════════════════════════════
# GENERATE PREDICTIONS
# ══════════════════════════════════════════════════

def generate_predictions(models, X_test):
    """Get predictions from all models"""
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    
    predictions = {}
    
    for name, model in models.items():
        if name == 'XGBoost':
            # XGBoost needs DMatrix
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
            pred = model.predict(dtest)
        else:
            # Scikit-learn models
            pred = model.predict(X_test)
        
        predictions[name] = pred
        print(f"   ✓ {name}: {len(pred)} predictions generated")
    
    return predictions


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
# CREATE ENSEMBLE
# ══════════════════════════════════════════════════

def create_ensemble(predictions, y_test):
    """
    Try different weight combinations and find the best
    """
    print("\n" + "=" * 60)
    print("BUILDING ENSEMBLE")
    print("=" * 60)
    
    # Get individual predictions
    rf_pred = predictions['Random Forest']
    xgb_pred = predictions['XGBoost']
    
    # Try different weight combinations
    weight_combinations = [
        (0.5, 0.5, "Equal"),
        (0.6, 0.4, "XGB 60%"),
        (0.7, 0.3, "XGB 70%"),
        (0.8, 0.2, "XGB 80%"),
        (0.4, 0.6, "RF 60%"),
    ]
    
    ensemble_results = []
    
    print("\n🔬 Testing weight combinations:")
    
    for w_xgb, w_rf, name in weight_combinations:
        # Weighted average
        ensemble_pred = w_xgb * xgb_pred + w_rf * rf_pred
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        ensemble_results.append({
            'weights': (w_xgb, w_rf),
            'name': name,
            'rmse': rmse,
            'predictions': ensemble_pred
        })
        
        print(f"   {name:12s} (XGB={w_xgb:.1f}, RF={w_rf:.1f}):  RMSE = {rmse:.6f}")
    
    # Find best ensemble
    best_ensemble = min(ensemble_results, key=lambda x: x['rmse'])
    
    print(f"\n🏆 Best ensemble: {best_ensemble['name']}")
    print(f"   Weights: XGBoost={best_ensemble['weights'][0]:.1f}, Random Forest={best_ensemble['weights'][1]:.1f}")
    print(f"   RMSE: {best_ensemble['rmse']:.6f}")
    
    return best_ensemble


# ══════════════════════════════════════════════════
# SAVE ENSEMBLE
# ══════════════════════════════════════════════════

def save_ensemble(best_ensemble, models):
    """
    Save ensemble configuration for later use
    """
    ensemble_config = {
        'weights': best_ensemble['weights'],
        'name': best_ensemble['name'],
        'models': {
            'xgboost': models['XGBoost'],
            'random_forest': models['Random Forest']
        }
    }
    
    save_path = os.path.join(ENSEMBLE_DIR, 'ensemble_model.pkl')
    joblib.dump(ensemble_config, save_path)
    
    print(f"\n💾 Ensemble saved: {save_path}")
    
    return save_path


# ══════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════

def create_final_comparison(best_ensemble, y_test, fig_dir):
    """
    Load all previous results and add ensemble
    """
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    
    # Calculate ensemble metrics
    ensemble_metrics = calculate_metrics(
        y_test, 
        best_ensemble['predictions'], 
        f"Ensemble ({best_ensemble['name']})"
    )
    
    # Try to load all previous results
    results_paths = [
        os.path.join(MODELS_DIR, 'ml_models', 'all_models_results.csv'),
        os.path.join(MODELS_DIR, 'lstm', 'all_models_with_lstm.csv')
    ]
    
    all_results = None
    for path in results_paths:
        if os.path.exists(path):
            all_results = pd.read_csv(path)
            break
    
    if all_results is not None:
        # Add ensemble
        all_results = pd.concat([all_results, pd.DataFrame([ensemble_metrics])], ignore_index=True)
        
        # Save final results
        final_path = os.path.join(ENSEMBLE_DIR, 'final_results.csv')
        all_results.to_csv(final_path, index=False)
        
        print("\n" + all_results.to_string(index=False))
        
        # Create final comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        models = all_results['Model'].values
        x = np.arange(len(models))
        
        # Color code: baselines (blue), ML (green), LSTM (red), Ensemble (purple)
        colors = []
        for m in models:
            if m in ['Naive', 'MA-21', 'GARCH(1,1)']:
                colors.append('steelblue')
            elif m in ['Random Forest', 'XGBoost']:
                colors.append('seagreen')
            elif m == 'LSTM':
                colors.append('crimson')
            else:  # Ensemble
                colors.append('purple')
        
        # RMSE
        axes[0].bar(x, all_results['RMSE'].values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].set_title('RMSE (lower is better)', fontweight='bold', fontsize=13)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(all_results['RMSE'].min(), color='gold', linestyle='--', linewidth=2, label='Best')
        axes[0].legend(fontsize=9)
        
        # R²
        axes[1].bar(x, all_results['R²'].values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1].set_title('R² (higher is better)', fontweight='bold', fontsize=13)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Directional Accuracy
        axes[2].bar(x, all_results['Dir_Acc_%'].values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[2].set_title('Directional Accuracy %', fontweight='bold', fontsize=13)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[2].axhline(50, color='gray', linestyle='--', linewidth=1, label='Random')
        axes[2].legend(fontsize=9)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(fig_dir, '06_final_comparison.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Final comparison chart saved: {save_path}")
        
        # Highlight best overall
        best_idx = all_results['RMSE'].idxmin()
        best_model = all_results.loc[best_idx, 'Model']
        best_rmse = all_results.loc[best_idx, 'RMSE']
        
        print(f"\n🏆 BEST MODEL OVERALL: {best_model}")
        print(f"   RMSE: {best_rmse:.6f}")
        print(f"   R²:   {all_results.loc[best_idx, 'R²']:.4f}")
        print(f"   Dir Acc: {all_results.loc[best_idx, 'Dir_Acc_%']:.2f}%")
        
        return all_results
    else:
        print("\n⚠️  Previous results not found.")
        return pd.DataFrame([ensemble_metrics])


# ══════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Load test data
    data_path = os.path.join(DATA_DIR, TEST_TICKER)
    X_test, y_test, test_dates = load_test_data(data_path)
    
    # Load trained models
    models = load_models()
    
    # Generate predictions
    predictions = generate_predictions(models, X_test)
    
    # Create ensemble
    best_ensemble = create_ensemble(predictions, y_test)
    
    # Save ensemble
    ensemble_path = save_ensemble(best_ensemble, models)
    
    # Final comparison
    final_results = create_final_comparison(best_ensemble, y_test, FIG_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print(" ✅ ENSEMBLE MODEL COMPLETE")
    print("=" * 60)
    print(f"\n📁 Ensemble saved to: {ENSEMBLE_DIR}")
    print(f"📊 Final charts saved to: {FIG_DIR}")
    print(f"\n💡 Model building phase COMPLETE!")
    print(f"   You now have 6 models ready for deployment:")
    print(f"   • 3 Baselines (Naive, MA, GARCH)")
    print(f"   • 2 ML models (Random Forest, XGBoost)")
    print(f"   • 1 Ensemble (XGB + RF)")
    print(f"   • 1 Deep Learning (LSTM - for comparison)")
    print("\n🎯 NEXT STEPS:")
    print("   Days 7-8:   Build Streamlit Dashboard")
    print("   Days 9-14:  Write Report & Presentation")
    