# 05_lstm_model.py
# LSTM (Long Short-Term Memory) Neural Network for volatility prediction

import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print(" STEP 5: DEEP LEARNING MODEL")
print(" LSTM Neural Network")
print("=" * 60)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ── SETTINGS ──────────────────────────────────────
DATA_DIR = "../data/processed"
MODELS_DIR = "../models/lstm"
ML_MODELS_DIR = "../models/ml_models"
FIG_DIR = "../results/figures"

os.makedirs(MODELS_DIR, exist_ok=True)

# Test on SPY (S&P 500)
TEST_TICKER = "SPY_features.csv"

# LSTM hyperparameters
SEQUENCE_LENGTH = 30  # Use last 30 days to predict
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ══════════════════════════════════════════════════
# LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════

def load_and_prepare_data(filepath):
    """
    Load data and prepare for LSTM (needs 3D input)
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"\n📊 Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Target variable
    target_col = 'vol_21d'
    
    # Feature selection
    exclude_cols = [
        target_col,
        'target_vol',
        'vol_21d_sq',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'log_return'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"🎯 Target: {target_col}")
    print(f"📈 Features: {len(feature_cols)} selected")
    
    # Clean data
    df_clean = df[feature_cols + [target_col]].dropna()
    
    print(f"✅ Clean data: {len(df_clean)} rows")
    
    # Split by date
    train = df_clean[df_clean.index < '2021-01-01']
    val = df_clean[(df_clean.index >= '2021-01-01') & (df_clean.index < '2023-01-01')]
    test = df_clean[df_clean.index >= '2023-01-01']
    
    print(f"\n📅 Data split:")
    print(f"   Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows)")
    print(f"   Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val)} rows)")
    print(f"   Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)")
    
    return train, val, test, feature_cols, target_col


# ══════════════════════════════════════════════════
# CREATE SEQUENCES FOR LSTM
# ══════════════════════════════════════════════════

def create_sequences(data, feature_cols, target_col, seq_length):
    """
    Convert data into sequences for LSTM
    
    Input: Daily data
    Output: Sequences of seq_length days
    
    Example: if seq_length=30, each sample contains last 30 days of features
    """
    X_sequences = []
    y_sequences = []
    
    features = data[feature_cols].values
    targets = data[target_col].values
    
    # Create overlapping sequences
    for i in range(seq_length, len(data)):
        # Take last seq_length days as input
        X_sequences.append(features[i-seq_length:i])
        # Take current day's target as output
        y_sequences.append(targets[i])
    
    return np.array(X_sequences), np.array(y_sequences)


# ══════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════

def preprocess_data(train, val, test, feature_cols, target_col, seq_length):
    """
    Scale features and create sequences
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING FOR LSTM")
    print("=" * 60)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    
    # Transform all datasets
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    print(f"\n📏 Features scaled (mean=0, std=1)")
    
    # Create sequences
    print(f"🔄 Creating sequences (length={seq_length})...")
    
    X_train, y_train = create_sequences(train_scaled, feature_cols, target_col, seq_length)
    X_val, y_val = create_sequences(val_scaled, feature_cols, target_col, seq_length)
    X_test, y_test = create_sequences(test_scaled, feature_cols, target_col, seq_length)
    
    print(f"\n✅ Sequences created:")
    print(f"   Train: {X_train.shape} → {y_train.shape}")
    print(f"   Val:   {X_val.shape} → {y_val.shape}")
    print(f"   Test:  {X_test.shape} → {y_test.shape}")
    
    # Save scaler for later use
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n💾 Scaler saved: {scaler_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ══════════════════════════════════════════════════
# BUILD LSTM MODEL
# ══════════════════════════════════════════════════

def build_lstm_model(input_shape, learning_rate):
    """
    Build LSTM neural network
    
    Architecture:
    - LSTM layer 1: 64 units, returns sequences
    - Dropout: 30% (prevents overfitting)
    - LSTM layer 2: 32 units
    - Dropout: 30%
    - Dense output: 1 unit (volatility prediction)
    """
    model = Sequential([
        # First LSTM layer
        LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        Dropout(0.3, name='dropout_1'),
        
        # Second LSTM layer
        LSTM(32, return_sequences=False, name='lstm_2'),
        Dropout(0.3, name='dropout_2'),
        
        # Output layer
        Dense(1, name='output')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Also track Mean Absolute Error
    )
    
    return model


# ══════════════════════════════════════════════════
# TRAIN LSTM
# ══════════════════════════════════════════════════

def train_lstm(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    """
    Train LSTM with early stopping
    """
    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_length, n_features)
    model = build_lstm_model(input_shape, learning_rate)
    
    print(f"\n🏗️  Model architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    print(f"\n🚀 Training started...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print(f"\n✓ Training complete!")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    return model, history


# ══════════════════════════════════════════════════
# EVALUATION
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


def evaluate_lstm(model, X_test, y_test):
    """
    Evaluate LSTM on test set
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Predict
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, "LSTM")
    
    print(f"\n📊 Test Results:")
    print(f"   RMSE:       {metrics['RMSE']:.6f}")
    print(f"   MAE:        {metrics['MAE']:.6f}")
    print(f"   R²:         {metrics['R²']:.4f}")
    print(f"   Dir Acc:    {metrics['Dir_Acc_%']:.2f}%")
    
    return y_pred, metrics


# ══════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════

def plot_training_history(history, fig_dir):
    """
    Plot loss curves during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, '05_lstm_training.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Training history chart saved: {save_path}")


def compare_with_ml_models(lstm_metrics, fig_dir):
    """
    Load ML model results and compare with LSTM
    """
    print("\n" + "=" * 60)
    print("COMPARISON WITH ALL MODELS")
    print("=" * 60)
    
    # Load previous results
    ml_results_path = os.path.join(ML_MODELS_DIR, 'all_models_results.csv')
    
    if os.path.exists(ml_results_path):
        all_results = pd.read_csv(ml_results_path)
        
        # Add LSTM
        all_results = pd.concat([all_results, pd.DataFrame([lstm_metrics])], ignore_index=True)
        
        # Save updated results
        updated_path = os.path.join(MODELS_DIR, 'all_models_with_lstm.csv')
        all_results.to_csv(updated_path, index=False)
        
        print("\n" + all_results.to_string(index=False))
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        models = all_results['Model'].values
        x = np.arange(len(models))
        
        # RMSE
        colors = ['steelblue'] * (len(models) - 1) + ['purple']  # LSTM in purple
        axes[0].bar(x, all_results['RMSE'].values, color=colors, alpha=0.7)
        axes[0].set_title('RMSE (lower is better)', fontweight='bold', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(all_results['RMSE'].min(), color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        # R²
        axes[1].bar(x, all_results['R²'].values, color=colors, alpha=0.7)
        axes[1].set_title('R² (higher is better)', fontweight='bold', fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Directional Accuracy
        axes[2].bar(x, all_results['Dir_Acc_%'].values, color=colors, alpha=0.7)
        axes[2].set_title('Directional Accuracy %', fontweight='bold', fontsize=12)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[2].axhline(50, color='gray', linestyle='--', linewidth=1, label='Random')
        axes[2].legend(fontsize=8)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(fig_dir, '05_all_models_comparison.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Comparison chart saved: {save_path}")
        
        # Highlight best model
        best_idx = all_results['RMSE'].idxmin()
        best_model = all_results.loc[best_idx, 'Model']
        best_rmse = all_results.loc[best_idx, 'RMSE']
        
        print(f"\n🏆 Best model overall: {best_model} (RMSE: {best_rmse:.6f})")
        
        return all_results
    else:
        print("\n⚠️  ML results not found. Showing LSTM only.")
        return pd.DataFrame([lstm_metrics])


# ══════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Load data
    data_path = os.path.join(DATA_DIR, TEST_TICKER)
    train, val, test, feature_cols, target_col = load_and_prepare_data(data_path)
    
    # Preprocess and create sequences
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(
        train, val, test, feature_cols, target_col, SEQUENCE_LENGTH
    )
    
    # Build and train LSTM
    model, history = train_lstm(
        X_train, y_train, X_val, y_val, 
        EPOCHS, BATCH_SIZE, LEARNING_RATE
    )
    
    # Evaluate
    y_pred, lstm_metrics = evaluate_lstm(model, X_test, y_test)
    
    # Visualizations
    plot_training_history(history, FIG_DIR)
    all_results = compare_with_ml_models(lstm_metrics, FIG_DIR)
    
    # Final summary
    print("\n" + "=" * 60)
    print(" ✅ LSTM MODEL COMPLETE")
    print("=" * 60)
    print(f"\n📁 Model saved to: {MODELS_DIR}")
    print(f"📊 Charts saved to: {FIG_DIR}")
    print(f"\n💡 LSTM Results:")
    print(f"   RMSE: {lstm_metrics['RMSE']:.6f}")
    print(f"   R²:   {lstm_metrics['R²']:.4f}")
    print("\nNEXT STEP: Run  python 06_ensemble.py")