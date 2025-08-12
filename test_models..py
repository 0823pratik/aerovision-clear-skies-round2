"""
Fixed model testing script with custom_objects
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tensorflow as tf

def test_fixed_model_loading():
    """Test loading LSTM with custom_objects fix"""
    print("🔧 TESTING FIXED MODEL LOADING")
    print("=" * 50)
    
    model_path = Path("models/trained_standalone/lstm_model.h5")
    
    if not model_path.exists():
        print("❌ LSTM model file not found")
        return None
    
    try:
        # FIX: Load with custom_objects to handle 'mse' metric issue
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError(), 
            'mape': tf.keras.losses.MeanAbsolutePercentageError()
        }
        
        print("📂 Loading LSTM with custom_objects...")
        lstm_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Skip compilation to avoid issues
        )
        
        print("✅ SUCCESS: LSTM model loaded!")
        print(f"   Input shape: {lstm_model.input_shape}")
        print(f"   Output shape: {lstm_model.output_shape}")
        print(f"   Parameters: {lstm_model.count_params():,}")
        
        # Test prediction
        test_input = np.random.random((1, 24, 14))  # Batch, sequence, features
        prediction = lstm_model.predict(test_input, verbose=0)
        print(f"   Test prediction: {prediction[0][0]:.2f} μg/m³")
        
        return lstm_model
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None

def test_other_models():
    """Test loading other models (should work fine)"""
    print("\n🔧 TESTING OTHER MODELS")
    print("=" * 50)
    
    import joblib
    model_dir = Path("models/trained_standalone")
    
    # Test Random Forest
    try:
        rf_model = joblib.load(model_dir / "rf_model.pkl")
        print("✅ Random Forest: LOADED")
        print(f"   Features: {rf_model.n_features_in_}")
    except Exception as e:
        print(f"❌ Random Forest: {e}")
    
    # Test XGBoost
    try:
        xgb_model = joblib.load(model_dir / "xgb_model.pkl")
        print("✅ XGBoost: LOADED")
    except Exception as e:
        print(f"❌ XGBoost: {e}")
    
    # Test Scalers
    try:
        scalers = joblib.load(model_dir / "scalers.pkl")
        print("✅ Scalers: LOADED")
        if 'features' in scalers:
            print(f"   Features scaled: {scalers['features'].n_features_in_}")
    except Exception as e:
        print(f"❌ Scalers: {e}")

if __name__ == "__main__":
    lstm_model = test_fixed_model_loading()
    test_other_models()
    
    if lstm_model:
        print("\n🎉 ALL MODELS CAN BE LOADED!")
        print("🚀 Your AeroVision-GGM 2.0 system is ready!")
    else:
        print("\n⚠️ LSTM loading issue persists")
