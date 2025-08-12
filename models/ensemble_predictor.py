# models/ensemble_predictor.py
"""
Advanced ensemble prediction system with uncertainty quantification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIGS

class EnsembleAirQualityPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_importance = None
        self.model_weights = MODEL_CONFIGS['ensemble']['weights']
        
    def create_lstm_model(self, input_shape):
        """Create advanced LSTM model with regularization"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(50, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(25, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_sequences(self, data, target, seq_length=24):
        """Prepare sequences for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(target[i + seq_length])
            
        return np.array(X), np.array(y)
    
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=MODEL_CONFIGS['ensemble']['rf_estimators'],
            random_state=42,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
        print(f"RF Cross-validation RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std() * 2):.2f})")
        
        # Train final model
        rf_model.fit(X, y)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': range(len(rf_model.feature_importances_)),
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return rf_model
    
    def train_xgboost(self, X, y):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=MODEL_CONFIGS['ensemble']['xgb_estimators'],
            random_state=42,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        # Train model
        xgb_model.fit(X, y, verbose=False)
        
        return xgb_model
    
    def train_lstm(self, X, y, validation_split=0.2):
        """Train LSTM model"""
        print("Training LSTM...")
        
        # Prepare sequences
        seq_length = MODEL_CONFIGS['lstm']['sequence_length']
        X_seq, y_seq = self.prepare_sequences(X, y, seq_length)
        
        if len(X_seq) == 0:
            print("Not enough data for LSTM training")
            return None
        
        # Create and train model
        lstm_model = self.create_lstm_model((seq_length, X.shape[1]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = lstm_model.fit(
            X_seq, y_seq,
            epochs=MODEL_CONFIGS['lstm']['epochs'],
            batch_size=MODEL_CONFIGS['lstm']['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return lstm_model
    
    def train_ensemble(self, feature_data, target_data):
        """Train complete ensemble"""
        print("Starting ensemble training...")
        
        # Prepare data
        X = feature_data.values
        y = target_data.values
        
        # Remove rows with NaN
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # FIXED: Handle case where all X values might have NaN
        if len(X) == 0:
            raise ValueError("No valid training data available")
        
        # Remove any remaining NaN from features
        nan_mask = ~np.isnan(X).any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data after removing NaN features")
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Train individual models
        self.models['rf'] = self.train_random_forest(X_scaled, y)
        self.models['xgb'] = self.train_xgboost(X_scaled, y)
        self.models['lstm'] = self.train_lstm(X_scaled, y)
        
        self.is_trained = True
        print("Ensemble training complete!")
        
        return self.evaluate_ensemble(X_scaled, y)
    
    def predict_single(self, features):
        """Make prediction for single sample"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        # Handle NaN in features
        if np.isnan(features).any():
            # Fill NaN with mean values from training
            features = np.where(np.isnan(features), 0, features)
        
        # Scale features
        features_scaled = self.scalers['features'].transform(features.reshape(1, -1))
        
        predictions = {}
        
        # Random Forest prediction
        if 'rf' in self.models and self.models['rf'] is not None:
            predictions['rf'] = self.models['rf'].predict(features_scaled)[0]
        
        # XGBoost prediction
        if 'xgb' in self.models and self.models['xgb'] is not None:
            predictions['xgb'] = self.models['xgb'].predict(features_scaled)[0]
        
        # LSTM prediction (if available)
        if 'lstm' in self.models and self.models['lstm'] is not None:
            seq_length = MODEL_CONFIGS['lstm']['sequence_length']
            if features_scaled.shape[1] >= seq_length:
                # Use the last seq_length features as a sequence
                lstm_input = features_scaled[0, -seq_length:].reshape(1, seq_length, 1)
                predictions['lstm'] = self.models['lstm'].predict(lstm_input, verbose=0)[0][0]
            else:
                # If not enough features, use available models only
                if predictions:
                    predictions['lstm'] = np.mean(list(predictions.values()))
        
        return predictions
    
    def predict_ensemble(self, features):
        """Make ensemble prediction with uncertainty"""
        predictions = self.predict_single(features)
        
        if not predictions:
            return {
                'prediction': 0,
                'confidence_lower': 0,
                'confidence_upper': 0,
                'uncertainty': 0,
                'model_agreement': 0,
                'individual_predictions': {}
            }
        
        # Calculate weighted ensemble prediction
        if len(predictions) == 3:  # All models available
            ensemble_pred = (
                self.model_weights[0] * predictions.get('rf', 0) +
                self.model_weights[1] * predictions.get('xgb', 0) +
                self.model_weights[2] * predictions.get('lstm', 0)
            )
        elif len(predictions) >= 2:  # Two models available
            ensemble_pred = np.mean(list(predictions.values()))
        else:
            ensemble_pred = list(predictions.values())[0]
        
        # Calculate uncertainty (standard deviation of predictions)
        pred_values = list(predictions.values())
        uncertainty = np.std(pred_values) if len(pred_values) > 1 else 0
        
        # Confidence intervals (assuming normal distribution)
        confidence_lower = max(0, ensemble_pred - 1.96 * uncertainty)
        confidence_upper = ensemble_pred + 1.96 * uncertainty
        
        # Model agreement score
        if len(pred_values) > 1:
            agreement = 1 - (uncertainty / max(ensemble_pred, 1))
            agreement = max(0, min(1, agreement))  # Clamp between 0 and 1
        else:
            agreement = 1.0
        
        return {
            'prediction': ensemble_pred,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'uncertainty': uncertainty,
            'model_agreement': agreement,
            'individual_predictions': predictions
        }
    
    def predict_batch(self, feature_batch):
        """Make batch predictions"""
        results = []
        
        for i in range(len(feature_batch)):
            result = self.predict_ensemble(feature_batch.iloc[i].values)
            results.append(result)
        
        return results
    
    def evaluate_ensemble(self, X, y):
        """Evaluate ensemble performance"""
        predictions = []
        
        for i in range(len(X)):
            pred_result = self.predict_ensemble(X[i])
            predictions.append(pred_result['prediction'])
        
        predictions = np.array(predictions)
        
        # Handle edge case where predictions might be empty
        if len(predictions) == 0 or len(y) == 0:
            return {
                'RMSE': 0,
                'MAE': 0,
                'MAPE': 0,
                'R2': 0,
                'MSE': 0
            }
        
        # Calculate metrics
        mse = np.mean((y - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        
        # Handle MAPE calculation to avoid division by zero
        mape_values = np.abs((y - predictions) / np.where(y == 0, 1e-8, y)) * 100
        mape = np.mean(mape_values)
        
        # R-squared
        ss_res = np.sum((y - predictions)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MSE': mse
        }
        
        print("Ensemble Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        
        return metrics
    
    def save_checkpoint(self, model_dir, epoch=None):
        """Save training checkpoint"""
        try:
            from pathlib import Path
            checkpoint_dir = Path(model_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save current state
            if hasattr(self, 'models') and self.models:
                checkpoint_name = f"checkpoint_epoch_{epoch}" if epoch else "checkpoint_latest"
                
                # Save sklearn models
                if 'rf' in self.models and self.models['rf'] is not None:
                    joblib.dump(self.models['rf'], checkpoint_dir / f"{checkpoint_name}_rf.pkl")
                
                if 'xgb' in self.models and self.models['xgb'] is not None:
                    joblib.dump(self.models['xgb'], checkpoint_dir / f"{checkpoint_name}_xgb.pkl")
                
                # Save LSTM if available
                if 'lstm' in self.models and self.models['lstm'] is not None:
                    self.models['lstm'].save(checkpoint_dir / f"{checkpoint_name}_lstm.h5")
                
                # Save scalers
                if hasattr(self, 'scalers') and self.scalers:
                    joblib.dump(self.scalers, checkpoint_dir / f"{checkpoint_name}_scalers.pkl")
                    
            print(f"✅ Checkpoint saved at epoch {epoch}")
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            return False
    
    def save_models(self, model_dir):
        """Save trained models"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        try:
            # Save sklearn models
            if 'rf' in self.models and self.models['rf'] is not None:
                joblib.dump(self.models['rf'], model_dir / "rf_model.pkl")
            
            if 'xgb' in self.models and self.models['xgb'] is not None:
                joblib.dump(self.models['xgb'], model_dir / "xgb_model.pkl")
            
            # Save LSTM model
            if 'lstm' in self.models and self.models['lstm'] is not None:
                self.models['lstm'].save(model_dir / "lstm_model.h5")
            
            # Save scalers
            if hasattr(self, 'scalers') and self.scalers:
                joblib.dump(self.scalers, model_dir / "scalers.pkl")
            
            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)
            
            print(f"Models saved to {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, model_dir):
        """Load pre-trained models with fixed custom_objects"""
        try:
            from pathlib import Path
            import tensorflow as tf
            model_dir = Path(model_dir)
            
            # Load sklearn models (these work fine)
            rf_path = model_dir / "rf_model.pkl"
            if rf_path.exists():
                self.models['rf'] = joblib.load(rf_path)
                print("✅ Random Forest loaded!")
            
            xgb_path = model_dir / "xgb_model.pkl"
            if xgb_path.exists():
                self.models['xgb'] = joblib.load(xgb_path)
                print("✅ XGBoost loaded!")
            
            # Load LSTM model with custom_objects fix
            lstm_path = model_dir / "lstm_model.h5"
            if lstm_path.exists():
                custom_objects = {
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.losses.MeanAbsoluteError(), 
                    'mape': tf.keras.losses.MeanAbsolutePercentageError()
                }
                
                self.models['lstm'] = tf.keras.models.load_model(
                    lstm_path,
                    custom_objects=custom_objects,
                    compile=False  # Skip compilation to avoid issues
                )
                print("✅ LSTM loaded!")
            
            # Load scalers
            scalers_path = model_dir / "scalers.pkl"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
                print("✅ Scalers loaded!")
            
            # Load feature importance
            importance_path = model_dir / "feature_importance.csv"
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path)
            
            if self.models:
                self.is_trained = True
                print("🎉 All models loaded successfully!")
                return True
            else:
                print("❌ No models found to load")
                return False
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            self.is_trained = False
            return False
