"""
Configuration for AeroVision-GGM 2.0 Milestone 1
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Gurugram focus (from your 3-city dataset)
TARGET_CITIES = ['Gurugram', 'Gurgaon']  # Handle both spellings

# Model settings (Fixed structure to match ensemble_predictor.py)
# Updated config.py recommendation
MODEL_CONFIGS = {
    'ensemble': {
        'rf_estimators': 100,
        'xgb_estimators': 100,
        'weights': [0.4, 0.4, 0.2]  # RF, XGB, LSTM
    },
    'lstm': {
        'sequence_length': 24,
        'epochs': 25,              # ← Reduced from 50
        'batch_size': 32
    }
}


# Dashboard settings
DASHBOARD_CONFIG = {
    'refresh_interval': 300,
    'map_zoom': 12,
    'gurugram_center': [28.4595, 77.0266]
}
